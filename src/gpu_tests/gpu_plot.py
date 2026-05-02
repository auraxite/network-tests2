#!/usr/bin/env python3

from __future__ import annotations

"""Build line plots bytes -> latency for one src/dst pair from gpu logs.

Arguments:
  input            .txt file or directory with .txt files
  -o, --out        output directory for PNG/CSV files
  --src, --dst     pair spec: rank number or axis label like 0.0
  --metric         avg_us | med_us | min_us | max_us | var_us | std_us
"""

import argparse
import csv
import math
import re
import sys
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

import gpu_heatmap as gh


KIB = 1024
MIB = 1024 * 1024
GIB = 1024 * 1024 * 1024


def format_bytes_human(nbytes: int) -> str:
	if nbytes >= GIB and nbytes % GIB == 0:
		return f"{nbytes // GIB} GiB"
	if nbytes >= MIB and nbytes % MIB == 0:
		return f"{nbytes // MIB} MiB"
	if nbytes >= KIB and nbytes % KIB == 0:
		return f"{nbytes // KIB} KiB"
	return f"{nbytes} B"


def pair_display_label(meta: dict[str, Any], rank: int) -> str:
	n = meta.get("ranks")
	if isinstance(n, int):
		rank_hosts = gh.rank_hosts_from_meta(meta, n)
		if rank_hosts:
			labels = gh.axis_labels_by_node(rank_hosts)
			if 0 <= rank < len(labels):
				return labels[rank]
	return str(rank)


def pair_metric_value(
	meta: dict[str, Any],
	pairs: list[tuple[int, int, list[float]]],
	src_spec: str,
	dst_spec: str,
	metric: str,
) -> tuple[int, int, float | None]:
	src_rank = gh.pair_label_to_rank(src_spec.strip(), meta)
	dst_rank = gh.pair_label_to_rank(dst_spec.strip(), meta)
	if src_rank is None or dst_rank is None:
		return -1, -1, None

	idx = gh.METRIC_KEYS.index(metric)
	value: float | None = None
	for src, dst, vals in pairs:
		if src != src_rank or dst != dst_rank:
			continue
		if idx >= len(vals):
			continue
		candidate = float(vals[idx])
		if math.isnan(candidate):
			continue
		value = candidate
	return src_rank, dst_rank, value


def pair_series_label(meta: dict[str, Any]) -> str:
	mode = str(meta.get("mode", "unknown"))
	env = str(meta.get("env", "unknown"))
	timer = str(meta.get("timer", "unknown"))
	return f"{mode} / {env} / {timer}"


def sanitize_token(text: str) -> str:
	return re.sub(r"[^0-9A-Za-z_.-]+", "_", text.strip()) or "na"


def render_pair_plot_from_paths(
	txt_paths: list[Path],
	out_dir: Path,
	*,
	src_spec: str,
	dst_spec: str,
	metric: str,
) -> int:
	if not txt_paths:
		print("gpu_plot: no input .txt files.", file=sys.stderr)
		return 1

	series: dict[str, dict[int, list[float]]] = {}
	first_meta: dict[str, Any] | None = None
	resolved_src_rank: int | None = None
	resolved_dst_rank: int | None = None
	skipped = 0

	for txt in txt_paths:
		text = txt.read_text(encoding="utf-8", errors="replace")
		meta, pairs = gh.parse_gpu_one_to_one_text(text)
		nbytes = meta.get("bytes")
		if not isinstance(nbytes, int) or nbytes <= 0:
			print(f"gpu_plot: [{txt}] missing valid Bytes: header, skip.", file=sys.stderr)
			skipped += 1
			continue

		src_rank, dst_rank, value = pair_metric_value(meta, pairs, src_spec, dst_spec, metric)
		if value is None:
			print(
				f"gpu_plot: [{txt}] no value for pair {src_spec}->{dst_spec} metric={metric}, skip.",
				file=sys.stderr,
			)
			skipped += 1
			continue

		if first_meta is None:
			first_meta = meta
			resolved_src_rank = src_rank
			resolved_dst_rank = dst_rank
		elif resolved_src_rank != src_rank or resolved_dst_rank != dst_rank:
			print(
				f"gpu_plot: [{txt}] pair spec resolves to another rank pair, skip.",
				file=sys.stderr,
			)
			skipped += 1
			continue

		series_label = pair_series_label(meta)
		series.setdefault(series_label, {}).setdefault(nbytes, []).append(value)

	if first_meta is None or resolved_src_rank is None or resolved_dst_rank is None:
		print("gpu_plot: could not collect any data for the requested pair.", file=sys.stderr)
		return 1

	if not series:
		print("gpu_plot: no series to plot after filtering.", file=sys.stderr)
		return 1

	out_dir.mkdir(parents=True, exist_ok=True)
	src_label = pair_display_label(first_meta, resolved_src_rank)
	dst_label = pair_display_label(first_meta, resolved_dst_rank)
	metric_title = gh.METRIC_TITLE.get(metric, metric)
	metric_unit = gh.METRIC_CBAR_LABEL.get(metric, "мкс")
	creation_time = gh.generation_timestamp()

	fig, ax = plt.subplots(figsize=(8.8, 5.6))
	cmap = mpl.colormaps["tab10"] if hasattr(mpl, "colormaps") else mpl.cm.get_cmap("tab10")
	csv_rows: list[dict[str, Any]] = []

	for idx, series_label in enumerate(sorted(series)):
		points = series[series_label]
		xs = sorted(points)
		ys = [float(np.mean(points[x])) for x in xs]
		yerr = [
			float(np.std(points[x], ddof=1)) if len(points[x]) > 1 else 0.0
			for x in xs
		]
		color = cmap(idx % max(1, getattr(cmap, "N", 10)))
		if any(err > 0.0 for err in yerr):
			ax.errorbar(
				xs,
				ys,
				yerr=yerr,
				marker="o",
				markersize=5,
				linewidth=1.8,
				capsize=3,
				label=series_label,
				color=color,
			)
		else:
			ax.plot(
				xs,
				ys,
				marker="o",
				markersize=5,
				linewidth=1.8,
				label=series_label,
				color=color,
			)

		for x, y, err in zip(xs, ys, yerr):
			csv_rows.append(
				{
					"series": series_label,
					"bytes": x,
					"bytes_human": format_bytes_human(x),
					"value": y,
					"stddev": err,
					"samples": len(points[x]),
				}
			)

	ax.set_title(
		"\n".join(
			[
				f"{metric_title} для пары {src_label} -> {dst_label}",
				"Зависимость задержки от размера сообщения",
			]
		),
		fontsize=11,
	)
	ax.set_xlabel("Размер сообщения", fontsize=10)
	ax.set_ylabel(f"{metric_title} ({metric_unit})", fontsize=10)
	ax.grid(True, which="major", color="#d4d4d8", linewidth=0.8, alpha=0.9)
	ax.xaxis.set_major_formatter(
		FuncFormatter(lambda x, pos: format_bytes_human(int(x)) if x >= 1 else "0 B")
	)
	ax.ticklabel_format(style="plain", axis="x", useOffset=False)
	if len(series) > 1:
		ax.legend(fontsize=9)
	param_text = "\n".join(
		[
			f"src={src_label} ({resolved_src_rank})  dst={dst_label} ({resolved_dst_rank})",
			f"metric={metric}",
			f"файлов: {len(txt_paths)}  пропущено: {skipped}",
			f"сгенерировано: {creation_time}",
		]
	)
	ax.text(
		0.01,
		0.01,
		param_text,
		transform=ax.transAxes,
		fontsize=8,
		verticalalignment="bottom",
		bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "#d4d4d8"},
	)
	fig.tight_layout()

	base_name = (
		f"pair_src{sanitize_token(src_label)}"
		f"_dst{sanitize_token(dst_label)}"
		f"_{gh.METRIC_FILE_STEM.get(metric, metric)}_vs_bytes"
	)
	png_path = out_dir / f"{base_name}.png"
	csv_path = out_dir / f"{base_name}.csv"
	fig.savefig(png_path, dpi=150, bbox_inches="tight")
	plt.close(fig)

	with csv_path.open("w", encoding="utf-8", newline="") as f:
		writer = csv.DictWriter(
			f,
			fieldnames=["series", "bytes", "bytes_human", "value", "stddev", "samples"],
		)
		writer.writeheader()
		writer.writerows(csv_rows)

	print(png_path)
	print(csv_path)
	return 0


def main() -> int:
	p = argparse.ArgumentParser(
		description="Line plots bytes -> latency from gpu text output (pair ... lines).",
		epilog=(
			"Arguments:\n"
			"  input            .txt file or directory with .txt files\n"
			"  -o, --out        output directory for PNG/CSV files\n"
			"  --src, --dst     pair spec: rank number or axis label like 0.0\n"
			"  --metric         avg_us | med_us | min_us | max_us | var_us | std_us\n\n"
		),
		formatter_class=argparse.RawTextHelpFormatter,
	)
	p.add_argument(
		"input",
		help="Файл .txt или каталог с .txt файлами",
	)
	p.add_argument(
		"-o",
		"--out",
		dest="out_dir",
		type=Path,
		default=Path("."),
		help="Каталог для PNG/CSV",
	)
	p.add_argument(
		"--src",
		required=True,
		help="Источник пары: rank или подпись оси, например 0 или 0.0",
	)
	p.add_argument(
		"--dst",
		required=True,
		help="Приёмник пары: rank или подпись оси, например 1 или 0.1",
	)
	p.add_argument(
		"--metric",
		choices=gh.METRIC_KEYS,
		default="med_us",
		help="Метрика графика (по умолчанию med_us)",
	)
	args = p.parse_args()

	path = Path(args.input)
	if not path.exists():
		print(f"gpu_plot: path not found: {path.resolve()}", file=sys.stderr)
		return 1

	if path.is_dir():
		txts = sorted(path.glob("*.txt"))
		if not txts:
			print(f"gpu_plot: no *.txt in {path}", file=sys.stderr)
			return 1
		return render_pair_plot_from_paths(
			txts,
			args.out_dir,
			src_spec=args.src,
			dst_spec=args.dst,
			metric=args.metric,
		)

	if path.is_file():
		return render_pair_plot_from_paths(
			[path],
			args.out_dir,
			src_spec=args.src,
			dst_spec=args.dst,
			metric=args.metric,
		)

	print(f"gpu_plot: not a file or directory: {path}", file=sys.stderr)
	return 1


if __name__ == "__main__":
	raise SystemExit(main())
