#!/usr/bin/env python3

from __future__ import annotations

"""Build line plots bytes -> latency for one src/dst pair from gpu logs.

X axis uses a logarithmic scale with base 2 (log₂).

Arguments:
  input            .txt file or directory with .txt files
  -o, --out        output directory for PNG/CSV files
  --src, --dst     axis label only, for example 28.0
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


def detect_transitions(
	xs: list[int], ys: list[float], *, rel_threshold: float = 0.15
) -> list[dict[str, float]]:
	"""Detect sharp adjacent changes (possible protocol switch points)."""
	events: list[dict[str, float]] = []
	for i in range(1, min(len(xs), len(ys))):
		x_prev = xs[i - 1]
		x_cur = xs[i]
		y_prev = ys[i - 1]
		y_cur = ys[i]
		if not np.isfinite(y_prev) or not np.isfinite(y_cur) or y_prev == 0.0:
			continue
		rel = (y_cur - y_prev) / y_prev
		if abs(rel) < rel_threshold:
			continue
		events.append(
			{
				"from_bytes": float(x_prev),
				"to_bytes": float(x_cur),
				"from_value": float(y_prev),
				"to_value": float(y_cur),
				"delta_value": float(y_cur - y_prev),
				"rel_change_pct": float(rel * 100.0),
			}
		)
	return events


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
	src_rank = resolve_axis_pair_label(src_spec, meta)
	dst_rank = resolve_axis_pair_label(dst_spec, meta)
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


def plot_series_label(meta: dict[str, Any]) -> str:
	"""Legend label without the default MPI timer noise."""
	env = str(meta.get("env", "unknown"))
	timer = str(meta.get("timer", "unknown"))
	if timer in ("", "mpi", "unknown"):
		return env
	return f"{env} / {timer}"


def mode_title(mode: str) -> str:
	return mode.replace("_", " ")


def sanitize_token(text: str) -> str:
	return re.sub(r"[^0-9A-Za-z_.-]+", "_", text.strip()) or "na"


AXIS_LABEL_RE = re.compile(r"^[0-9A-Za-z_.-]+\.[0-9]+$")


def resolve_axis_pair_label(label: str, meta: dict[str, Any]) -> int | None:
	"""Resolve only axis labels like 28.0, intentionally rejecting raw ranks."""
	token = label.strip()
	if not AXIS_LABEL_RE.fullmatch(token):
		return None
	return gh.pair_label_to_rank(token, meta)


def infer_nbytes_from_path(txt_path: Path) -> int | None:
	"""Fallback for logs that missed the Bytes: header but keep _bNNN_ in filename."""
	match = gh.BYTES_TAG_RE.search(txt_path.stem)
	if not match:
		return None
	try:
		nbytes = int(match.group(1))
	except ValueError:
		return None
	return nbytes if nbytes > 0 else None


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

	grouped_series: dict[str, dict[str, dict[int, list[float]]]] = {}
	group_envs: dict[str, set[str]] = {}
	first_meta: dict[str, Any] | None = None
	resolved_src_rank: int | None = None
	resolved_dst_rank: int | None = None
	skipped = 0

	for txt in txt_paths:
		text = txt.read_text(encoding="utf-8", errors="replace")
		meta, pairs = gh.parse_gpu_one_to_one_text(text)
		nbytes = meta.get("bytes")
		if not isinstance(nbytes, int) or nbytes <= 0:
			nbytes = infer_nbytes_from_path(txt)
			if nbytes is not None:
				meta["bytes"] = nbytes
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

		mode = str(meta.get("mode", "unknown"))
		env = str(meta.get("env", "unknown"))
		series_label = plot_series_label(meta)
		grouped_series.setdefault(mode, {}).setdefault(series_label, {}).setdefault(
			nbytes, []
		).append(value)
		group_envs.setdefault(mode, set()).add(env)

	if first_meta is None or resolved_src_rank is None or resolved_dst_rank is None:
		print("gpu_plot: could not collect any data for the requested pair.", file=sys.stderr)
		return 1

	if not grouped_series:
		print("gpu_plot: no series to plot after filtering.", file=sys.stderr)
		return 1

	out_dir.mkdir(parents=True, exist_ok=True)
	src_label = pair_display_label(first_meta, resolved_src_rank)
	dst_label = pair_display_label(first_meta, resolved_dst_rank)
	metric_title = gh.METRIC_TITLE.get(metric, metric)
	metric_unit = gh.METRIC_CBAR_LABEL.get(metric, "мкс")
	written_paths: list[Path] = []
	cmap = mpl.colormaps["tab10"] if hasattr(mpl, "colormaps") else mpl.cm.get_cmap("tab10")

	for mode in sorted(grouped_series):
		series = grouped_series[mode]
		fig, ax = plt.subplots(figsize=(8.8, 5.6))
		csv_rows: list[dict[str, Any]] = []
		transitions_rows: list[dict[str, Any]] = []
		x_ticks: set[int] = set()

		for idx, series_label in enumerate(sorted(series)):
			points = series[series_label]
			xs = sorted(points)
			ys = [float(np.mean(points[x])) for x in xs]
			yerr = [
				float(np.std(points[x], ddof=1)) if len(points[x]) > 1 else 0.0
				for x in xs
			]
			x_ticks.update(xs)
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
			events = detect_transitions(xs, ys)
			for e in events:
				x_tr = int(e["to_bytes"])
				ax.axvline(
					x=x_tr,
					color=color,
					linestyle="--",
					linewidth=1.0,
					alpha=0.45,
				)
				transitions_rows.append(
					{
						"series": series_label,
						"mode": mode,
						"from_bytes": int(e["from_bytes"]),
						"to_bytes": x_tr,
						"from_bytes_human": format_bytes_human(int(e["from_bytes"])),
						"to_bytes_human": format_bytes_human(x_tr),
						"from_value": e["from_value"],
						"to_value": e["to_value"],
						"delta_value": e["delta_value"],
						"rel_change_pct": e["rel_change_pct"],
					}
				)

			for x, y, err in zip(xs, ys, yerr):
				csv_rows.append(
					{
						"series": series_label,
						"mode": mode,
						"bytes": x,
						"bytes_human": format_bytes_human(x),
						"value": y,
						"stddev": err,
						"samples": len(points[x]),
					}
				)

		env_text = ", ".join(sorted(group_envs.get(mode, set())))
		ax.set_title(
			"\n".join(
				[
					f"{metric_title} для пары {src_label} -> {dst_label}",
					f"Режим: {mode_title(mode)}; среда копирования: {env_text}",
				]
			),
			fontsize=11,
		)
		ax.set_xlabel("Размер сообщения (байты), шкала log₂", fontsize=10)
		ax.set_ylabel(f"{metric_title} ({metric_unit})", fontsize=10)
		ax.grid(True, which="major", color="#d4d4d8", linewidth=0.8, alpha=0.9)
		xticks_sorted = sorted(x_ticks)
		ax.set_xscale("log", base=2)
		ax.set_xticks(xticks_sorted)
		ax.set_xticklabels(
			[format_bytes_human(x) for x in xticks_sorted], rotation=35, ha="right"
		)
		if series:
			ax.legend(
				loc="upper left",
				bbox_to_anchor=(0.0, -0.18),
				fontsize=9,
				frameon=True,
			)
		fig.subplots_adjust(bottom=0.28)

		base_name = (
			f"pair_src{sanitize_token(src_label)}"
			f"_dst{sanitize_token(dst_label)}"
			f"_{sanitize_token(mode)}"
			f"_{gh.METRIC_FILE_STEM.get(metric, metric)}_vs_bytes"
		)
		png_path = out_dir / f"{base_name}.png"
		csv_path = out_dir / f"{base_name}.csv"
		transitions_csv_path = out_dir / f"{base_name}_transitions.csv"
		fig.savefig(png_path, dpi=150, bbox_inches="tight")
		plt.close(fig)

		with csv_path.open("w", encoding="utf-8", newline="") as f:
			writer = csv.DictWriter(
				f,
				fieldnames=[
					"series",
					"mode",
					"bytes",
					"bytes_human",
					"value",
					"stddev",
					"samples",
				],
			)
			writer.writeheader()
			writer.writerows(csv_rows)
		with transitions_csv_path.open("w", encoding="utf-8", newline="") as f:
			writer = csv.DictWriter(
				f,
				fieldnames=[
					"series",
					"mode",
					"from_bytes",
					"to_bytes",
					"from_bytes_human",
					"to_bytes_human",
					"from_value",
					"to_value",
					"delta_value",
					"rel_change_pct",
				],
			)
			writer.writeheader()
			writer.writerows(transitions_rows)

		written_paths.extend([png_path, csv_path, transitions_csv_path])

	for path in written_paths:
		print(path)
	return 0


def main() -> int:
	p = argparse.ArgumentParser(
		description="Line plots bytes -> latency from gpu text output (pair ... lines). X axis: log₂.",
		epilog=(
			"Arguments:\n"
			"  input            .txt file or directory with .txt files\n"
			"  -o, --out        output directory for PNG/CSV files\n"
			"  --src, --dst     axis label only, like 28.0 or cn28.0\n"
			"  --metric         avg_us | med_us | min_us | max_us | var_us | std_us\n"
			"                   default: med_us\n\n"
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
		help="Источник пары: только подпись оси, например 28.0",
	)
	p.add_argument(
		"--dst",
		required=True,
		help="Приёмник пары: только подпись оси, например 28.1",
	)
	p.add_argument(
		"--metric",
		choices=gh.METRIC_KEYS,
		default="med_us",
		help="Метрика графика (по умолчанию med_us)",
	)
	args = p.parse_args()

	if not AXIS_LABEL_RE.fullmatch(args.src.strip()):
		print(
			f"gpu_plot: --src must be an axis label like 28.0, got: {args.src}",
			file=sys.stderr,
		)
		return 1
	if not AXIS_LABEL_RE.fullmatch(args.dst.strip()):
		print(
			f"gpu_plot: --dst must be an axis label like 28.1, got: {args.dst}",
			file=sys.stderr,
		)
		return 1

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
