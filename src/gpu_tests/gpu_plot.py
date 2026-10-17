#!/usr/bin/env python3

from __future__ import annotations

"""Графики «размер сообщения -> задержка» для одной пары по результатам gpu_tests.

Ось X логарифмическая по основанию 2. На каждый режим и среду — свой PNG и CSV.

Аргументы:
  input         .txt или каталог с .txt
  -o, --out     каталог для PNG/CSV
  --src, --dst  метки осей, например 28.0
  --metric      avg_us | med_us | min_us | max_us | var_us | std_us (по умолчанию med_us)
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

import gpu_io as io

AXIS_LABEL_RE = re.compile(r"^[0-9A-Za-z_.-]+\.[0-9]+$")
CSV_FIELDS = ["series", "mode", "bytes", "bytes_human", "value", "stddev", "samples"]


def format_bytes_human(nbytes: int) -> str:
	for unit, size in (("GiB", 1 << 30), ("MiB", 1 << 20), ("KiB", 1 << 10)):
		if nbytes >= size and nbytes % size == 0:
			return f"{nbytes // size} {unit}"
	return f"{nbytes} B"


def pair_display_label(meta: dict[str, Any], rank: int) -> str:
	n = meta.get("ranks")
	rank_hosts = io.rank_hosts_from_meta(meta, n) if isinstance(n, int) else None
	if rank_hosts and 0 <= rank < len(rank_hosts):
		return io.axis_labels_by_node(rank_hosts)[rank]
	return str(rank)


def resolve_axis_pair_label(label: str, meta: dict[str, Any]) -> int | None:
	"""Только метки осей вида 28.0, номер ранга сам по себе не принимается."""
	token = label.strip()
	return io.pair_label_to_rank(token, meta) if AXIS_LABEL_RE.fullmatch(token) else None


def pair_metric_value(meta: dict[str, Any], pairs: list[tuple[int, int, list[float]]],
                      src_spec: str, dst_spec: str, metric: str) -> tuple[int, int, float | None]:
	"""Ранги пары и значение величины (последнее непустое, если строк несколько)."""
	src_rank = resolve_axis_pair_label(src_spec, meta)
	dst_rank = resolve_axis_pair_label(dst_spec, meta)
	if src_rank is None or dst_rank is None:
		return -1, -1, None
	idx = io.METRIC_KEYS.index(metric)
	value: float | None = None
	for src, dst, vals in pairs:
		if src == src_rank and dst == dst_rank and not math.isnan(vals[idx]):
			value = float(vals[idx])
	return src_rank, dst_rank, value


def plot_series_label(meta: dict[str, Any]) -> str:
	"""Подпись среды без таймера по умолчанию (mpi)."""
	parts = [str(meta.get("env", "unknown"))]
	transport = str(meta.get("transport", "unknown"))
	timer = str(meta.get("timer", "unknown"))
	if transport not in ("", "unknown"):
		parts.append(transport)
	if timer not in ("", "mpi", "unknown"):
		parts.append(timer)
	return " / ".join(parts)


def sanitize_token(text: str) -> str:
	return re.sub(r"[^0-9A-Za-z_.-]+", "_", text.strip()) or "na"


def file_nbytes(meta: dict[str, Any], txt_path: Path) -> int | None:
	"""Размер из шапки, при её отсутствии — из тега _bNNN_ в имени файла."""
	nbytes = meta.get("bytes")
	if isinstance(nbytes, int) and nbytes > 0:
		return nbytes
	m = io.BYTES_TAG_RE.search(txt_path.stem)
	return int(m.group(1)) if m and int(m.group(1)) > 0 else None


def render_pair_plot_from_paths(txt_paths: list[Path], out_dir: Path, *,
                                src_spec: str, dst_spec: str, metric: str) -> int:
	# series[mode][подпись среды][размер] -> значения из разных файлов
	series: dict[str, dict[str, dict[int, list[float]]]] = {}
	first: tuple[dict[str, Any], int, int] | None = None

	for txt in txt_paths:
		meta, pairs = io.parse_gpu_one_to_one_text(txt.read_text(encoding="utf-8", errors="replace"))
		nbytes = file_nbytes(meta, txt)
		if nbytes is None:
			print(f"gpu_plot: [{txt}] missing valid Bytes: header, skip.", file=sys.stderr)
			continue
		src_rank, dst_rank, value = pair_metric_value(meta, pairs, src_spec, dst_spec, metric)
		if value is None:
			print(f"gpu_plot: [{txt}] no value for pair {src_spec}->{dst_spec} metric={metric}, skip.",
			      file=sys.stderr)
			continue
		if first is None:
			first = (meta, src_rank, dst_rank)
		elif (first[1], first[2]) != (src_rank, dst_rank):
			print(f"gpu_plot: [{txt}] pair spec resolves to another rank pair, skip.", file=sys.stderr)
			continue
		mode = str(meta.get("mode", "unknown"))
		series.setdefault(mode, {}).setdefault(plot_series_label(meta), {}) \
		      .setdefault(nbytes, []).append(value)

	if first is None:
		print("gpu_plot: could not collect any data for the requested pair.", file=sys.stderr)
		return 1

	out_dir.mkdir(parents=True, exist_ok=True)
	src_label = pair_display_label(first[0], first[1])
	dst_label = pair_display_label(first[0], first[2])
	metric_title = io.METRIC_TITLE.get(metric, metric)
	metric_unit = io.METRIC_CBAR_LABEL.get(metric, "мкс")
	cmap = mpl.colormaps["tab10"] if hasattr(mpl, "colormaps") else mpl.cm.get_cmap("tab10")
	written: list[Path] = []

	for mode in sorted(series):
		for idx, label in enumerate(sorted(series[mode])):
			points = series[mode][label]
			xs = sorted(points)
			ys = [float(np.mean(points[x])) for x in xs]
			yerr = [float(np.std(points[x], ddof=1)) if len(points[x]) > 1 else 0.0 for x in xs]
			color = cmap(idx % max(1, getattr(cmap, "N", 10)))

			fig, ax = plt.subplots(figsize=(8.8, 5.6))
			style = dict(marker="o", markersize=5, linewidth=1.8, color=color)
			if any(e > 0.0 for e in yerr):
				ax.errorbar(xs, ys, yerr=yerr, capsize=3, **style)
			else:
				ax.plot(xs, ys, **style)
			ax.set_title("\n".join([
				f"{metric_title} для пары {src_label} -> {dst_label}",
				f"Режим передачи: {mode.replace('-', '_')}",
				f"Среда передачи: {label}",
			]), fontsize=11)
			ax.set_xlabel("Размер сообщения (байты), шкала log₂", fontsize=10)
			ax.set_ylabel(f"{metric_title} ({metric_unit})", fontsize=10)
			ax.grid(True, which="major", color="#d4d4d8", linewidth=0.8, alpha=0.9)
			ax.set_xscale("log", base=2)
			ax.set_xticks(xs)
			ax.set_xticklabels([str(x) for x in xs], rotation=35, ha="right")
			fig.subplots_adjust(bottom=0.28)
			fig.text(0.5, 0.082, f"сгенерировано: {io.generation_timestamp()}", ha="center",
			         va="bottom", fontsize=9, color="black", transform=fig.transFigure)

			base_name = (f"src{sanitize_token(src_label)}_dst{sanitize_token(dst_label)}"
			             f"_{sanitize_token(mode)}_{sanitize_token(label)}"
			             f"_{io.METRIC_FILE_STEM.get(metric, metric)}")
			png_path = out_dir / f"{base_name}.png"
			csv_path = out_dir / f"{base_name}.csv"
			fig.savefig(png_path, dpi=150, bbox_inches="tight")
			plt.close(fig)

			with csv_path.open("w", encoding="utf-8", newline="") as f:
				writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
				writer.writeheader()
				writer.writerows({"series": label, "mode": mode, "bytes": x,
				                  "bytes_human": format_bytes_human(x), "value": y, "stddev": e,
				                  "samples": len(points[x])} for x, y, e in zip(xs, ys, yerr))
			written += [png_path, csv_path]

	for path in written:
		print(path)
	return 0


def main() -> int:
	p = argparse.ArgumentParser(
		description="Line plots bytes -> latency from gpu text output (pair ... lines). X axis: log₂.")
	p.add_argument("input", help="Файл .txt или каталог с .txt файлами")
	p.add_argument("-o", "--out", dest="out_dir", type=Path, default=Path("."), help="Каталог для PNG/CSV")
	p.add_argument("--src", required=True, help="Источник пары: подпись оси, например 28.0")
	p.add_argument("--dst", required=True, help="Приёмник пары: подпись оси, например 28.1")
	p.add_argument("--metric", choices=io.METRIC_KEYS, default="med_us",
	               help="Величина (по умолчанию med_us)")
	args = p.parse_args()

	for name, spec in (("--src", args.src), ("--dst", args.dst)):
		if not AXIS_LABEL_RE.fullmatch(spec.strip()):
			print(f"gpu_plot: {name} must be an axis label like 28.0, got: {spec}", file=sys.stderr)
			return 1

	path = Path(args.input)
	txts = sorted(path.glob("*.txt")) if path.is_dir() else [path] if path.is_file() else []
	if not txts:
		print(f"gpu_plot: no .txt input at {path.resolve()}", file=sys.stderr)
		return 1
	return render_pair_plot_from_paths(txts, args.out_dir, src_spec=args.src, dst_spec=args.dst,
	                                   metric=args.metric)


if __name__ == "__main__":
	raise SystemExit(main())
