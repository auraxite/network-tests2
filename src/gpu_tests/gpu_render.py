#!/usr/bin/env python3
"""
Строит heatmap-матрицы по текстовому выводу gpu_one_to_one (stdout или --out).

В логе можно задать hostname строкой «Hostname: cn2» (или --hostname cn2).
PNG: cn2_avg.png, cn2_median.png, … (если hostname неизвестен: node<N>_...).
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Colormap, LinearSegmentedColormap

LATENCY_GR = LinearSegmentedColormap.from_list(
	"latency_gr",
	["#045a2d", "#16a34a", "#f97316", "#dc2626", "#7f1d1d"],
	N=256,
)


def resolve_colormap(name: str) -> Colormap:
	if name == "latency_gr":
		base: Colormap = LATENCY_GR
	else:
		base = mpl.colormaps[name] if hasattr(mpl, "colormaps") else mpl.cm.get_cmap(name)
	cmap = base.copy()
	cmap.set_bad(color="white", alpha=1.0)
	return cmap

# G… — глобальный id; r… — MPI rank (индекс в матрице).
PAIR_LINE_RE = re.compile(
	r"^pair (?:g|G|GPU)(\d+) -> (?:g|G|GPU)(\d+) \(r(\d+):0 -> r(\d+):0\) "
	r"avg_us=([0-9.eE+-]+) median_us=([0-9.eE+-]+) min_us=([0-9.eE+-]+) "
	r"max_us=([0-9.eE+-]+) var_us=([0-9.eE+-]+)\s*$"
)

RANKS_LINE_RE = re.compile(r"^Ranks:\s*(\d+)")
NODE_LINE_RE = re.compile(r"^Node:\s*(\d+)\s*$", re.I)
HOSTNAME_LINE_RE = re.compile(r"^Hostname:\s*(\S+)\s*$", re.I)
BYTES_LINE_RE = re.compile(r"^Bytes:\s*(\d+)\s*$")
_DECIMAL_MB = 1_000_000
WARMUP_LINE_RE = re.compile(r"^Warmup:\s*(\d+)\s*$")
ITERS_LINE_RE = re.compile(r"^Iters:\s*(\d+)\s*$")


METRIC_KEYS = ("avg_us", "median_us", "min_us", "max_us", "var_us")

# Суффикс файла: cn2_avg.png или node0_avg.png.
METRIC_FILE_STEM = {
	"avg_us": "avg",
	"median_us": "median",
	"min_us": "min",
	"max_us": "max",
	"var_us": "var",
}


METRIC_CBAR_LABEL = {
	"avg_us": "мкс",
	"median_us": "мкс",
	"min_us": "мкс",
	"max_us": "мкс",
	"var_us": "мкс²",
}
METRIC_TITLE = {
	"avg_us": "Среднее арифметическое задержек",
	"median_us": "Медиана задержек",
	"min_us": "Минимальное задержек",
	"max_us": "Максимальное задержек",
	"var_us": "Выборочная дисперсия задержек",
}


def parse_gpu_one_to_one_text(
	text: str,
) -> tuple[dict[str, Any], list[tuple[int, int, list[float]]]]:
	meta: dict[str, Any] = {}
	pairs: list[tuple[int, int, list[float]]] = []

	for raw in text.splitlines():
		line = raw.strip()
		if line.startswith("CUDA-aware MPI:"):
			meta["cuda_aware"] = line.split(":", 1)[1].strip()
		elif line.startswith("Mode:"):
			meta["mode"] = line.split(":", 1)[1].strip()
		elif line.startswith("Ranks:"):
			m = RANKS_LINE_RE.match(line)
			if m:
				meta["ranks"] = int(m.group(1))
		elif (m := NODE_LINE_RE.match(line)):
			meta["node"] = int(m.group(1))
		elif (m := HOSTNAME_LINE_RE.match(line)):
			meta["hostname"] = m.group(1)
		elif (m := BYTES_LINE_RE.match(line)):
			meta["bytes"] = int(m.group(1))
		elif (m := WARMUP_LINE_RE.match(line)):
			meta["warmup"] = int(m.group(1))
		elif (m := ITERS_LINE_RE.match(line)):
			meta["iters"] = int(m.group(1))
		else:
			m = PAIR_LINE_RE.match(line)
			if m:
				src_g = int(m.group(1))
				dst_g = int(m.group(2))
				src_r = int(m.group(3))
				dst_r = int(m.group(4))
				vals = [float(m.group(i)) for i in range(5, 10)]
				gmap: dict[int, int] = meta.setdefault("global_by_rank", {})
				gmap[src_r] = src_g
				gmap[dst_r] = dst_g
				pairs.append((src_r, dst_r, vals))

	return meta, pairs


def infer_n(meta: dict[str, Any], pairs: list[tuple[int, int, list[float]]]) -> int:
	if "ranks" in meta:
		return int(meta["ranks"])
	if not pairs:
		return 0
	return max(max(s, d) for s, d, _ in pairs) + 1


def fill_matrices(
	n: int, pairs: list[tuple[int, int, list[float]]]
) -> dict[str, np.ndarray]:
	mats: dict[str, np.ndarray] = {k: np.full((n, n), np.nan, dtype=float) for k in METRIC_KEYS}
	for src, dst, vals in pairs:
		if src == dst:
			continue
		if 0 <= src < n and 0 <= dst < n:
			for k, v in zip(METRIC_KEYS, vals):
				mats[k][src, dst] = v
	return mats


def _title_block(
	meta: dict[str, Any], metric: str, *, node_id: int | None, hostname: str | None
) -> str:
	mode = str(meta.get("mode", "unknown"))
	title_line = METRIC_TITLE.get(metric, metric)
	parts: list[str] = []
	if hostname:
		parts.append(f"Узел {hostname}")
	elif node_id is not None:
		parts.append(f"Узел node{node_id}")
	parts.extend(
		[
			title_line,
			"Схема обмена: GPU one-to-one",
			f"Режим копирования: {mode}",
		]
	)
	return "\n".join(parts)


def _format_size_mb_decimal(b: int) -> str:
	mb = b / _DECIMAL_MB
	text = f"{mb:.12f}".rstrip("0").rstrip(".")
	return f"size: {text} MB"


def _param_block(meta: dict[str, Any]) -> str:
	lines: list[str] = []
	if "bytes" in meta:
		lines.append(_format_size_mb_decimal(int(meta["bytes"])))
	if "warmup" in meta:
		lines.append(f"warmup: {int(meta['warmup'])}")
	if "iters" in meta:
		lines.append(f"iters: {int(meta['iters'])}")
	return "\n".join(lines)


def draw_heatmap(
	matrix: np.ndarray,
	metric: str,
	out_path: Path,
	*,
	title_block: str,
	param_block: str,
	cmap: Colormap,
	dpi: int,
	tick_labels: list[str],
) -> None:
	labels = tick_labels
	fig_w = max(5.0, 0.45 * matrix.shape[1] + 2.5)
	fig_h = max(4.5, 0.45 * matrix.shape[0] + 2.4)
	fig, ax = plt.subplots(figsize=(fig_w, fig_h))

	masked = np.ma.masked_invalid(matrix)
	im = ax.imshow(masked, cmap=cmap, aspect="equal", interpolation="nearest")

	ax.set_xticks(range(len(labels)))
	ax.set_yticks(range(len(labels)))
	ax.set_xticklabels(labels, rotation=0, fontsize=9)
	ax.set_yticklabels(labels, fontsize=9)

	if param_block:
		ax.set_xlabel(
			f"dst (глоб. GPU)\n\n{param_block}",
			fontsize=8,
			family="monospace",
			labelpad=4,
		)
	else:
		ax.set_xlabel("dst (глоб. GPU)")
	ax.set_ylabel("src (глоб. GPU)")

	ax.set_title(title_block, fontsize=9)

	cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
	cbar.set_label(METRIC_CBAR_LABEL.get(metric, "мкс"), rotation=0, labelpad=12)

	fig.tight_layout()

	fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
	plt.close(fig)


def main() -> int:
	p = argparse.ArgumentParser(
		description="Heatmaps from gpu_one_to_one text output (pair ... lines)."
	)
	p.add_argument(
		"input",
		nargs="?",
		default="-",
		help="Path to captured output, or '-' for stdin (default: -)",
	)
	p.add_argument(
		"-o",
		"--out-dir",
		type=Path,
		default=Path("."),
		help="Directory for PNG files (default: current directory)",
	)
	p.add_argument(
		"--node-id",
		type=int,
		default=None,
		help="(legacy) Номер узла для fallback-имени node<N>_....png, если hostname не задан",
	)
	p.add_argument(
		"--hostname",
		default=None,
		help="Hostname для заголовка и имени <hostname>_....png; иначе из строки Hostname: в логе",
	)
	p.add_argument(
		"--metrics",
		default=",".join(METRIC_KEYS),
		help=f"Подмножество через запятую: {','.join(METRIC_KEYS)} (default: all)",
	)
	p.add_argument(
		"--cmap",
		default="latency_gr",
		help="Colormap: latency_gr или имя matplotlib",
	)
	p.add_argument("--dpi", type=int, default=150, help="Разрешение PNG (default: 150)")

	args = p.parse_args()

	in_path = args.input
	if in_path == "-":
		text = sys.stdin.read()
	else:
		path = Path(in_path)
		text = path.read_text(encoding="utf-8", errors="replace")

	meta, pairs = parse_gpu_one_to_one_text(text)
	if not pairs:
		print("gpu_render: no matching 'pair GPU...' lines found.", file=sys.stderr)
		return 1

	n = infer_n(meta, pairs)
	if n <= 0:
		print("gpu_render: could not infer matrix size.", file=sys.stderr)
		return 1

	mats = fill_matrices(n, pairs)
	gmap = meta.get("global_by_rank")
	if isinstance(gmap, dict) and len(gmap) > 0:
		tick_labels = [f"gpu{int(gmap.get(i, i))}" for i in range(n)]
	else:
		tick_labels = [f"gpu{i}" for i in range(n)]

	want = {m.strip() for m in args.metrics.split(",") if m.strip()}
	unknown = want - set(METRIC_KEYS)
	if unknown:
		print(f"gpu_render: unknown metrics: {unknown}", file=sys.stderr)
		return 1

	node_id = args.node_id
	if node_id is None and "node" in meta:
		node_id = int(meta["node"])
	if node_id is None:
		node_id = 0
	hostname = args.hostname
	if hostname is None and "hostname" in meta:
		hostname = str(meta["hostname"])
	if hostname:
		# Для имени файла оставляем только безопасные символы.
		hostname = re.sub(r"[^0-9A-Za-z_.-]", "_", hostname)

	args.out_dir.mkdir(parents=True, exist_ok=True)
	params = _param_block(meta)
	cmap_resolved = resolve_colormap(args.cmap)

	for key in METRIC_KEYS:
		if key not in want:
			continue
		stem = METRIC_FILE_STEM.get(key, key)
		prefix = hostname if hostname else f"node{node_id}"
		out_file = args.out_dir / f"{prefix}_{stem}.png"
		draw_heatmap(
			mats[key],
			key,
			out_file,
			title_block=_title_block(meta, key, node_id=node_id, hostname=hostname),
			param_block=params,
			cmap=cmap_resolved,
			dpi=args.dpi,
			tick_labels=tick_labels,
		)
		print(out_file)

	return 0


if __name__ == "__main__":
	raise SystemExit(main())
