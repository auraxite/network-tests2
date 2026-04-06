#!/usr/bin/env python3
"""
Строит heatmap-матрицы по текстовому выводу gpu_one_to_one (stdout или --out).

Если первый аргумент — каталог, обрабатываются все *.txt; PNG кладутся в
--out-dir/<имя_файла_без_txt>/ для каждого лога.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import math
import re
import sys
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Colormap, LinearSegmentedColormap


_PAIR_HEAD = (
	r"^pair (?:(?:g|G|GPU))?(\d+) -> (?:(?:g|G|GPU))?(\d+) "
	r"\(r(\d+)(?::0)? -> r(\d+)(?::0)?\) "
)

# Полная строка (как по умолчанию в gpu_one_to_one --stat all).
PAIR_LINE_RE = re.compile(
	_PAIR_HEAD
	+ r"avg_us=([0-9.eE+-]+)\s+"
	+ r"median_us=([0-9.eE+-]+)\s+"
	+ r"min_us=([0-9.eE+-]+)\s+"
	+ r"max_us=([0-9.eE+-]+)\s+"
	+ r"var_us=([0-9.eE+-]+)"
	+ r"(?:\s+std_us=([0-9.eE+-]+))?\s*$"
)

# Одна метрика (--stat avg|median|min|max|var|std).
PAIR_AVG_ONLY = re.compile(_PAIR_HEAD + r"avg_us=([0-9.eE+-]+)\s*$")
PAIR_MEDIAN_ONLY = re.compile(_PAIR_HEAD + r"median_us=([0-9.eE+-]+)\s*$")
PAIR_MIN_ONLY = re.compile(_PAIR_HEAD + r"min_us=([0-9.eE+-]+)\s*$")
PAIR_MAX_ONLY = re.compile(_PAIR_HEAD + r"max_us=([0-9.eE+-]+)\s*$")
PAIR_VAR_ONLY = re.compile(_PAIR_HEAD + r"var_us=([0-9.eE+-]+)\s*$")
PAIR_STD_ONLY = re.compile(_PAIR_HEAD + r"std_us=([0-9.eE+-]+)\s*$")
METRIC_KEYS = ("avg_us", "median_us", "min_us", "max_us", "var_us", "std_us")
METRIC_FILE_STEM = {
	"avg_us": "avg",
	"median_us": "median",
	"min_us": "min",
	"max_us": "max",
	"var_us": "var",
	"std_us": "std",
}
METRIC_CBAR_LABEL = {
	"avg_us": "мкс",
	"median_us": "мкс",
	"min_us": "мкс",
	"max_us": "мкс",
	"var_us": "мкс²",
	"std_us": "мкс",
}
METRIC_TITLE = {
	"avg_us": "Среднее арифметическое задержек",
	"median_us": "Медиана задержек",
	"min_us": "Минимальное задержек",
	"max_us": "Максимальное задержек",
	"var_us": "Выборочная дисперсия задержек",
	"std_us": "Стандартное отклонение задержек",
}


RANKS_LINE_RE = re.compile(r"^Ranks:\s*(\d+)")
HOSTNAME_LINE_RE = re.compile(r"^Hostname:\s*(\S+)\s*$", re.I)
BYTES_LINE_RE = re.compile(r"^Bytes:\s*(\d+)\s*$")
DECIMAL_MB = 1_000_000
WARMUP_LINE_RE = re.compile(r"^Warmup:\s*(\d+)\s*$")
ITERS_LINE_RE = re.compile(r"^Iters:\s*(\d+)\s*$")
TOTAL_TIME_LINE_RE = re.compile(r"^TotalTimeSec:\s*([0-9.eE+-]+)\s*$")
TOTAL_ELAPSED_LINE_RE = re.compile(r"^TotalElapsedSec:\s*([0-9.eE+-]+)\s*$")
REP_TAG_RE = re.compile(r"(?:^|_)rep(\d+)(?:_|$)", re.I)
MODE_TAG_RE = re.compile(r"(?:^|_)m([A-Za-z0-9]+)(?:_|$)", re.I)
BYTES_TAG_RE = re.compile(r"(?:^|_)b(\d+)(?:_|$)", re.I)
WARMUP_TAG_RE = re.compile(r"(?:^|_)w(\d+)(?:_|$)", re.I)
ITERS_TAG_RE = re.compile(r"(?:^|_)i(\d+)(?:_|$)", re.I)


LATENCY_GR = LinearSegmentedColormap.from_list(
	"latency_gr",
	["#045a2d", "#16a34a", "#f97316", "#dc2626", "#7f1d1d"],
	N=256,
)


"""Выбор colormap и цвет для NaN."""
def resolve_colormap(name: str) -> Colormap:
	if name == "latency_gr":
		base: Colormap = LATENCY_GR
	else:
		base = mpl.colormaps[name] if hasattr(mpl, "colormaps") else mpl.cm.get_cmap(name)
	cmap = base.copy()
	cmap.set_bad(color="white", alpha=1.0)
	return cmap

"""Разбор текста лога в метаданные и список пар rank->rank."""
def parse_gpu_one_to_one_text(
	text: str,
) -> tuple[dict[str, Any], list[tuple[int, int, list[float]]]]:
	meta: dict[str, Any] = {}
	pairs: list[tuple[int, int, list[float]]] = []

	for raw in text.splitlines():
		line = raw.strip()
		if line.startswith("Mode:"):
			meta["mode"] = line.split(":", 1)[1].strip()
		elif line.startswith("Ranks:"):
			m = RANKS_LINE_RE.match(line)
			if m:
				meta["ranks"] = int(m.group(1))
		elif (m := HOSTNAME_LINE_RE.match(line)):
			meta["hostname"] = m.group(1)
		elif (m := BYTES_LINE_RE.match(line)):
			meta["bytes"] = int(m.group(1))
		elif (m := WARMUP_LINE_RE.match(line)):
			meta["warmup"] = int(m.group(1))
		elif (m := ITERS_LINE_RE.match(line)):
			meta["iters"] = int(m.group(1))
		elif (m := TOTAL_TIME_LINE_RE.match(line)):
			meta["total_elapsed_s"] = float(m.group(1))
		elif (m := TOTAL_ELAPSED_LINE_RE.match(line)):
			meta["total_elapsed_s"] = float(m.group(1))
		else:
			m = PAIR_LINE_RE.match(line)
			if m:
				src_r = int(m.group(3))
				dst_r = int(m.group(4))
				std_v = float(m.group(10)) if m.group(10) is not None else math.nan
				vals = [float(m.group(i)) for i in range(5, 10)] + [std_v]
				pairs.append((src_r, dst_r, vals))
				continue
			nan6 = [math.nan, math.nan, math.nan, math.nan, math.nan, math.nan]
			if (m := PAIR_AVG_ONLY.match(line)):
				src_r = int(m.group(3))
				dst_r = int(m.group(4))
				v = float(m.group(5))
				vals = [v, nan6[1], nan6[2], nan6[3], nan6[4], nan6[5]]
				pairs.append((src_r, dst_r, vals))
				continue
			if (m := PAIR_MEDIAN_ONLY.match(line)):
				src_r = int(m.group(3))
				dst_r = int(m.group(4))
				v = float(m.group(5))
				vals = [nan6[0], v, nan6[2], nan6[3], nan6[4], nan6[5]]
				pairs.append((src_r, dst_r, vals))
				continue
			if (m := PAIR_MIN_ONLY.match(line)):
				src_r = int(m.group(3))
				dst_r = int(m.group(4))
				v = float(m.group(5))
				vals = [nan6[0], nan6[1], v, nan6[3], nan6[4], nan6[5]]
				pairs.append((src_r, dst_r, vals))
				continue
			if (m := PAIR_MAX_ONLY.match(line)):
				src_r = int(m.group(3))
				dst_r = int(m.group(4))
				v = float(m.group(5))
				vals = [nan6[0], nan6[1], nan6[2], v, nan6[4], nan6[5]]
				pairs.append((src_r, dst_r, vals))
				continue
			if (m := PAIR_VAR_ONLY.match(line)):
				src_r = int(m.group(3))
				dst_r = int(m.group(4))
				v = float(m.group(5))
				vals = [nan6[0], nan6[1], nan6[2], nan6[3], v, nan6[5]]
				pairs.append((src_r, dst_r, vals))
				continue
			if (m := PAIR_STD_ONLY.match(line)):
				src_r = int(m.group(3))
				dst_r = int(m.group(4))
				v = float(m.group(5))
				vals = [nan6[0], nan6[1], nan6[2], nan6[3], nan6[4], v]
				pairs.append((src_r, dst_r, vals))

	return meta, pairs


"""Определяет размер матрицы n x n."""
def matrix_size(meta: dict[str, Any], pairs: list[tuple[int, int, list[float]]]) -> int:
	if "ranks" in meta:
		return int(meta["ranks"])
	if not pairs:
		return 0
	return max(max(s, d) for s, d, _ in pairs) + 1


"""Заполняет матрицы метрик по парам src/dst rank."""
def fill_matrices(
	n: int, pairs: list[tuple[int, int, list[float]]]
) -> dict[str, np.ndarray]:
	mats: dict[str, np.ndarray] = {k: np.full((n, n), np.nan, dtype=float) for k in METRIC_KEYS}
	for src, dst, vals in pairs:
		if src == dst:
			continue
		if 0 <= src < n and 0 <= dst < n:
			for k, v in zip(METRIC_KEYS, vals):
				if isinstance(v, float) and math.isnan(v):
					continue
				mats[k][src, dst] = v
	return mats


"""Собирает многострочный заголовок графика."""
def title_block(meta: dict[str, Any], metric: str, *, hostname: str | None) -> str:
	mode = str(meta.get("mode", "unknown"))
	title_line = METRIC_TITLE.get(metric, metric)
	parts: list[str] = []
	if hostname:
		parts.append(f"Узел {hostname}")
	parts.extend(
		[
			title_line,
			"Схема обмена: one-to-one",
			f"Режим копирования: {mode}",
		]
	)
	return "\n".join(parts)


"""Форматирует размер в десятичных MB."""
def format_size_mb_decimal(b: int) -> str:
	mb = b / DECIMAL_MB
	text = f"{mb:.12f}".rstrip("0").rstrip(".")
	return f"size: {text} MB"


def run_tags(source_path: Path | None, meta: dict[str, Any]) -> dict[str, str]:
	stem = source_path.stem if source_path is not None else ""
	mode_meta = str(meta.get("mode", "unknown")).lower()
	mode_default = "host" if mode_meta == "host" else ("auto" if mode_meta == "gpudirect" else mode_meta)

	def pick(pattern: re.Pattern[str], fallback: str) -> str:
		m = pattern.search(stem)
		return m.group(1) if m else fallback

	return {
		"rep": pick(REP_TAG_RE, "na"),
		"mode": pick(MODE_TAG_RE, mode_default),
		"b": pick(BYTES_TAG_RE, str(meta.get("bytes", "na"))),
		"w": pick(WARMUP_TAG_RE, str(meta.get("warmup", "na"))),
		"i": pick(ITERS_TAG_RE, str(meta.get("iters", "na"))),
	}


def total_time_line(meta: dict[str, Any]) -> str:
	v = meta.get("total_elapsed_s")
	if isinstance(v, (int, float)):
		return f"затраченное время: {float(v):.6f}s"
	return "затраченное время: n/a"


"""Собирает блок параметров под графиком."""
def param_block(
	meta: dict[str, Any], tags: dict[str, str], creation_time: str, total_time: str
) -> str:
	lines: list[str] = []
	lines.append(f"w: {tags['w']}  i: {tags['i']}")
	lines.append(f"время создания: {creation_time}")
	lines.append(total_time)
	return "\n".join(lines)


def format_cell_value(metric: str, v: float) -> str:
	av = abs(v)
	if metric == "var_us":
		return f"{v:.2g}" if av >= 1000 else f"{v:.1f}"
	if av >= 1000:
		return f"{v:.0f}"
	if av >= 100:
		return f"{v:.1f}"
	if av >= 10:
		return f"{v:.2f}"
	return f"{v:.3f}"


def annotate_cells(ax: Any, im: Any, matrix: np.ndarray, metric: str) -> None:
	n = matrix.shape[0]
	fontsize = max(5, min(9, int(90 / max(1, n))))
	for i in range(matrix.shape[0]):
		for j in range(matrix.shape[1]):
			v = matrix[i, j]
			if np.isnan(v):
				continue
			ax.text(
				j,
				i,
				format_cell_value(metric, float(v)),
				ha="center",
				va="center",
				fontsize=fontsize,
				color="white",
			)


"""Рисует и сохраняет одну heatmap-картинку."""
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
	axis_fs = 9
	fig_w = max(5.0, 0.45 * matrix.shape[1] + 2.5)
	fig_h = max(4.5, 0.45 * matrix.shape[0] + 2.4)
	fig, ax = plt.subplots(figsize=(fig_w, fig_h))

	masked = np.ma.masked_invalid(matrix)
	im = ax.imshow(masked, cmap=cmap, aspect="equal", interpolation="nearest")

	ax.set_xticks(range(len(labels)))
	ax.set_yticks(range(len(labels)))
	ax.set_xticklabels(labels, rotation=0, fontsize=axis_fs)
	ax.set_yticklabels(labels, fontsize=axis_fs)

	ax.set_xlabel(
		f"dst GPU\n\n{param_block}",
		fontsize=axis_fs,
		labelpad=4,
	)
	ax.set_ylabel("src GPU", fontsize=axis_fs, labelpad=4)

	ax.set_title(title_block, fontsize=9)

	annotate_cells(ax, im, matrix, metric)

	cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
	cbar.set_label(METRIC_CBAR_LABEL.get(metric, "мкс"), rotation=0, labelpad=12)

	fig.tight_layout()

	fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
	plt.close(fig)


def render_one_text(
	text: str, out_dir: Path, *, label: str, source_path: Path | None = None
) -> int:
	meta, pairs = parse_gpu_one_to_one_text(text)
	if not pairs:
		print(f"gpu_render: [{label}] no matching 'pair ...' lines found.", file=sys.stderr)
		return 1

	n = matrix_size(meta, pairs)
	if n <= 0:
		print(f"gpu_render: [{label}] could not infer matrix size.", file=sys.stderr)
		return 1

	mats = fill_matrices(n, pairs)
	tick_labels = [str(i) for i in range(n)]

	want = set(METRIC_KEYS)

	hostname = str(meta["hostname"]) if "hostname" in meta else None
	if hostname:
		hostname = re.sub(r"[^0-9A-Za-z_.-]", "_", hostname)
	else:
		hostname = "host-unknown"

	out_dir.mkdir(parents=True, exist_ok=True)
	tags = run_tags(source_path, meta)
	creation_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
	total_time = total_time_line(meta)
	params = param_block(meta, tags, creation_time, total_time)
	cmap_resolved = resolve_colormap("latency_gr")

	written = 0
	for key in METRIC_KEYS:
		if key not in want:
			continue
		if bool(np.all(np.isnan(mats[key]))):
			continue
		stem = METRIC_FILE_STEM.get(key, key)
		out_file = out_dir / (
			f"{hostname}_rep{tags['rep']}_m{tags['mode']}"
			f"_b{tags['b']}_w{tags['w']}_i{tags['i']}_{stem}.png"
		)
		draw_heatmap(
			mats[key],
			key,
			out_file,
			title_block=title_block(meta, key, hostname=hostname),
			param_block=params,
			cmap=cmap_resolved,
			dpi=150,
			tick_labels=tick_labels,
		)
		print(out_file)
		written += 1

	if written == 0:
		print(f"gpu_render: [{label}] no non-empty metric matrices to plot.", file=sys.stderr)
		return 1

	return 0


def main() -> int:
	p = argparse.ArgumentParser(
		description="Heatmaps from gpu_one_to_one text output (pair ... lines)."
	)
	p.add_argument(
		"input",
		nargs="?",
		default="-",
		help="Файл .txt, каталог с .txt, или '-' для stdin",
	)
	p.add_argument(
		"-o",
		"--out-dir",
		type=Path,
		default=Path("."),
		help="Каталог для PNG; для каждого входного .txt создаётся подкаталог по имени файла",
	)
	args = p.parse_args()

	in_path = args.input
	if in_path == "-":
		text = sys.stdin.read()
		return render_one_text(text, args.out_dir, label="stdin", source_path=None)

	path = Path(in_path)
	if not path.exists():
		print(
			f"gpu_render: path not found: {path.resolve()}",
			file=sys.stderr,
		)
		return 1
	if path.is_dir():
		txts = sorted(path.glob("*.txt"))
		if not txts:
			print(f"gpu_render: no *.txt in {path}", file=sys.stderr)
			return 1
		code = 0
		for txt in txts:
			text = txt.read_text(encoding="utf-8", errors="replace")
			sub = args.out_dir / txt.stem
			r = render_one_text(text, sub, label=str(txt), source_path=txt)
			if r != 0:
				code = 1
		return code

	if path.is_file():
		text = path.read_text(encoding="utf-8", errors="replace")
		return render_one_text(text, args.out_dir, label=str(path), source_path=path)

	print(f"gpu_render: not a file or directory: {path}", file=sys.stderr)
	return 1


if __name__ == "__main__":
	raise SystemExit(main())
