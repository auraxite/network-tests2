#!/usr/bin/env python3

from __future__ import annotations

"""Render heatmaps from gpu (MPI GPU tests) text output.

Arguments:
  input            .txt file, directory with .txt files, or '-' (stdin)
  -o, --out-dir    output directory for PNG files
  -t, --timer      choose timer source: mpi | cpu | cuda (default: cuda)
  --sort, --sorted enable raw sorting pipeline (default: none)
"""

import argparse
from datetime import datetime
import math
import re
import shutil
import sys
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Colormap, LinearSegmentedColormap, ListedColormap


PAIR_TIMER_SOURCES = ("mpi", "cpu", "cuda")
RENDER_STYLES = ("heatmap", "plain")

_PAIR_HEAD = (
	r"^pair(?:_(mpi|cpu|cuda|gpu))? (\S+) -> (\S+) "
)

PAIR_LINE_RE = re.compile(
	_PAIR_HEAD
	+ r"avg_us=([0-9.eE+-]+)\s+"
	+ r"med_us=([0-9.eE+-]+)\s+"
	+ r"min_us=([0-9.eE+-]+)\s+"
	+ r"max_us=([0-9.eE+-]+)\s+"
	+ r"var_us=([0-9.eE+-]+)\s+"
	+ r"std_us=([0-9.eE+-]+)\s*$"
)

# Одна метрика (--stat avg|med|min|max|var|std).
PAIR_AVG_ONLY = re.compile(_PAIR_HEAD + r"avg_us=([0-9.eE+-]+)\s*$")
PAIR_MED_ONLY = re.compile(_PAIR_HEAD + r"med_us=([0-9.eE+-]+)\s*$")
PAIR_MIN_ONLY = re.compile(_PAIR_HEAD + r"min_us=([0-9.eE+-]+)\s*$")
PAIR_MAX_ONLY = re.compile(_PAIR_HEAD + r"max_us=([0-9.eE+-]+)\s*$")
PAIR_VAR_ONLY = re.compile(_PAIR_HEAD + r"var_us=([0-9.eE+-]+)\s*$")
PAIR_STD_ONLY = re.compile(_PAIR_HEAD + r"std_us=([0-9.eE+-]+)\s*$")
METRIC_KEYS = ("avg_us", "med_us", "min_us", "max_us", "var_us", "std_us")
METRIC_FILE_STEM = {
	"avg_us": "avg",
	"med_us": "med",
	"min_us": "min",
	"max_us": "max",
	"var_us": "var",
	"std_us": "std",
}
METRIC_CBAR_LABEL = {
	"avg_us": "мкс",
	"med_us": "мкс",
	"min_us": "мкс",
	"max_us": "мкс",
	"var_us": "мкс²",
	"std_us": "мкс",
}
METRIC_TITLE = {
	"avg_us": "Среднее арифметическое задержек",
	"med_us": "Медиана задержек",
	"min_us": "Минимальное задержек",
	"max_us": "Максимальное задержек",
	"var_us": "Выборочная дисперсия задержек",
	"std_us": "Стандартное отклонение задержек",
}


RANKS_LINE_RE = re.compile(r"^Ranks:\s*(\d+)")
RANK_MAP_LINE_RE = re.compile(r"^r(\d+)\s+hostname=(\S+)")
BYTES_LINE_RE = re.compile(r"^Bytes:\s*(\d+)\s*$")
DECIMAL_MB = 1_000_000
WARMUP_LINE_RE = re.compile(r"^Warmup:\s*(\d+)\s*$")
ITERS_LINE_RE = re.compile(r"^Iters:\s*(\d+)\s*$")
TOTAL_TIME_LINE_RE = re.compile(r"^TotalTimeSec:\s*([0-9.eE+-]+)\s*$")
TIMER_LINE_RE = re.compile(r"^Timer:\s*(\S+)\s*$")
SIZE_LINE_RE = re.compile(r"^Size:\s*(\S+)\s*$")
SIDE_LINE_RE = re.compile(r"^Side:\s*(\S+)\s*$")
MEASURE_SIDE_LINE_RE = re.compile(r"^MeasureSide:\s*(\S+)\s*$")
REP_TAG_RE = re.compile(r"(?:^|_)rep(\d+)(?:_|$)", re.I)
ENV_TAG_RE = re.compile(r"(?:^|_)(auto|host)(?:_|$)", re.I)
MODE_TAG_RE = re.compile(r"(?:^|_)(one_to_one|all_to_all)(?:_|$)", re.I)
TIMER_TAG_RE = re.compile(r"(?:^|_)(?:timer|t)(mpi|cpu|cuda|all)(?:_|$)", re.I)
SIDE_TAG_RE = re.compile(
	r"(?:^|_)(?:size|side)?(sender|receiver|both|snd|rcv)(?:_|$)", re.I
)
BYTES_TAG_RE = re.compile(r"(?:^|_)b(\d+)(?:_|$)", re.I)
WARMUP_TAG_RE = re.compile(r"(?:^|_)w(\d+)(?:_|$)", re.I)
ITERS_TAG_RE = re.compile(r"(?:^|_)i(\d+)(?:_|$)", re.I)


LATENCY_GR = LinearSegmentedColormap.from_list(
	"latency_gr",
	["#045a2d", "#16a34a", "#f97316", "#dc2626", "#7f1d1d"],
	N=256,
)

def _short_host(hostname: str) -> str:
	return hostname.split(".", 1)[0].strip() or hostname

def rank_hosts_from_meta(meta: dict[str, Any], n: int) -> list[str] | None:
	'''Возвращает список узлов для каждого ранга'''
	hosts_map = meta.get("rank_hosts")
	if not isinstance(hosts_map, dict):
		return None
	rank_hosts: list[str] = []
	for r in range(n):
		host = hosts_map.get(r)
		if not isinstance(host, str) or not host:
			return None
		rank_hosts.append(host)
	return rank_hosts

def _compress_int_ranges(nums: list[int]) -> str:
	'''Сжимает список чисел в диапазоны'''
	if not nums:
		return ""
	sorted_unique = sorted(set(nums))
	ranges: list[str] = []
	start = sorted_unique[0]
	prev = sorted_unique[0]
	for x in sorted_unique[1:]:
		if x == prev + 1:
			prev = x
			continue
		ranges.append(f"{start}-{prev}" if start != prev else f"{start}")
		start = x
		prev = x
	ranges.append(f"{start}-{prev}" if start != prev else f"{start}")
	return ",".join(ranges)

def nodes_title(rank_hosts: list[str]) -> str:
	'''Возвращает заголовок для узлов'''
	short = [_short_host(h) for h in rank_hosts]
	if not short:
		return ""
	uniq: list[str] = []
	for h in short:
		if h not in uniq:
			uniq.append(h)
	if len(uniq) == 1:
		return f"Узел: {uniq[0]}"

	m = [re.match(r"^([A-Za-z_]+)(\d+)$", h) for h in uniq]
	if all(mm is not None for mm in m):
		prefixes = {mm.group(1) for mm in m if mm}
		if len(prefixes) == 1:
			prefix = next(iter(prefixes))
			nums = [int(mm.group(2)) for mm in m if mm]
			return f"Узлы: {prefix}[{_compress_int_ranges(nums)}]"
	return f"Узлы: {','.join(uniq)}"

def axis_labels_by_node(rank_hosts: list[str]) -> list[str]:
	'''Формирует метки для узлов на оси X и Y'''
	short = [_short_host(h) for h in rank_hosts]
	seen: dict[str, int] = {}
	labels: list[str] = []
	for h in short:
		local_idx = seen.get(h, 0)
		seen[h] = local_idx + 1
		m = re.search(r"(\d+)$", h)
		node_id = m.group(1) if m else h
		labels.append(f"{node_id}.{local_idx}")
	return labels

def pair_label_to_rank(label: str, meta: dict[str, Any]) -> int | None:
	'''Возвращает ранг для заданного узла'''
	if re.fullmatch(r"\d+", label):
		return int(label)
	cache = meta.get("_pair_label_to_rank")
	if isinstance(cache, dict):
		v = cache.get(label)
		if isinstance(v, int):
			return v
	n = meta.get("ranks")
	if not isinstance(n, int):
		return None
	rank_hosts = rank_hosts_from_meta(meta, n)
	if not rank_hosts:
		return None
	mapping = {lab: i for i, lab in enumerate(axis_labels_by_node(rank_hosts))}
	meta["_pair_label_to_rank"] = mapping
	v = mapping.get(label)
	return v if isinstance(v, int) else None

def pair_source_of_match(m: re.Match[str], meta: dict[str, Any]) -> str:
	src = m.group(1)
	if src is None:
		src = str(meta.get("timer", "mpi"))
	src = src.lower()
	if src == "gpu":
		return "cuda"
	return src

def pair_source_allowed(m: re.Match[str], timer_source: str, meta: dict[str, Any]) -> bool:
	return pair_source_of_match(m, meta) == timer_source

def host_stem_for_output(meta: dict[str, Any], rank_hosts: list[str] | None) -> str:
	if rank_hosts:
		short = [_short_host(h) for h in rank_hosts]
		uniq: list[str] = []
		for s in short:
			if s not in uniq:
				uniq.append(s)
		return re.sub(r"[^0-9A-Za-z_.-]", "_", "_".join(uniq))
	hostname = str(meta.get("hostname", "host-unknown"))
	return re.sub(r"[^0-9A-Za-z_.-]", "_", hostname)

def node_boundaries(rank_hosts: list[str]) -> list[float]:
	'''Находит границы узлов на оси X и Y для дальнейшей их отрисовки'''
	bounds: list[float] = []
	for i in range(len(rank_hosts) - 1):
		if rank_hosts[i] != rank_hosts[i + 1]:
			bounds.append(float(i) + 0.5)
	return bounds

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
	timer_source: str,
) -> tuple[dict[str, Any], list[tuple[int, int, list[float]]]]:
	meta: dict[str, Any] = {}
	pairs: list[tuple[int, int, list[float]]] = []

	for raw in text.splitlines():
		line = raw.strip()
		if line.startswith("Env:"):
			meta["env"] = line.split(":", 1)[1].strip()
		elif line.startswith("Mode:"):
			meta["mode"] = line.split(":", 1)[1].strip()
		elif line.startswith("Ranks:"):
			m = RANKS_LINE_RE.match(line)
			if m:
				meta["ranks"] = int(m.group(1))
		elif (m := RANK_MAP_LINE_RE.match(line)):
			r = int(m.group(1))
			h = m.group(2)
			meta.setdefault("rank_hosts", {})[r] = h
		elif (m := BYTES_LINE_RE.match(line)):
			meta["bytes"] = int(m.group(1))
		elif (m := WARMUP_LINE_RE.match(line)):
			meta["warmup"] = int(m.group(1))
		elif (m := ITERS_LINE_RE.match(line)):
			meta["iters"] = int(m.group(1))
		elif (m := TOTAL_TIME_LINE_RE.match(line)):
			meta["total_elapsed_s"] = float(m.group(1))
		elif (m := TIMER_LINE_RE.match(line)):
			meta["timer"] = normalize_timer_token(m.group(1))
		elif (m := SIZE_LINE_RE.match(line)):
			meta["size"] = normalize_size_token(m.group(1))
		elif (m := SIDE_LINE_RE.match(line)):
			meta["size"] = normalize_size_token(m.group(1))
		elif (m := MEASURE_SIDE_LINE_RE.match(line)):
			meta["size"] = normalize_size_token(m.group(1))
		else:
			m = PAIR_LINE_RE.match(line)
			if m and pair_source_allowed(m, timer_source, meta):
				src_r = pair_label_to_rank(m.group(2), meta)
				dst_r = pair_label_to_rank(m.group(3), meta)
				if src_r is None or dst_r is None:
					continue
				std_v = float(m.group(9)) if m.group(9) is not None else math.nan
				vals = [float(m.group(i)) for i in range(4, 9)] + [std_v]
				pairs.append((src_r, dst_r, vals))
				continue
			nan6 = [math.nan, math.nan, math.nan, math.nan, math.nan, math.nan]
			if (m := PAIR_AVG_ONLY.match(line)) and pair_source_allowed(m, timer_source, meta):
				src_r = pair_label_to_rank(m.group(2), meta)
				dst_r = pair_label_to_rank(m.group(3), meta)
				if src_r is None or dst_r is None:
					continue
				v = float(m.group(4))
				vals = [v, nan6[1], nan6[2], nan6[3], nan6[4], nan6[5]]
				pairs.append((src_r, dst_r, vals))
				continue
			if (m := PAIR_MED_ONLY.match(line)) and pair_source_allowed(m, timer_source, meta):
				src_r = pair_label_to_rank(m.group(2), meta)
				dst_r = pair_label_to_rank(m.group(3), meta)
				if src_r is None or dst_r is None:
					continue
				v = float(m.group(4))
				vals = [nan6[0], v, nan6[2], nan6[3], nan6[4], nan6[5]]
				pairs.append((src_r, dst_r, vals))
				continue
			if (m := PAIR_MIN_ONLY.match(line)) and pair_source_allowed(m, timer_source, meta):
				src_r = pair_label_to_rank(m.group(2), meta)
				dst_r = pair_label_to_rank(m.group(3), meta)
				if src_r is None or dst_r is None:
					continue
				v = float(m.group(4))
				vals = [nan6[0], nan6[1], v, nan6[3], nan6[4], nan6[5]]
				pairs.append((src_r, dst_r, vals))
				continue
			if (m := PAIR_MAX_ONLY.match(line)) and pair_source_allowed(m, timer_source, meta):
				src_r = pair_label_to_rank(m.group(2), meta)
				dst_r = pair_label_to_rank(m.group(3), meta)
				if src_r is None or dst_r is None:
					continue
				v = float(m.group(4))
				vals = [nan6[0], nan6[1], nan6[2], v, nan6[4], nan6[5]]
				pairs.append((src_r, dst_r, vals))
				continue
			if (m := PAIR_VAR_ONLY.match(line)) and pair_source_allowed(m, timer_source, meta):
				src_r = pair_label_to_rank(m.group(2), meta)
				dst_r = pair_label_to_rank(m.group(3), meta)
				if src_r is None or dst_r is None:
					continue
				v = float(m.group(4))
				vals = [nan6[0], nan6[1], nan6[2], nan6[3], v, nan6[5]]
				pairs.append((src_r, dst_r, vals))
				continue
			if (m := PAIR_STD_ONLY.match(line)) and pair_source_allowed(m, timer_source, meta):
				src_r = pair_label_to_rank(m.group(2), meta)
				dst_r = pair_label_to_rank(m.group(3), meta)
				if src_r is None or dst_r is None:
					continue
				v = float(m.group(4))
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
def title_block(
	meta: dict[str, Any],
	metric: str,
	*,
	n: int,
	rank_hosts: list[str] | None,
) -> str:
	env = str(meta.get("env", "unknown"))
	mode = str(meta.get("mode", "unknown"))
	title_line = METRIC_TITLE.get(metric, metric)
	parts: list[str] = []
	if rank_hosts:
		parts.append(nodes_title(rank_hosts))
	elif n <= 1 and "hostname" in meta:
		parts.append(f"Узел: {_short_host(str(meta['hostname']))}")
	parts.extend(
		[
			title_line,
			f"Режим запуска: {mode}",
			f"Среда копирования: {env}",
		]
	)
	return "\n".join(parts)


"""Форматирует размер в десятичных MB."""
def format_size_mb_decimal(b: int) -> str:
	mb = b / DECIMAL_MB
	text = f"{mb:.12f}".rstrip("0").rstrip(".")
	return f"size: {text} MB"


def normalize_timer_token(value: str) -> str:
	v = value.strip().lower()
	if v == "gpu":
		return "cuda"
	return v


def normalize_size_token(value: str) -> str:
	v = value.strip().lower()
	if v in ("sender", "snd"):
		return "snd"
	if v in ("receiver", "rcv"):
		return "rcv"
	return v


def run_tags(source_path: Path | None, meta: dict[str, Any], timer_source: str) -> dict[str, str]:
	stem = source_path.stem if source_path is not None else ""
	env_meta = str(meta.get("env", "unknown")).lower()
	env_default = "host" if env_meta == "host" else ("auto" if env_meta == "gpudirect" else env_meta)
	mode_default = str(meta.get("mode", "unknown")).lower()
	timer_default = normalize_timer_token(str(meta.get("timer", timer_source)))
	size_default = normalize_size_token(str(meta.get("size", "na")))

	def pick(pattern: re.Pattern[str], fallback: str) -> str:
		m = pattern.search(stem)
		return m.group(1) if m else fallback

	return {
		"rep": pick(REP_TAG_RE, "na"),
		"env": pick(ENV_TAG_RE, env_default),
		"mode": pick(MODE_TAG_RE, mode_default),
		"timer": normalize_timer_token(pick(TIMER_TAG_RE, timer_default)),
		"size": normalize_size_token(pick(SIDE_TAG_RE, size_default)),
		"b": pick(BYTES_TAG_RE, str(meta.get("bytes", "na"))),
		"w": pick(WARMUP_TAG_RE, str(meta.get("warmup", "na"))),
		"i": pick(ITERS_TAG_RE, str(meta.get("iters", "na"))),
	}


def total_time_line(meta: dict[str, Any]) -> str:
	v = meta.get("total_elapsed_s")
	if isinstance(v, (int, float)):
		return f"затраченное время: {float(v):.6f} сек"
	return "затраченное время: n/a"


def generation_timestamp() -> str:
	"""Дата/время рендера: дд/мм/гггг чч:мм:сс и смещение от UTC (+03:00)."""
	dt = datetime.now().astimezone()
	base = dt.strftime("%d/%m/%Y %H:%M:%S")
	raw = dt.strftime("%z")
	if not raw:
		return base
	tz_off = f"{raw[0]}{raw[1:3]}:{raw[3:5]}"
	return f"{base}{tz_off}"


"""Собирает блок параметров под графиком."""
def param_block(
	meta: dict[str, Any], tags: dict[str, str], total_time: str, creation_time: str
) -> str:
	lines: list[str] = []
	lines.append(f"timer: {tags['timer']}  size: {tags['size']}")
	lines.append(f"w: {tags['w']}  i: {tags['i']}")
	lines.append(f"b: {tags['b']} байт")
	lines.append(f"сгенерировано: {creation_time}")
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


def annotate_cells(
	ax: Any, matrix: np.ndarray, metric: str, *, text_color: str = "white"
) -> None:
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
				color=text_color,
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
	node_bounds: list[float],
	render_style: str,
) -> None:
	labels = tick_labels
	axis_fs = 9
	fig_w = max(5.0, 0.45 * matrix.shape[1] + 2.5)
	fig_h = max(4.5, 0.45 * matrix.shape[0] + 2.4)
	fig, ax = plt.subplots(figsize=(fig_w, fig_h))

	masked = np.ma.masked_invalid(matrix)
	# origin="lower": ряд/ранг 0 внизу, ось ординат растёт снизу вверх (как привычные координаты).
	if render_style == "plain":
		white_bg = np.zeros(matrix.shape, dtype=float)
		im = ax.imshow(
			white_bg,
			cmap=ListedColormap(["#ffffff"]),
			vmin=0.0,
			vmax=1.0,
			aspect="equal",
			interpolation="nearest",
			origin="lower",
		)
		text_color = "black"
	else:
		im = ax.imshow(
			masked, cmap=cmap, aspect="equal", interpolation="nearest", origin="lower"
		)
		text_color = "white"

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

	annotate_cells(ax, matrix, metric, text_color=text_color)
	if render_style == "plain":
		# Рисуем сетку ячеек, чтобы картинка выглядела как таблица.
		ax.set_xticks(np.arange(-0.5, matrix.shape[1], 1), minor=True)
		ax.set_yticks(np.arange(-0.5, matrix.shape[0], 1), minor=True)
		ax.grid(which="minor", color="#000000", linewidth=0.6, alpha=1.0)
		ax.tick_params(which="minor", bottom=False, left=False)

	for pos in node_bounds:
		ax.axvline(
			pos,
			color="#000000",
			linewidth=1.0,
			alpha=1.0,
			antialiased=False,
			snap=True,
			solid_capstyle="butt",
		)
		ax.axhline(
			pos,
			color="#000000",
			linewidth=1.0,
			alpha=1.0,
			antialiased=False,
			snap=True,
			solid_capstyle="butt",
		)

	if render_style != "plain":
		cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
		cbar.set_label(METRIC_CBAR_LABEL.get(metric, "мкс"), rotation=0, labelpad=12)

	fig.tight_layout()

	fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
	plt.close(fig)


def render_one_text(
	text: str,
	out_dir: Path,
	*,
	label: str,
	source_path: Path | None = None,
	timer_source: str = "mpi",
	render_style: str = "heatmap",
) -> int:
	meta, pairs = parse_gpu_one_to_one_text(text, timer_source)
	if not pairs:
		print(
			f"gpu_render: [{label}] no matching 'pair_{timer_source} ...' lines found.",
			file=sys.stderr,
		)
		return 1

	n = matrix_size(meta, pairs)
	if n <= 0:
		print(f"gpu_render: [{label}] could not infer matrix size.", file=sys.stderr)
		return 1

	mats = fill_matrices(n, pairs)
	rank_hosts = rank_hosts_from_meta(meta, n)
	# Метки на осях: номер_узла.номер_гпу (если есть host map), иначе gpu0..gpuN.
	tick_labels = axis_labels_by_node(rank_hosts) if rank_hosts else [f"gpu{i}" for i in range(n)]
	if rank_hosts:
		node_bounds = node_boundaries(rank_hosts)
	else:
		node_bounds = []

	want = set(METRIC_KEYS)

	hostname = host_stem_for_output(meta, rank_hosts)

	out_dir.mkdir(parents=True, exist_ok=True)
	tags = run_tags(source_path, meta, timer_source)
	total_time = total_time_line(meta)
	creation_time = generation_timestamp()
	params = param_block(meta, tags, total_time, creation_time)
	cmap_resolved = resolve_colormap("latency_gr")

	written = 0
	for key in METRIC_KEYS:
		if key not in want:
			continue
		if bool(np.all(np.isnan(mats[key]))):
			continue
		stem = METRIC_FILE_STEM.get(key, key)
		out_file = out_dir / (
			f"{hostname}_rep{tags['rep']}"
			f"_size{tags['size']}_timer{timer_source}"
			f"_b{tags['b']}_w{tags['w']}_i{tags['i']}_{stem}.png"
		)
		draw_heatmap(
			mats[key],
			key,
			out_file,
			title_block=title_block(meta, key, n=n, rank_hosts=rank_hosts),
			param_block=params,
			cmap=cmap_resolved,
			dpi=150,
			tick_labels=tick_labels,
			node_bounds=node_bounds,
			render_style=render_style,
		)
		print(out_file)
		written += 1

	if written == 0:
		print(f"gpu_render: [{label}] no non-empty metric matrices to plot.", file=sys.stderr)
		return 1

	return 0


def _sort_raw_with_helper(raw_files: list[Path]) -> int:
	try:
		import sort_raw as srs
	except ImportError:
		try:
			import sort_raw_samples as srs  # type: ignore[import-not-found]
		except ImportError as e:
			print(f"gpu_render: cannot import sort_raw.py / sort_raw_samples.py: {e}", file=sys.stderr)
			return 1
	except Exception as e:  # noqa: BLE001
		print(f"gpu_render: cannot load raw sort helper: {e}", file=sys.stderr)
		return 1

	code = 0
	for raw_path in raw_files:
		try:
			dst, n = srs.process_file(raw_path)
			print(f"{raw_path} -> {dst} ({n} values)")
		except Exception as e:  # noqa: BLE001
			print(f"{raw_path}: error: {e}", file=sys.stderr)
			code = 1
	return code


def create_and_sort_raw_for_text(source_path: Path, out_dir: Path, sort_mode: str) -> int:
	# Собираем raw только для конкретного .txt прогона.
	pattern = f"{source_path.stem}_src*_dst*.raw"
	src_raw_files = sorted(p for p in source_path.parent.glob(pattern) if p.is_file())
	if not src_raw_files:
		print(
			f"gpu_render: [{source_path}] no matching raw files for pattern {pattern}",
			file=sys.stderr,
		)
		return 1

	raw_dir = out_dir / "raw"
	raw_dir.mkdir(parents=True, exist_ok=True)
	copied_raw: list[Path] = []
	for src in src_raw_files:
		dst = raw_dir / src.name
		shutil.copy2(src, dst)
		copied_raw.append(dst)
	print(f"gpu_render: copied {len(copied_raw)} raw files to {raw_dir}")

	if sort_mode != "sorted":
		return 0

	return _sort_raw_with_helper(copied_raw)


def main() -> int:
	p = argparse.ArgumentParser(
		description="Heatmaps from gpu text output (pair ... lines).",
		epilog=(
			"Arguments:\n"
			"  input            .txt file, directory with .txt files, or '-' for stdin\n"
			"  -o, --out-dir    output directory for PNG files\n"
			"  -t, --timer      choose timer source: mpi | cpu | cuda\n"
			"  --sort, --sorted enable raw sorting pipeline (default: none)\n\n"
		),
		formatter_class=argparse.RawTextHelpFormatter,
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
	p.add_argument(
		"-t",
		"--timer",
		dest="timer_source",
		choices=PAIR_TIMER_SOURCES,
		default="cuda",
		help="Источник времени в txt: mpi | cpu | cuda",
	)
	p.add_argument(
		"--sort",
		"--sorted",
		action="store_const",
		const="sorted",
		dest="sort_mode",
		default="none",
		help="Создать output/raw и отсортировать задержки через sort_raw_samples.py",
	)
	p.add_argument(
		"--style",
		choices=RENDER_STYLES,
		default="heatmap",
		help="Стиль рендера: heatmap (цвета + colorbar) | plain (белый фон, без colorbar)",
	)
	args = p.parse_args()

	in_path = args.input
	if in_path == "-":
		text = sys.stdin.read()
		if args.sort_mode == "sorted":
			print("gpu_render: stdin input cannot create raw files.", file=sys.stderr)
			return 1
		return render_one_text(
			text,
			args.out_dir,
			label="stdin",
			source_path=None,
			timer_source=args.timer_source,
			render_style=args.style,
		)

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
			r = render_one_text(
				text,
				sub,
				label=str(txt),
				source_path=txt,
				timer_source=args.timer_source,
				render_style=args.style,
			)
			if r == 0 and args.sort_mode == "sorted":
				rr = create_and_sort_raw_for_text(txt, sub, args.sort_mode)
				if rr != 0:
					code = 1
			if r != 0:
				code = 1
		return code

	if path.is_file():
		text = path.read_text(encoding="utf-8", errors="replace")
		r = render_one_text(
			text,
			args.out_dir,
			label=str(path),
			source_path=path,
			timer_source=args.timer_source,
			render_style=args.style,
		)
		if r != 0:
			return r
		if args.sort_mode != "sorted":
			return 0
		return create_and_sort_raw_for_text(path, args.out_dir, args.sort_mode)

	print(f"gpu_render: not a file or directory: {path}", file=sys.stderr)
	return 1


if __name__ == "__main__":
	raise SystemExit(main())
