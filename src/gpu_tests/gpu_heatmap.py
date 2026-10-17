#!/usr/bin/env python3

from __future__ import annotations

"""Тепловые карты по результатам gpu_tests (.txt).

Аргументы:
  input       .txt, каталог с .txt или '-' (stdin)
  -o, --out   каталог для PNG
  --style     heatmap (цвет + шкала) | plain (белый фон, таблица чисел)
"""

import argparse
import math
import re
import sys
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Colormap, LinearSegmentedColormap, ListedColormap

from gpu_io import (  # noqa: F401 — часть имён реэкспортируется для внешних скриптов
	BYTES_TAG_RE,
	ENV_TAG_RE,
	ITERS_TAG_RE,
	METRIC_CBAR_LABEL,
	METRIC_FILE_STEM,
	METRIC_KEYS,
	METRIC_TITLE,
	MODE_TAG_RE,
	REP_TAG_RE,
	WARMUP_TAG_RE,
	axis_labels_by_node,
	generation_timestamp,
	pair_label_to_rank,
	parse_gpu_one_to_one_text,
	rank_hosts_from_meta,
	short_host,
)

RENDER_STYLES = ("heatmap", "plain")

LATENCY_GR = LinearSegmentedColormap.from_list(
	"latency_gr",
	["#045a2d", "#16a34a", "#f97316", "#dc2626", "#7f1d1d"],
	N=256,
)


def _compress_int_ranges(nums: list[int]) -> str:
	"""[1, 2, 3, 7] -> '1-3,7'."""
	ranges: list[str] = []
	for x in sorted(set(nums)):
		if ranges and x == int(ranges[-1].split("-")[-1]) + 1:
			ranges[-1] = f"{ranges[-1].split('-')[0]}-{x}"
		else:
			ranges.append(str(x))
	return ",".join(ranges)


def nodes_title(rank_hosts: list[str]) -> str:
	"""«Узел: g8600-1» или «Узлы: g5500-[2,4]»."""
	uniq = list(dict.fromkeys(short_host(h) for h in rank_hosts))
	if not uniq:
		return ""
	if len(uniq) == 1:
		return f"Узел: {uniq[0]}"
	m = [re.match(r"^([A-Za-z_]+)(\d+)$", h) for h in uniq]
	if all(m) and len({mm.group(1) for mm in m}) == 1:
		nums = [int(mm.group(2)) for mm in m]
		return f"Узлы: {m[0].group(1)}[{_compress_int_ranges(nums)}]"
	return f"Узлы: {','.join(uniq)}"


def node_boundaries(rank_hosts: list[str]) -> list[float]:
	"""Границы между узлами на осях."""
	return [i + 0.5 for i in range(len(rank_hosts) - 1) if rank_hosts[i] != rank_hosts[i + 1]]


def resolve_colormap(name: str) -> Colormap:
	"""Палитра и белый цвет для NaN."""
	base = LATENCY_GR if name == "latency_gr" else (
		mpl.colormaps[name] if hasattr(mpl, "colormaps") else mpl.cm.get_cmap(name))
	cmap = base.copy()
	cmap.set_bad(color="white", alpha=1.0)
	return cmap


def matrix_size(meta: dict[str, Any], pairs: list[tuple[int, int, list[float]]]) -> int:
	if "ranks" in meta:
		return int(meta["ranks"])
	return max((max(s, d) for s, d, _ in pairs), default=-1) + 1


def fill_matrices(n: int, pairs: list[tuple[int, int, list[float]]]) -> dict[str, np.ndarray]:
	"""Матрицы n×n по каждой величине. Диагональ и пропуски — NaN."""
	mats = {k: np.full((n, n), np.nan, dtype=float) for k in METRIC_KEYS}
	for src, dst, vals in pairs:
		if src != dst and 0 <= src < n and 0 <= dst < n:
			for k, v in zip(METRIC_KEYS, vals):
				if not math.isnan(v):
					mats[k][src, dst] = v
	return mats


def title_block(
	meta: dict[str, Any],
	metric: str,
	*,
	n: int,
	rank_hosts: list[str] | None,
	render_style: str = "heatmap",
) -> str:
	"""Заголовок картинки. В стиле plain единица дописывается в заголовок,
	в стиле heatmap она стоит на шкале."""
	title_line = METRIC_TITLE.get(metric, metric)
	node_scope = ""
	if rank_hosts:
		nodes = nodes_title(rank_hosts)
		node_scope = nodes.split(": ", 1)[1] if ": " in nodes else nodes
	elif n <= 1 and "hostname" in meta:
		node_scope = short_host(str(meta["hostname"]))
	if node_scope:
		title_line = f"{title_line} для {node_scope}"
	if render_style == "plain":
		title_line = f"{title_line} ({METRIC_CBAR_LABEL.get(metric, 'мкс')})"
	return "\n".join([
		title_line,
		f"Режим передачи: {meta.get('mode', 'unknown')}",
		f"Транспорт: {meta.get('transport', 'unknown')}",
		f"Режим копирования: {meta.get('env', 'unknown')}",
	])


def run_tags(source_path: Path | None, meta: dict[str, Any]) -> dict[str, str]:
	"""Параметры прогона из имени файла, при отсутствии — из шапки."""
	stem = source_path.stem if source_path is not None else ""
	env_meta = str(meta.get("env", "unknown")).lower()
	env_default = "host" if env_meta == "host" else ("auto" if env_meta == "gpudirect" else env_meta)

	def pick(pattern: re.Pattern[str], fallback: str) -> str:
		m = pattern.search(stem)
		return m.group(1) if m else fallback

	return {
		"rep": pick(REP_TAG_RE, "na"),
		"env": pick(ENV_TAG_RE, env_default),
		"mode": pick(MODE_TAG_RE, str(meta.get("mode", "unknown")).lower()),
		"b": pick(BYTES_TAG_RE, str(meta.get("bytes", "na"))),
		"w": pick(WARMUP_TAG_RE, str(meta.get("warmup", "na"))),
		"i": pick(ITERS_TAG_RE, str(meta.get("iters", "na"))),
	}


def param_block(meta: dict[str, Any], tags: dict[str, str], creation_time: str) -> str:
	"""Блок параметров под картинкой."""
	return "\n".join([
		f"w: {tags['w']}  i: {tags['i']}",
		f"b: {tags['b']} байт",
		f"сгенерировано: {creation_time}",
	])


# Минимальный относительный разброс, при котором раскраска несёт смысл.
# Если фактический разброс меньше, верхнюю границу шкалы растягиваем до этого
# порога — тогда почти одинаковые значения и выглядят почти одинаково.
COLOR_SPREAD_FLOOR = 0.05

# Во сколько погрешностей должен укладываться разброс, чтобы раскраска
# считалась осмысленной. Меньше — значит различия между ячейками сопоставимы
# с погрешностью самих чисел, и цвет рисует не структуру, а случайность.
COLOR_ERR_SIGMAS = 3.0


def median_stderr(sigma: float | None, iters: int | None) -> float | None:
	"""Погрешность нарисованного числа.

	σ из результатов — разброс отдельных замеров, а на карте нарисована медиана
	по iters итерациям: её точность выше в √iters раз. Например, all_to_all с
	σ ≈ 1220 мкс при i = 200 даёт погрешность медианы около 108 мкс, то есть
	различия в 1700 мкс между парами — настоящие. 1.2533 = √(π/2), асимптотика
	для медианы.
	"""
	if sigma is None or iters is None or iters < 2:
		return None
	if not math.isfinite(sigma) or sigma <= 0:
		return None
	return 1.2533 * sigma / math.sqrt(iters)


def color_limits(
	matrix: np.ndarray, err: float | None = None
) -> tuple[float | None, float | None, float | None, str | None]:
	"""Границы цветовой шкалы, относительный разброс и причина растяжения.

	matplotlib по умолчанию растягивает палитру ровно на [min, max]. Для
	однородных матриц это врёт: разброс 0.6% раскрашивается так же контрастно,
	как разброс в два раза. Поэтому диапазон не опускается ниже
	COLOR_SPREAD_FLOOR от минимума и COLOR_ERR_SIGMAS погрешностей.

	Возвращает (vmin, vmax, rel, reason), reason — "spread" или "err",
	если шкалу растянули, иначе None.
	"""
	finite = matrix[np.isfinite(matrix)]
	if finite.size == 0:
		return None, None, None, None
	lo = float(finite.min())
	hi = float(finite.max())
	if hi <= lo:
		return None, None, 0.0, None

	span, reason = hi - lo, None
	rel = (hi - lo) / lo if lo > 0 else None
	if lo > 0 and lo * COLOR_SPREAD_FLOOR > span:
		span, reason = lo * COLOR_SPREAD_FLOOR, "spread"
	if err is not None and COLOR_ERR_SIGMAS * err > span:
		span, reason = COLOR_ERR_SIGMAS * err, "err"
	return lo, lo + span, rel, reason


def spread_caption(
	matrix: np.ndarray, metric: str, rel: float | None, err: float | None,
	*, reason: str | None
) -> str:
	"""Подпись с разбросом и погрешностью — чтобы цвет не приходилось додумывать."""
	finite = matrix[np.isfinite(matrix)]
	if finite.size == 0:
		return ""
	lo, hi = float(finite.min()), float(finite.max())
	if hi <= lo:
		return "все значения одинаковы"

	unit = METRIC_CBAR_LABEL.get(metric, "мкс")
	head = f"разброс {format_cell_value(metric, lo)}…{format_cell_value(metric, hi)} {unit}"
	if rel is not None:
		head += f" ({rel * 100:.1f}%)"
	if err is not None:
		head += f", погрешность ±{format_cell_value(metric, err)} {unit}"
	if reason == "err":
		head += "\nразличия в пределах погрешности — шкала приглушена"
	elif reason == "spread":
		head += f"\nразличия меньше {COLOR_SPREAD_FLOOR * 100:.0f}% — шкала приглушена"
	return head


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


def annotate_cells(ax: Any, matrix: np.ndarray, metric: str, *, text_color: str = "white") -> None:
	fontsize = max(5, min(9, int(90 / max(1, matrix.shape[0]))))
	for i in range(matrix.shape[0]):
		for j in range(matrix.shape[1]):
			if not np.isnan(matrix[i, j]):
				ax.text(j, i, format_cell_value(metric, float(matrix[i, j])),
				        ha="center", va="center", fontsize=fontsize, color=text_color)


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
	noise: float | None = None,
	iters: int | None = None,
) -> None:
	"""Рисует и сохраняет одну картинку."""
	axis_fs = 9
	fig_w = max(5.0, 0.45 * matrix.shape[1] + 2.5)
	fig_h = max(4.5, 0.45 * matrix.shape[0] + 2.4)
	fig, ax = plt.subplots(figsize=(fig_w, fig_h))

	err = median_stderr(noise, iters)
	vmin, vmax, rel, reason = color_limits(matrix, err)
	caption = spread_caption(
		matrix, metric, rel, err,
		# в стиле plain цвета нет, оговорка про шкалу там не к месту
		reason=reason if render_style != "plain" else None,
	)
	# origin="upper": ранг 0 сверху, как в таблице.
	if render_style == "plain":
		im = ax.imshow(np.zeros(matrix.shape, dtype=float), cmap=ListedColormap(["#ffffff"]),
		               vmin=0.0, vmax=1.0, aspect="equal", interpolation="nearest", origin="upper")
		text_color = "black"
	else:
		im = ax.imshow(np.ma.masked_invalid(matrix), cmap=cmap, aspect="equal",
		               interpolation="nearest", origin="upper", vmin=vmin, vmax=vmax)
		text_color = "white"

	ax.set_xticks(range(len(tick_labels)))
	ax.set_yticks(range(len(tick_labels)))
	ax.set_xticklabels(tick_labels, rotation=0, fontsize=axis_fs)
	ax.set_yticklabels(tick_labels, fontsize=axis_fs)
	ax.set_xlabel(f"dst GPU\n\n{param_block}" + (f"\n{caption}" if caption else ""),
	              fontsize=axis_fs, labelpad=4)
	ax.set_ylabel("src GPU", fontsize=axis_fs, labelpad=4)
	ax.set_title(title_block, fontsize=9)

	annotate_cells(ax, matrix, metric, text_color=text_color)
	if render_style == "plain":
		# сетка ячеек, чтобы картинка выглядела как таблица
		ax.set_xticks(np.arange(-0.5, matrix.shape[1], 1), minor=True)
		ax.set_yticks(np.arange(-0.5, matrix.shape[0], 1), minor=True)
		ax.grid(which="minor", color="#000000", linewidth=0.6, alpha=1.0)
		ax.tick_params(which="minor", bottom=False, left=False)

	line = dict(color="#000000", linewidth=1.0, alpha=1.0, antialiased=False, snap=True,
	            solid_capstyle="butt")
	for pos in node_bounds:
		ax.axvline(pos, **line)
		ax.axhline(pos, **line)

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
	render_style: str = "heatmap",
) -> int:
	meta, pairs = parse_gpu_one_to_one_text(text)
	if not pairs:
		print(f"gpu_heatmap: [{label}] no matching 'pair ...' lines found.", file=sys.stderr)
		return 1
	n = matrix_size(meta, pairs)
	if n <= 0:
		print(f"gpu_heatmap: [{label}] could not infer matrix size.", file=sys.stderr)
		return 1

	mats = fill_matrices(n, pairs)
	rank_hosts = rank_hosts_from_meta(meta, n)
	# Метки осей: номер_узла.номер_GPU, если есть Rank map, иначе gpu0..gpuN.
	tick_labels = axis_labels_by_node(rank_hosts) if rank_hosts else [f"gpu{i}" for i in range(n)]
	node_bounds = node_boundaries(rank_hosts) if rank_hosts else []

	out_dir.mkdir(parents=True, exist_ok=True)
	params = param_block(meta, run_tags(source_path, meta), generation_timestamp())
	cmap = resolve_colormap("latency_gr")
	iters = int(meta["iters"]) if isinstance(meta.get("iters"), (int, float)) else None

	# Типичный шум внутри ячейки: медиана std по парам. По нему видно, сопоставим
	# ли разброс между ячейками с разбросом внутри одного замера.
	std_finite = mats["std_us"][np.isfinite(mats["std_us"])]
	noise = float(np.median(std_finite)) if std_finite.size else None

	written = 0
	for key in METRIC_KEYS:
		if bool(np.all(np.isnan(mats[key]))):
			continue
		out_file = out_dir / f"{out_dir.name}_{METRIC_FILE_STEM[key]}.png"
		draw_heatmap(
			mats[key], key, out_file,
			title_block=title_block(meta, key, n=n, rank_hosts=rank_hosts, render_style=render_style),
			param_block=params,
			cmap=cmap,
			dpi=150,
			tick_labels=tick_labels,
			node_bounds=node_bounds,
			render_style=render_style,
			# для карт std и var шум — это их собственные значения
			noise=None if key in ("std_us", "var_us") else noise,
			iters=iters,
		)
		print(out_file)
		written += 1

	if written == 0:
		print(f"gpu_heatmap: [{label}] no non-empty metric matrices to plot.", file=sys.stderr)
		return 1
	return 0


def main() -> int:
	p = argparse.ArgumentParser(description="Heatmaps from gpu text output (pair ... lines).")
	p.add_argument("input", nargs="?", default="-",
	               help="Файл .txt, каталог с .txt или '-' для stdin")
	p.add_argument("-o", "--out", dest="out_dir", type=Path, default=Path("."),
	               help="Каталог для PNG; для каждого .txt из каталога создаётся подкаталог")
	p.add_argument("--style", choices=RENDER_STYLES, default="heatmap",
	               help="heatmap (цвет + шкала) | plain (белый фон, без шкалы)")
	args = p.parse_args()

	if args.input == "-":
		return render_one_text(sys.stdin.read(), args.out_dir, label="stdin",
		                       render_style=args.style)

	path = Path(args.input)
	if path.is_dir():
		txts = sorted(path.glob("*.txt"))
		if not txts:
			print(f"gpu_heatmap: no *.txt in {path}", file=sys.stderr)
			return 1
		code = 0
		for txt in txts:
			clean_stem = re.sub(r"_w\d+", "", re.sub(r"_i\d+", "", txt.stem))
			code |= render_one_text(txt.read_text(encoding="utf-8", errors="replace"),
			                        args.out_dir / clean_stem, label=str(txt),
			                        source_path=txt, render_style=args.style)
		return code
	if path.is_file():
		return render_one_text(path.read_text(encoding="utf-8", errors="replace"), args.out_dir,
		                       label=str(path), source_path=path, render_style=args.style)
	print(f"gpu_heatmap: path not found: {path.resolve()}", file=sys.stderr)
	return 1


if __name__ == "__main__":
	raise SystemExit(main())
