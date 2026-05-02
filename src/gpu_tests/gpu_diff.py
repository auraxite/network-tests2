#!/usr/bin/env python3

from __future__ import annotations

"""Поэлементная разность двух матриц задержек из gpu бенчмарка.

Использование:
  gpu_diff.py A.txt B.txt -o out_dir [--style heatmap|plain] [--rel]

На вход подаются два .txt лога одного формата (как у gpu_heatmap.py).
Скрипт парсит pair-строки в обоих, строит матрицы для каждой метрики и
сохраняет PNG c матрицей разности (A - B). Это удобно, когда хочется на
одной картинке увидеть, насколько одна конфигурация быстрее/медленнее
другой (например, env=auto против env=host, разные итерации, разные
прогоны).

Цветовая схема heatmap — диверентная (синий = A быстрее, красный = A
медленнее), центр шкалы фиксирован на нуле. В стиле plain единица
дописана в заголовок, как в gpu_heatmap.

Если матрицы разной формы (по числу рангов), выводится ошибка и работа
прекращается: сравнивать матрицы разных топологий смысла нет.
"""

import argparse
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import (
	Colormap,
	LinearSegmentedColormap,
	ListedColormap,
	TwoSlopeNorm,
)

# Переиспользуем разбор и общие утилиты у gpu_heatmap — он лежит рядом.
import gpu_heatmap as gr


# Диверентная палитра: A быстрее (отрицательное dt) — холодные тона,
# A медленнее (положительное dt) — тёплые тона, ноль — белый.
DIFF_CMAP_NAME = "diff_bwr"
DIFF_CMAP = LinearSegmentedColormap.from_list(
	DIFF_CMAP_NAME,
	["#1d4ed8", "#93c5fd", "#ffffff", "#fca5a5", "#b91c1c"],
	N=256,
)


def resolve_diff_cmap() -> Colormap:
	cmap = DIFF_CMAP.copy()
	cmap.set_bad(color="white", alpha=1.0)
	return cmap


def diff_title_block(
	meta_a: dict[str, Any],
	meta_b: dict[str, Any],
	metric: str,
	*,
	rank_hosts: list[str] | None,
	render_style: str,
	relative: bool,
) -> str:
	"""Заголовок: что за метрика, какие два прогона сравниваем, единица."""
	base_title = gr.METRIC_TITLE.get(metric, metric)
	title_line = f"Разность: {base_title.lower()} (A − B)"
	if render_style == "plain":
		unit = "%" if relative else gr.METRIC_CBAR_LABEL.get(metric, "мкс")
		title_line = f"{title_line} ({unit})"

	def short_descr(meta: dict[str, Any]) -> str:
		bits = []
		if "mode" in meta:
			bits.append(f"mode={meta['mode']}")
		if "env" in meta:
			bits.append(f"env={meta['env']}")
		if "bytes" in meta:
			bits.append(f"b={meta['bytes']}")
		return " ".join(bits) if bits else "?"

	parts: list[str] = []
	if rank_hosts:
		parts.append(gr.nodes_title(rank_hosts))
	parts.append(title_line)
	parts.append(f"A: {short_descr(meta_a)}")
	parts.append(f"B: {short_descr(meta_b)}")
	return "\n".join(parts)


def diff_param_block(
	tags_a: dict[str, str],
	tags_b: dict[str, str],
	creation_time: str,
	relative: bool,
) -> str:
	"""Краткий блок параметров под графиком: w/i/b у A и B + время рендера."""
	def line_for(tag: dict[str, str], label: str) -> str:
		return (
			f"{label}: w={tag['w']} i={tag['i']} b={tag['b']} байт"
		)

	lines: list[str] = []
	lines.append(line_for(tags_a, "A"))
	lines.append(line_for(tags_b, "B"))
	if relative:
		lines.append("значения: относительная разность, (A − B) / B · 100%")
	else:
		lines.append("значения: абсолютная разность, A − B")
	lines.append(f"сгенерировано: {creation_time}")
	return "\n".join(lines)


def format_diff_value(metric: str, v: float, *, relative: bool) -> str:
	if relative:
		av = abs(v)
		if av >= 100:
			return f"{v:+.0f}"
		if av >= 10:
			return f"{v:+.1f}"
		return f"{v:+.2f}"
	av = abs(v)
	if metric == "var_us":
		return f"{v:+.2g}" if av >= 1000 else f"{v:+.1f}"
	if av >= 1000:
		return f"{v:+.0f}"
	if av >= 100:
		return f"{v:+.1f}"
	if av >= 10:
		return f"{v:+.2f}"
	return f"{v:+.3f}"


def annotate_diff_cells(
	ax: Any,
	matrix: np.ndarray,
	metric: str,
	*,
	relative: bool,
	text_color: str,
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
				format_diff_value(metric, float(v), relative=relative),
				ha="center",
				va="center",
				fontsize=fontsize,
				color=text_color,
			)


def draw_diff_heatmap(
	matrix: np.ndarray,
	metric: str,
	out_path: Path,
	*,
	title_block: str,
	param_block: str,
	dpi: int,
	tick_labels: list[str],
	node_bounds: list[float],
	render_style: str,
	relative: bool,
) -> None:
	"""Рисует и сохраняет одну матрицу разности.

	В стиле heatmap используется TwoSlopeNorm с центром в нуле, чтобы
	0 всегда был белым, а абсолютные величины слева/справа были видны на
	одной шкале. В стиле plain рисуем чёрный текст на белом фоне с
	таблицей-сеткой (как и у gpu_heatmap).
	"""
	axis_fs = 9
	fig_w = max(5.0, 0.45 * matrix.shape[1] + 2.5)
	fig_h = max(4.5, 0.45 * matrix.shape[0] + 2.4)
	fig, ax = plt.subplots(figsize=(fig_w, fig_h))

	masked = np.ma.masked_invalid(matrix)

	if render_style == "plain":
		white_bg = np.zeros(matrix.shape, dtype=float)
		_ = ax.imshow(
			white_bg,
			cmap=ListedColormap(["#ffffff"]),
			vmin=0.0,
			vmax=1.0,
			aspect="equal",
			interpolation="nearest",
			origin="upper",
		)
		text_color = "black"
		im = None
	else:
		cmap = resolve_diff_cmap()
		# симметричная шкала вокруг нуля; если все значения одного знака,
		# TwoSlopeNorm всё равно требует vmin<center<vmax — расширим до 1
		finite = matrix[np.isfinite(matrix)]
		if finite.size == 0:
			vmin, vmax = -1.0, 1.0
		else:
			a = float(np.nanmin(finite))
			b = float(np.nanmax(finite))
			lim = max(abs(a), abs(b), 1e-9)
			vmin, vmax = -lim, lim
		norm = TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
		im = ax.imshow(
			masked,
			cmap=cmap,
			norm=norm,
			aspect="equal",
			interpolation="nearest",
			origin="upper",
		)
		text_color = "black"

	ax.set_xticks(range(len(tick_labels)))
	ax.set_yticks(range(len(tick_labels)))
	ax.set_xticklabels(tick_labels, rotation=0, fontsize=axis_fs)
	ax.set_yticklabels(tick_labels, fontsize=axis_fs)

	ax.set_xlabel(f"dst GPU\n\n{param_block}", fontsize=axis_fs, labelpad=4)
	ax.set_ylabel("src GPU", fontsize=axis_fs, labelpad=4)
	ax.set_title(title_block, fontsize=9)

	annotate_diff_cells(ax, matrix, metric, relative=relative, text_color=text_color)

	if render_style == "plain":
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

	if render_style != "plain" and im is not None:
		cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
		unit = "%" if relative else gr.METRIC_CBAR_LABEL.get(metric, "мкс")
		cbar.set_label(f"Δ, {unit}", rotation=0, labelpad=12)

	fig.tight_layout()
	fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
	plt.close(fig)


def load_matrices(
	path: Path,
) -> tuple[dict[str, Any], dict[str, np.ndarray], int, list[str] | None]:
	"""Парсит .txt и возвращает (meta, matrices_by_metric, n, rank_hosts)."""
	text = path.read_text(encoding="utf-8", errors="replace")
	meta, pairs = gr.parse_gpu_one_to_one_text(text)
	if not pairs:
		raise RuntimeError(f"{path}: no pair lines found")
	n = gr.matrix_size(meta, pairs)
	if n <= 0:
		raise RuntimeError(f"{path}: cannot infer matrix size")
	mats = gr.fill_matrices(n, pairs)
	rank_hosts = gr.rank_hosts_from_meta(meta, n)
	return meta, mats, n, rank_hosts


def compute_diff(
	a: np.ndarray, b: np.ndarray, *, relative: bool
) -> np.ndarray:
	"""A - B (или (A-B)/B*100% при relative). NaN-ячейки распространяются."""
	out = np.full_like(a, np.nan, dtype=float)
	mask = np.isfinite(a) & np.isfinite(b)
	if relative:
		denom_ok = mask & (b != 0.0)
		out[denom_ok] = (a[denom_ok] - b[denom_ok]) / b[denom_ok] * 100.0
	else:
		out[mask] = a[mask] - b[mask]
	return out


def diff_one_pair(
	path_a: Path,
	path_b: Path,
	out_dir: Path,
	*,
	render_style: str,
	relative: bool,
) -> int:
	meta_a, mats_a, n_a, hosts_a = load_matrices(path_a)
	meta_b, mats_b, n_b, hosts_b = load_matrices(path_b)

	if n_a != n_b:
		print(
			f"gpu_diff: matrix sizes differ: A={n_a} (in {path_a}), "
			f"B={n_b} (in {path_b}); refusing to subtract.",
			file=sys.stderr,
		)
		return 1
	if hosts_a and hosts_b and hosts_a != hosts_b:
		print(
			f"gpu_diff: WARNING: rank→host map differs between A and B. "
			"Subtracting cell-wise; ось подписей берётся из A.",
			file=sys.stderr,
		)

	rank_hosts = hosts_a or hosts_b
	tick_labels = (
		gr.axis_labels_by_node(rank_hosts) if rank_hosts else [f"gpu{i}" for i in range(n_a)]
	)
	node_bounds = gr.node_boundaries(rank_hosts) if rank_hosts else []

	out_dir.mkdir(parents=True, exist_ok=True)
	tags_a = gr.run_tags(path_a, meta_a)
	tags_b = gr.run_tags(path_b, meta_b)
	creation_time = gr.generation_timestamp()
	pblock = diff_param_block(tags_a, tags_b, creation_time, relative)

	stem_pair = f"{path_a.stem}_minus_{path_b.stem}"
	suffix = "_rel" if relative else "_abs"

	written = 0
	for key in gr.METRIC_KEYS:
		if bool(np.all(np.isnan(mats_a[key]))) or bool(np.all(np.isnan(mats_b[key]))):
			continue
		diff = compute_diff(mats_a[key], mats_b[key], relative=relative)
		if bool(np.all(np.isnan(diff))):
			continue
		stem = gr.METRIC_FILE_STEM.get(key, key)
		out_file = out_dir / f"{stem_pair}_{stem}{suffix}.png"
		title = diff_title_block(
			meta_a,
			meta_b,
			key,
			rank_hosts=rank_hosts,
			render_style=render_style,
			relative=relative,
		)
		draw_diff_heatmap(
			diff,
			key,
			out_file,
			title_block=title,
			param_block=pblock,
			dpi=150,
			tick_labels=tick_labels,
			node_bounds=node_bounds,
			render_style=render_style,
			relative=relative,
		)
		print(out_file)
		written += 1

	if written == 0:
		print(
			f"gpu_diff: [{path_a} vs {path_b}] no comparable metrics.",
			file=sys.stderr,
		)
		return 1
	return 0


def main() -> int:
	p = argparse.ArgumentParser(
		description=(
			"Поэлементная разность двух матриц задержек (A.txt − B.txt)."
		),
		epilog=(
			"Примеры:\n"
			"  gpu_diff.py auto.txt host.txt -o diff_out\n"
			"  gpu_diff.py auto.txt host.txt -o diff_out --rel --style plain\n"
		),
		formatter_class=argparse.RawTextHelpFormatter,
	)
	p.add_argument("a", type=Path, help="Файл A.txt (уменьшаемое)")
	p.add_argument("b", type=Path, help="Файл B.txt (вычитаемое)")
	p.add_argument(
		"-o",
		"--out-dir",
		type=Path,
		default=Path("."),
		help="Каталог для PNG (создаётся при необходимости)",
	)
	p.add_argument(
		"--style",
		choices=gr.RENDER_STYLES,
		default="heatmap",
		help="Стиль рендера: heatmap (диверентная палитра + colorbar) | plain (без цветов)",
	)
	p.add_argument(
		"--rel",
		"--relative",
		dest="relative",
		action="store_true",
		help="Считать относительную разность (A − B) / B · 100%% вместо абсолютной",
	)
	args = p.parse_args()

	if not args.a.is_file():
		print(f"gpu_diff: not a file: {args.a}", file=sys.stderr)
		return 1
	if not args.b.is_file():
		print(f"gpu_diff: not a file: {args.b}", file=sys.stderr)
		return 1

	return diff_one_pair(
		args.a,
		args.b,
		args.out_dir,
		render_style=args.style,
		relative=args.relative,
	)


if __name__ == "__main__":
	raise SystemExit(main())
