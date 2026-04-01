#!/usr/bin/env python3
"""
Строит heatmap-матрицы по текстовому выводу gpu_one_to_one (stdout или --out).

Зависимости: numpy, matplotlib (например: pip install numpy matplotlib).

Ожидаемые строки:
  pair g<SRC> -> g<DST> (r...:0 -> r...:0) avg_us=... median_us=... min_us=... max_us=... var_us=...

Для каждой метрики пишется отдельный PNG: <prefix>_<metric>.png
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Строка как у format_pair_line в gpu_one_to_one.cpp
PAIR_LINE_RE = re.compile(
    r"^pair g(\d+) -> g(\d+) \(r\d+:0 -> r\d+:0\) "
    r"avg_us=([0-9.eE+-]+) median_us=([0-9.eE+-]+) min_us=([0-9.eE+-]+) "
    r"max_us=([0-9.eE+-]+) var_us=([0-9.eE+-]+)\s*$"
)

RANKS_LINE_RE = re.compile(r"^Ranks:\s*(\d+)")


METRIC_KEYS = ("avg_us", "median_us", "min_us", "max_us", "var_us")


def parse_gpu_one_to_one_text(text: str) -> tuple[dict[str, str | int], list[tuple[int, int, list[float]]]]:
    """
    Возвращает meta (cuda_aware, mode, ranks?) и список
    (src_rank, dst_rank, [avg, median, min, max, var]).
    """
    meta: dict[str, str | int] = {}
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
        else:
            m = PAIR_LINE_RE.match(line)
            if m:
                src = int(m.group(1))
                dst = int(m.group(2))
                vals = [float(m.group(i)) for i in range(3, 8)]
                pairs.append((src, dst, vals))

    return meta, pairs


def infer_n(meta: dict[str, str | int], pairs: list[tuple[int, int, list[float]]]) -> int:
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
        if 0 <= src < n and 0 <= dst < n:
            for k, v in zip(METRIC_KEYS, vals):
                mats[k][src, dst] = v
    return mats


def _title_suffix(meta: dict[str, str | int]) -> str:
    parts: list[str] = []
    if "mode" in meta:
        parts.append(str(meta["mode"]))
    if "cuda_aware" in meta:
        parts.append(f"CUDA-aware: {meta['cuda_aware']}")
    return " · ".join(parts) if parts else ""


def draw_heatmap(
    matrix: np.ndarray,
    metric: str,
    out_path: Path,
    *,
    title_suffix: str,
    cmap: str,
    dpi: int,
) -> None:
    labels = [f"g{i}" for i in range(matrix.shape[0])]
    fig_w = max(5.0, 0.45 * matrix.shape[1] + 2.5)
    fig_h = max(4.5, 0.45 * matrix.shape[0] + 2.0)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    # nan -> белые дырки на карте
    masked = np.ma.masked_invalid(matrix)
    im = ax.imshow(masked, cmap=cmap, aspect="equal", interpolation="nearest")

    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=0, fontsize=9)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("destination rank (GPU index)")
    ax.set_ylabel("source rank (GPU index)")

    t = f"gpu_one_to_one — {metric} (µs)"
    if title_suffix:
        t = f"{t}\n{title_suffix}"
    ax.set_title(t, fontsize=10)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel("µs", rotation=270, labelpad=12)

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
        "--prefix",
        default="",
        help="Output filename prefix (default: stem of input file, or 'gpu_bench' for stdin)",
    )
    p.add_argument(
        "--metrics",
        default=",".join(METRIC_KEYS),
        help=f"Comma-separated subset of {','.join(METRIC_KEYS)} (default: all)",
    )
    p.add_argument(
        "--cmap",
        default="viridis",
        help="Matplotlib colormap name (default: viridis)",
    )
    p.add_argument("--dpi", type=int, default=150, help="PNG resolution (default: 150)")
    args = p.parse_args()

    in_path = args.input
    if in_path == "-":
        text = sys.stdin.read()
        prefix = args.prefix or "gpu_bench"
    else:
        path = Path(in_path)
        text = path.read_text(encoding="utf-8", errors="replace")
        prefix = args.prefix or path.stem

    meta, pairs = parse_gpu_one_to_one_text(text)
    if not pairs:
        print("gpu_render: no matching 'pair g...' lines found.", file=sys.stderr)
        return 1

    n = infer_n(meta, pairs)
    if n <= 0:
        print("gpu_render: could not infer matrix size.", file=sys.stderr)
        return 1

    mats = fill_matrices(n, pairs)
    want = {m.strip() for m in args.metrics.split(",") if m.strip()}
    unknown = want - set(METRIC_KEYS)
    if unknown:
        print(f"gpu_render: unknown metrics: {unknown}", file=sys.stderr)
        return 1

    args.out_dir.mkdir(parents=True, exist_ok=True)
    suffix = _title_suffix(meta)

    for key in METRIC_KEYS:
        if key not in want:
            continue
        out_file = args.out_dir / f"{prefix}_{key}.png"
        draw_heatmap(mats[key], key, out_file, title_suffix=suffix, cmap=args.cmap, dpi=args.dpi)
        print(out_file)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
