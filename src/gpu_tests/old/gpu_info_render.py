#!/usr/bin/env python3
"""
Рендер gpu_snapshot.json в PNG-матрицы.

Пишет:
  - node{n}_matrix_gpu_gpu.png
  - node{n}_matrix_gpu_nic.png (если есть NIC)
  - node{n}_matrix_nic_nic.png (если есть NIC)
  - node{n}_inter_node_matrix.png (если в snapshot есть inter_node.raw)
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

from gpu_info_collect import parse_topology_device_matrices, strip_ansi_escapes


# Коды маршрутов/связей из topo/inter matrices -> цвет ячейки.
PALETTE: dict[str, str] = {
    "X": "#e0f2fe",
    "NV18": "#0c4a6e",
    "NV12": "#0369a1",
    "NV8": "#0284c7",
    "NV7": "#0ea5e9",
    "NV6": "#38bdf8",
    "NV5": "#7dd3fc",
    "NV4": "#93c5fd",
    "NV3": "#bae6fd",
    "NV2": "#dbeafe",
    "NV1": "#f0f9ff",
    "PIX": "#2563eb",
    "PXB": "#1d4ed8",
    "PHB": "#1e40af",
    "NODE": "#4338ca",
    "SYS": "#5b21b6",
    "NET_PATH": "#9333ea",
    "NET_PATH_IB": "#a855f7",
    "NET_PATH_IB_HDR": "#c026d3",
    "NET_PATH_IB_NDR": "#db2777",
    "NET_PATH_RDMA": "#e11d48",
    "NET_PATH_RDMA_ETH": "#f472b6",
    "NET_PATH_ETH": "#ec4899",
    "UNK": "#fbbf24",
}


def _cmap_and_norm() -> tuple[ListedColormap, mcolors.BoundaryNorm]:
    """Дискретная карта цветов: индекс 0 -> 1-й цвет, 1 -> 2-й, ... без интерполяции."""
    colors = tuple(PALETTE.values())
    cmap = ListedColormap(colors)
    bounds = np.arange(-0.5, len(colors) + 0.5, 1.0)
    return cmap, mcolors.BoundaryNorm(bounds, cmap.N)


def _matrix_to_color_index(matrix: np.ndarray) -> np.ndarray:
    """Строковые коды матрицы -> целочисленные индексы палитры."""
    code_to_idx = {code: idx for idx, code in enumerate(PALETTE)}
    idx = np.zeros(matrix.shape, dtype=int)
    fallback = code_to_idx["UNK"]
    for r in range(matrix.shape[0]):
        for c in range(matrix.shape[1]):
            code = str(matrix[r, c]).strip() if matrix[r, c] is not None else "UNK"
            idx[r, c] = code_to_idx.get(code, fallback)
    return idx


def draw_matrix(
    out_path: Path,
    row_labels: list[str],
    col_labels: list[str],
    matrix: np.ndarray,
    title: str,
    *,
    tick_font: int,
    text_font: int,
    xlabel: str | None = None,
    ylabel: str | None = None,
    footer_text: str | None = None,
) -> None:
    """Общий рендер матрицы (квадратной или прямоугольной) в PNG."""
    cmap, norm = _cmap_and_norm()
    idx = _matrix_to_color_index(matrix)
    width = max(6.5, 0.55 * len(col_labels) + 4.0)
    height = max(5.5, 0.42 * len(row_labels) + 3.0)
    if footer_text:
        height += 1.1

    fig, ax = plt.subplots(figsize=(width, height))
    ax.imshow(idx, cmap=cmap, norm=norm, interpolation="nearest", aspect="auto")
    ax.set_xticks(range(len(col_labels)))
    ax.set_yticks(range(len(row_labels)))
    ax.set_xticklabels(col_labels, fontsize=tick_font, rotation=90)
    ax.set_yticklabels(row_labels, fontsize=tick_font)
    ax.set_title(title, fontsize=10)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)

    for r in range(len(row_labels)):
        for c in range(len(col_labels)):
            ax.text(c, r, str(matrix[r, c]), ha="center", va="center", fontsize=text_font, color="black")

    bottom = 0.14 if footer_text else 0.0
    plt.tight_layout(rect=(0.0, bottom, 1.0, 1.0))
    if footer_text:
        fig.text(
            0.5,
            0.02,
            footer_text,
            ha="center",
            va="bottom",
            fontsize=6,
            family="monospace",
        )
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def parse_inter_node_lines(lines: list[str]) -> tuple[list[str], np.ndarray]:
    """Текст inter-node матрицы (nodeX row/col) -> labels + numpy matrix."""
    header_idx = next((i for i, line in enumerate(lines) if re.search(r"\bnode\d+\b", line)), None)
    if header_idx is None:
        raise RuntimeError("Cannot find node header in inter-node text")

    nodes = [t for t in lines[header_idx].split() if t.startswith("node")]
    n = len(nodes)
    matrix = np.full((n, n), "UNK", dtype=object)
    row_map = {name: i for i, name in enumerate(nodes)}
    for line in lines[header_idx + 1 :]:
        if not line.strip() or line.lstrip().startswith("Legend"):
            break
        tokens = line.split()
        if not tokens or not tokens[0].startswith("node"):
            continue
        row_name = tokens[0]
        if row_name not in row_map:
            continue
        r = row_map[row_name]
        for c in range(min(n, len(tokens) - 1)):
            matrix[r, c] = tokens[c + 1]
    return nodes, matrix


def _resolve_node_id(snapshot: dict, out_dir: Path, override: int | None) -> int:
    """node{n} в именах PNG: override > snapshot['node_id'] > существующие node*.png > 4."""
    if override is not None:
        return int(override)

    try:
        stored = int(snapshot.get("node_id", 0))
        if stored > 0:
            return stored
    except (TypeError, ValueError):
        pass

    found: list[int] = []
    for p in out_dir.glob("node*_matrix_*.png"):
        m = re.match(r"node(\d+)_matrix_", p.name)
        if m:
            found.append(int(m.group(1)))
    if found:
        return max(found)

    return 4


def format_mpi_gpudirect_footer(snapshot: dict) -> str | None:
    """
    Краткая подпись для PNG: эвристики GPUDirect / CUDA-aware MPI (не факт RDMA на линке).

    Полный сырой вывод лежит в snapshot['mpi_gpudirect_hints'].
    """
    h = snapshot.get("mpi_gpudirect_hints")
    if not isinstance(h, dict):
        return None
    parts: list[str] = []

    peer = str(h.get("nvidia_peermem_lsmod") or "").strip()
    if peer and not peer.upper().startswith("ERROR"):
        parts.append("GPUDirect kernel: nvidia_peermem/nv_peer_mem loaded")
    elif peer.upper().startswith("ERROR"):
        parts.append("GPUDirect kernel: lsmod check failed")
    else:
        parts.append("GPUDirect kernel: peermem not in lsmod (often need for GPU buffer RDMA)")

    ompi = str(h.get("ompi_info_cuda_lines") or "").strip()
    if ompi and not ompi.upper().startswith("ERROR"):
        first = ompi.split("\n", 1)[0].strip()
        if len(first) > 96:
            first = first[:93] + "…"
        parts.append(f"Open MPI CUDA: {first}")
    else:
        parts.append("Open MPI CUDA: no ompi_info lines (other MPI or not in PATH)")

    ucx = str(h.get("ucx_info_head") or "").strip()
    if ucx:
        u0 = ucx.split("\n", 1)[0].strip()
        if len(u0) > 90:
            u0 = u0[:87] + "…"
        parts.append(f"UCX: {u0}")

    return "\n".join(parts)


def _resolve_intra(snapshot: dict) -> dict:
    """
    Берём intra из snapshot, а если там пусто — парсим сырой nvidia_smi_topo_m_raw.
    Это делает рендер устойчивым к частично заполненным JSON.
    """
    intra = snapshot.get("intra") or {}
    if intra.get("error"):
        raise RuntimeError(f"Snapshot intra parse error: {intra['error']}")
    if intra.get("gpu_labels") and intra.get("matrix_gpu_gpu"):
        return intra

    raw = str(snapshot.get("nvidia_smi_topo_m_raw") or "").strip()
    if raw and not raw.startswith("ERROR:"):
        return parse_topology_device_matrices(strip_ansi_escapes(raw).splitlines())
    raise RuntimeError("Snapshot has no parseable intra topology")


def render_bundle(snapshot: dict, out_dir: Path, node_id_override: int | None = None) -> list[Path]:
    """Рендер всех доступных матриц из snapshot. Возвращает список созданных PNG."""
    out_dir.mkdir(parents=True, exist_ok=True)
    if snapshot.get("format_version") not in (None, 1):
        raise RuntimeError(f"Unsupported format_version: {snapshot.get('format_version')!r}")

    written: list[Path] = []
    node_id = _resolve_node_id(snapshot, out_dir, node_id_override)
    prefix = f"node{node_id}"
    subtitle = str(snapshot.get("hostname") or "")

    intra = _resolve_intra(snapshot)
    gpu_labels = [str(x) for x in (intra.get("gpu_labels") or [])]
    nic_labels = [str(x) for x in (intra.get("nic_labels") or [])]
    gpu_gpu = np.array(intra.get("matrix_gpu_gpu") or [], dtype=object)
    if not gpu_labels or gpu_gpu.size == 0:
        raise RuntimeError("intra missing gpu_labels / matrix_gpu_gpu")

    footer = format_mpi_gpudirect_footer(snapshot)

    p = out_dir / f"{prefix}_matrix_gpu_gpu.png"
    draw_matrix(
        p,
        gpu_labels,
        gpu_labels,
        gpu_gpu,
        f"GPU×GPU — {subtitle}",
        tick_font=8,
        text_font=6,
        footer_text=footer,
    )
    written.append(p)

    gpu_nic_raw = intra.get("matrix_gpu_nic") or []
    if nic_labels and gpu_nic_raw:
        p = out_dir / f"{prefix}_matrix_gpu_nic.png"
        draw_matrix(
            p,
            gpu_labels,
            nic_labels,
            np.array(gpu_nic_raw, dtype=object),
            f"GPU×NIC — {subtitle}",
            tick_font=9,
            text_font=8,
            footer_text=footer,
        )
        written.append(p)

    nic_nic_raw = intra.get("matrix_nic_nic") or []
    if nic_labels and nic_nic_raw:
        p = out_dir / f"{prefix}_matrix_nic_nic.png"
        draw_matrix(
            p,
            nic_labels,
            nic_labels,
            np.array(nic_nic_raw, dtype=object),
            f"NIC×NIC — {subtitle}",
            tick_font=8,
            text_font=6,
            footer_text=footer,
        )
        written.append(p)

    inter = snapshot.get("inter_node")
    if inter and inter.get("raw"):
        nodes, inter_m = parse_inter_node_lines(str(inter["raw"]).splitlines())
        p = out_dir / f"{prefix}_inter_node_matrix.png"
        draw_matrix(
            p,
            nodes,
            nodes,
            inter_m,
            "Inter-node route/fabric matrix (node x node)",
            tick_font=10,
            text_font=8,
            xlabel="Destination node",
            ylabel="Source node",
            footer_text=footer,
        )
        written.append(p)

    return written


def main() -> None:
    parser = argparse.ArgumentParser(description="Render gpu_snapshot.json to PNGs.")
    parser.add_argument("snapshot", type=Path, help="Path to gpu_snapshot.json")
    parser.add_argument("--out-dir", type=Path, default=None, help="Output directory (default: next to snapshot)")
    parser.add_argument("--node-id", type=int, default=None, help="Override node{n} in output filenames")
    args = parser.parse_args()

    snap_path = args.snapshot
    out_dir = args.out_dir if args.out_dir else snap_path.parent
    snapshot = json.loads(snap_path.read_text(encoding="utf-8"))

    for out_png in render_bundle(snapshot, out_dir, node_id_override=args.node_id):
        print(f"Wrote {out_png}")


if __name__ == "__main__":
    main()
