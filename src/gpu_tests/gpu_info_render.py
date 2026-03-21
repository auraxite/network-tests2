#!/usr/bin/env python3
"""
Read gpu_snapshot.json from gpu_info_collect.py and write PNG matrices (matplotlib, numpy).
May run on a login node with deps installed, or copy JSON to a machine that has matplotlib.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

from gpu_info_collect import parse_topology_device_matrices, strip_ansi_escapes

PALETTE_ORDER = [
    "X",
    "NV4",
    "NV12",
    "NV3",
    "NV2",
    "NV1",
    "PIX",
    "PXB",
    "PHB",
    "NODE",
    "SYS",
    "NET_PATH",
    "NET_PATH_IB",
    "NET_PATH_IB_HDR",
    "NET_PATH_IB_NDR",
    "NET_PATH_RDMA",
    "NET_PATH_RDMA_ETH",
    "NET_PATH_ETH",
    "UNK",
]

PALETTE_COLORS = [
    "#eaf6ff",
    "#9ad8ff",
    "#87ceff",
    "#76c5ff",
    "#53b2ff",
    "#2f9fff",
    "#4f8ef7",
    "#4878cf",
    "#4663d8",
    "#5e60ce",
    "#6a4cbb",
    "#7b2cbf",
    "#7f3fbf",
    "#8b5cf6",
    "#6d28d9",
    "#7c3aed",
    "#8b5cf6",
    "#9d4edd",
    "#9aa0a6",
]


def parse_inter_node_lines(lines: list[str]) -> tuple[list[str], np.ndarray]:
    header_idx = None
    for i, line in enumerate(lines):
        if re.search(r"\bnode\d+\b", line):
            header_idx = i
            break
    if header_idx is None:
        raise RuntimeError("Cannot find node header in inter-node text")

    header_tokens = lines[header_idx].split()
    nodes = [t for t in header_tokens if t.startswith("node")]
    n = len(nodes)
    m = np.full((n, n), "UNK", dtype=object)
    row_map = {name: i for i, name in enumerate(nodes)}
    for line in lines[header_idx + 1 :]:
        if not line.strip():
            break
        if line.lstrip().startswith("Legend"):
            break
        tokens = line.split()
        if not tokens or not tokens[0].startswith("node"):
            continue
        row_node = tokens[0]
        if row_node not in row_map:
            continue
        r = row_map[row_node]
        for c in range(min(n, len(tokens) - 1)):
            m[r, c] = tokens[c + 1]
    return nodes, m


def colorize_matrix(m: np.ndarray) -> tuple[np.ndarray, dict]:
    code_to_idx = {k: i for i, k in enumerate(PALETTE_ORDER)}
    idx = np.zeros(m.shape, dtype=int)
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            idx[i, j] = code_to_idx.get(m[i, j], code_to_idx["UNK"])
    return idx, code_to_idx


def draw_single_matrix(ax, labels: list[str], m: np.ndarray, title: str, **kw) -> None:
    tick_font = kw.get("tick_font", 7)
    text_font = kw.get("text_font", 5)
    annotate = kw.get("annotate", True)
    idx, _ = colorize_matrix(m)
    cmap = ListedColormap(PALETTE_COLORS)
    ax.imshow(idx, cmap=cmap, interpolation="nearest")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=tick_font, rotation=90)
    ax.set_yticklabels(labels, fontsize=tick_font)
    ax.set_title(title, fontsize=10)
    if annotate:
        for i in range(len(labels)):
            for j in range(len(labels)):
                ax.text(
                    j,
                    i,
                    str(m[i, j]),
                    ha="center",
                    va="center",
                    fontsize=text_font,
                    color="black",
                )


def draw_rect_matrix(
    ax,
    row_labels: list[str],
    col_labels: list[str],
    m: np.ndarray,
    title: str,
    **kw,
) -> None:
    tick_font = kw.get("tick_font", 6)
    text_font = kw.get("text_font", 4)
    annotate = kw.get("annotate", True)
    idx, _ = colorize_matrix(m)
    cmap = ListedColormap(PALETTE_COLORS)
    ax.imshow(idx, cmap=cmap, interpolation="nearest", aspect="auto")
    ax.set_xticks(range(len(col_labels)))
    ax.set_yticks(range(len(row_labels)))
    ax.set_xticklabels(col_labels, fontsize=tick_font, rotation=90)
    ax.set_yticklabels(row_labels, fontsize=tick_font)
    ax.set_title(title, fontsize=10)
    if annotate:
        for i in range(len(row_labels)):
            for j in range(len(col_labels)):
                ax.text(
                    j,
                    i,
                    str(m[i, j]),
                    ha="center",
                    va="center",
                    fontsize=text_font,
                    color="black",
                )


def _intra_matrix_gpu_gpu(intra: dict) -> object:
    """GPU×GPU block; older snapshots used key ``matrix``."""
    if "matrix_gpu_gpu" in intra:
        return intra.get("matrix_gpu_gpu")
    return intra.get("matrix")


def _resolve_intra_from_snap_v1(snap: dict) -> dict:
    raw_topo = (snap.get("nvidia_smi_topo_m_raw") or "").strip()
    if raw_topo:
        try:
            return parse_topology_device_matrices(strip_ansi_escapes(raw_topo).splitlines())
        except RuntimeError:
            pass
    intra = snap.get("intra") or {}
    if intra.get("error"):
        raise RuntimeError(f"Snapshot intra parse error: {intra['error']}")
    if intra.get("gpu_labels") and _intra_matrix_gpu_gpu(intra):
        return intra
    raise RuntimeError("Snapshot has no parseable intra topology")


def _resolve_intra_from_node_entry(ent: dict) -> dict:
    raw = (ent.get("nvidia_smi_topo_m_raw") or "").strip()
    if raw:
        try:
            return parse_topology_device_matrices(strip_ansi_escapes(raw).splitlines())
        except RuntimeError:
            pass
    intra = ent.get("intra") or {}
    if intra.get("error"):
        raise RuntimeError(f"Intra parse error: {intra['error']}")
    if intra.get("gpu_labels") and _intra_matrix_gpu_gpu(intra):
        return intra
    raise RuntimeError("Node entry has no parseable intra topology")


def _node_id_from_name(name: str) -> int:
    m = re.fullmatch(r"node(\d+)", name.strip())
    return int(m.group(1)) if m else 0


def render_node_topo_matrices(
    intra: dict,
    out_dir: Path,
    file_prefix: str,
    subtitle: str,
) -> list[Path]:
    """Write GPU×GPU, GPU×NIC, NIC×NIC PNGs with prefix node{n}_matrix_*.png."""
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    gpus = intra.get("gpu_labels") or []
    nics = intra.get("nic_labels") or []
    gg = _intra_matrix_gpu_gpu(intra)
    if not gpus or not gg:
        raise RuntimeError("intra missing gpu_labels / matrix_gpu_gpu")

    mgg = np.array(gg, dtype=object)
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    draw_single_matrix(
        ax,
        [str(x) for x in gpus],
        mgg,
        f"GPU×GPU — {subtitle}",
        tick_font=8,
        text_font=6,
    )
    plt.tight_layout()
    p = out_dir / f"{file_prefix}_matrix_gpu_gpu.png"
    fig.savefig(p, dpi=180)
    plt.close(fig)
    written.append(p)

    gpn = intra.get("matrix_gpu_nic") or []
    if nics and gpn:
        mgn = np.array(gpn, dtype=object)
        w = max(9.0, 0.55 * len(nics) + 4.0)
        fig, ax = plt.subplots(figsize=(w, 6.0))
        draw_rect_matrix(
            ax,
            [str(x) for x in gpus],
            [str(x) for x in nics],
            mgn,
            f"GPU×NIC — {subtitle}",
            tick_font=7,
            text_font=5,
        )
        plt.tight_layout()
        p = out_dir / f"{file_prefix}_matrix_gpu_nic.png"
        fig.savefig(p, dpi=180)
        plt.close(fig)
        written.append(p)

    ncn = intra.get("matrix_nic_nic") or []
    if nics and ncn:
        mnn = np.array(ncn, dtype=object)
        fig, ax = plt.subplots(figsize=(6.5, 5.5))
        draw_single_matrix(
            ax,
            [str(x) for x in nics],
            mnn,
            f"NIC×NIC — {subtitle}",
            tick_font=8,
            text_font=6,
        )
        plt.tight_layout()
        p = out_dir / f"{file_prefix}_matrix_nic_nic.png"
        fig.savefig(p, dpi=180)
        plt.close(fig)
        written.append(p)

    return written


def render_intra_from_snapshot_v1(snap: dict, out_dir: Path) -> list[Path]:
    intra = _resolve_intra_from_snap_v1(snap)
    node_id = int(snap.get("node_id", 0))
    pfx = f"node{node_id}"
    hn = snap.get("hostname", "")
    return render_node_topo_matrices(intra, out_dir, pfx, hn)


def render_intra_multi_from_snapshot_v2(snap: dict, out_dir: Path) -> list[Path]:
    nodes = snap.get("nodes") or []
    host = snap.get("hostname", "")
    all_written: list[Path] = []
    for ent in nodes:
        name = ent.get("name", "node0")
        intra = _resolve_intra_from_node_entry(ent)
        pfx = f"node{_node_id_from_name(name)}"
        sub = f"{name} @ {host}" if host else name
        all_written.extend(render_node_topo_matrices(intra, out_dir, pfx, sub))
    return all_written


def render_inter_from_raw(raw: str, out_path: Path) -> None:
    nodes, m = parse_inter_node_lines(raw.splitlines())
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    draw_single_matrix(
        ax,
        nodes,
        m,
        "Inter-node route/fabric matrix (node x node)",
        tick_font=10,
        text_font=8,
    )
    ax.set_xlabel("Destination node")
    ax.set_ylabel("Source node")
    plt.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def render_global_matrix_from_snapshot_v2(snap: dict, out_path: Path) -> None:
    """Combined intra blocks + inter-node labels (same idea as former gpu_info.py)."""
    nodes = snap.get("nodes") or []
    inter = snap.get("inter_node") or {}
    raw_inter = inter.get("raw") or ""
    if not raw_inter:
        raise RuntimeError("format_version=2 snapshot needs inter_node.raw for global matrix")

    inter_nodes, inter_m = parse_inter_node_lines(raw_inter.splitlines())
    node_idx = {n: i for i, n in enumerate(inter_nodes)}

    per_node: list[tuple[str, list[str], np.ndarray]] = []
    labels: list[str] = []
    offsets: list[int] = []
    cur = 0
    for ent in nodes:
        name = ent.get("name", "?")
        intra = ent.get("intra") or {}
        if "error" in intra:
            raise RuntimeError(f"Intra parse error for {name}: {intra['error']}")
        gpus = intra.get("gpu_labels")
        matrix = _intra_matrix_gpu_gpu(intra)
        if not gpus or matrix is None:
            raise RuntimeError(f"Node {name}: missing intra data")
        m = np.array(matrix, dtype=object)
        per_node.append((name, gpus, m))
        offsets.append(cur)
        cur += len(gpus)
        labels.extend([f"{name}:{g}" for g in gpus])

    total = sum(len(g) for _, g, _ in per_node)
    global_m = np.full((total, total), "NET_PATH", dtype=object)

    node_ranges: dict[str, tuple[int, int]] = {}
    for (node_name, gpus, m), off in zip(per_node, offsets):
        nloc = len(gpus)
        node_ranges[node_name] = (off, off + nloc)
        global_m[off : off + nloc, off : off + nloc] = m

    for a in node_ranges:
        for b in node_ranges:
            if a == b:
                continue
            if a in node_idx and b in node_idx:
                lbl = inter_m[node_idx[a], node_idx[b]]
            else:
                lbl = "NET_PATH"
            a0, a1 = node_ranges[a]
            b0, b1 = node_ranges[b]
            global_m[a0:a1, b0:b1] = lbl

    fig, ax = plt.subplots(figsize=(14, 12))
    idx, code_to_idx = colorize_matrix(global_m)
    cmap = ListedColormap(PALETTE_COLORS)
    ax.imshow(idx, cmap=cmap, interpolation="nearest")
    ax.set_xticks(range(total))
    ax.set_yticks(range(total))
    ax.set_xticklabels(labels, fontsize=6, rotation=90)
    ax.set_yticklabels(labels, fontsize=6)
    ax.set_xlabel("Destination GPU")
    ax.set_ylabel("Source GPU")
    ax.set_title("Global connectivity matrix (intra + inter route class)")

    for off in offsets[1:]:
        ax.axhline(off - 0.5, color="white", linewidth=2.2)
        ax.axvline(off - 0.5, color="white", linewidth=2.2)

    spans = []
    for i, off in enumerate(offsets):
        end = offsets[i + 1] if i + 1 < len(offsets) else total
        spans.append((off, end))
    for (r0, r1) in spans:
        for i in range(r0, r1):
            for j in range(r0, r1):
                ax.text(j, i, global_m[i, j], ha="center", va="center", fontsize=4.8, color="black")

    node_names = [n for n, _, _ in per_node]
    for i, (r0, r1) in enumerate(spans):
        for j, (c0, c1) in enumerate(spans):
            if i == j:
                continue
            a = node_names[i]
            b = node_names[j]
            if a in node_idx and b in node_idx:
                lbl = inter_m[node_idx[a], node_idx[b]]
            else:
                lbl = "NET_PATH"
            cx = (c0 + c1 - 1) / 2.0
            cy = (r0 + r1 - 1) / 2.0
            ax.text(cx, cy, lbl, ha="center", va="center", fontsize=7.5, color="white", fontweight="bold")

    used: list[str] = []
    seen: set[str] = set()
    for row in global_m:
        for code in row:
            if code not in seen:
                seen.add(code)
                used.append(str(code))
    handles = []
    legend_labels = []
    for code in used:
        idx_color = code_to_idx.get(code, code_to_idx["UNK"])
        handles.append(
            plt.Line2D(
                [0],
                [0],
                marker="s",
                linestyle="",
                markersize=8,
                markerfacecolor=PALETTE_COLORS[idx_color],
                markeredgecolor="none",
            )
        )
        legend_labels.append(code)
    ax.legend(handles, legend_labels, loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0, fontsize=8)
    ax.text(
        1.02,
        -0.06,
        "Inter-node blocks show route/fabric class,\nnot direct GPU-to-GPU physical links.",
        transform=ax.transAxes,
        fontsize=8,
        va="top",
    )

    plt.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def render_bundle(snap: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    ver = snap.get("format_version")

    if ver == 1:
        for p in render_intra_from_snapshot_v1(snap, out_dir):
            print(f"Wrote {p}")
        inter = snap.get("inter_node")
        if inter and inter.get("raw"):
            nid = int(snap.get("node_id", 0))
            inter_png = out_dir / f"node{nid}_inter_node_matrix.png"
            render_inter_from_raw(inter["raw"], inter_png)
            print(f"Wrote {inter_png}")
        return

    if ver == 2:
        for p in render_intra_multi_from_snapshot_v2(snap, out_dir):
            print(f"Wrote {p}")
        inter = snap.get("inter_node")
        if inter and inter.get("raw"):
            inter_png = out_dir / "inter_node_matrix.png"
            render_inter_from_raw(inter["raw"], inter_png)
            print(f"Wrote {inter_png}")
            global_png = out_dir / "global_connectivity_matrix.png"
            render_global_matrix_from_snapshot_v2(snap, global_png)
            print(f"Wrote {global_png}")
        return

    raise SystemExit(f"Unsupported format_version in snapshot: {ver!r}")


def main() -> None:
    p = argparse.ArgumentParser(description="Render gpu_snapshot.json to PNGs.")
    p.add_argument("snapshot", type=Path, help="Path to gpu_snapshot.json")
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory (default: next to snapshot)",
    )
    args = p.parse_args()

    snap_path: Path = args.snapshot
    out_dir = args.out_dir if args.out_dir else snap_path.parent

    snap = json.loads(snap_path.read_text(encoding="utf-8"))
    render_bundle(snap, out_dir)


if __name__ == "__main__":
    main()
