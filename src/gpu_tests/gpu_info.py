import math
import re
import argparse
import json
import socket
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap


PALETTE_ORDER = [
    "X",
    "NV4",
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
    "#eaf6ff",  # X
    "#9ad8ff",  # NV4
    "#76c5ff",  # NV3
    "#53b2ff",  # NV2
    "#2f9fff",  # NV1
    "#4f8ef7",  # PIX
    "#4878cf",  # PXB
    "#4663d8",  # PHB
    "#5e60ce",  # NODE
    "#6a4cbb",  # SYS
    "#7b2cbf",  # NET_PATH
    "#7f3fbf",  # NET_PATH_IB
    "#8b5cf6",  # NET_PATH_IB_HDR
    "#6d28d9",  # NET_PATH_IB_NDR
    "#7c3aed",  # NET_PATH_RDMA
    "#8b5cf6",  # NET_PATH_RDMA_ETH
    "#9d4edd",  # NET_PATH_ETH
    "#9aa0a6",  # UNK
]


def run_cmd(cmd):
    """Run command and return (ok, stdout, stderr)."""
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, check=False)
        return p.returncode == 0, p.stdout.strip(), p.stderr.strip()
    except FileNotFoundError:
        return False, "", f"command not found: {' '.join(cmd)}"


def parse_topology_lines(lines):
    """Parse nvidia-smi topo -m output lines and return (gpus, matrix)."""
    header_idx = None
    for i, line in enumerate(lines):
        if re.search(r"\bGPU0\b", line):
            header_idx = i
            break
    if header_idx is None:
        raise RuntimeError("Cannot find GPU header in nvidia-smi topo output")

    header_tokens = lines[header_idx].split()
    gpus = [tok for tok in header_tokens if tok.startswith("GPU")]
    n = len(gpus)
    m = np.full((n, n), "UNK", dtype=object)
    row = 0
    for line in lines[header_idx + 1 :]:
        if not line.strip():
            break
        if line.lstrip().startswith("Legend"):
            break
        tokens = line.split()
        if not tokens or not tokens[0].startswith("GPU"):
            continue
        if row >= n:
            break
        for col in range(n):
            m[row, col] = tokens[col + 1]
        row += 1
    return gpus, m


def parse_topology_file(path: Path):
    """Parse one intra-node file in nvidia-smi topo -m style."""
    lines = path.read_text(encoding="utf-8").splitlines()
    return parse_topology_lines(lines)


def parse_inter_node_matrix(path: Path):
    """Parse node-level matrix file (input/1_8.in style)."""
    lines = path.read_text(encoding="utf-8").splitlines()
    header_idx = None
    for i, line in enumerate(lines):
        if "node0" in line or "node1" in line:
            header_idx = i
            break
    if header_idx is None:
        raise RuntimeError(f"Cannot find node header in inter-node file: {path}")

    header_tokens = lines[header_idx].split()
    nodes = [t for t in header_tokens if t.startswith("node")]
    n = len(nodes)
    m = np.full((n, n), "UNK", dtype=object)

    row_map = {nname: i for i, nname in enumerate(nodes)}
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


def colorize_matrix(m):
    code_to_idx = {k: i for i, k in enumerate(PALETTE_ORDER)}
    idx = np.zeros(m.shape, dtype=int)
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            idx[i, j] = code_to_idx.get(m[i, j], code_to_idx["UNK"])
    return idx, code_to_idx


def draw_single_matrix(ax, labels, m, title, annotate=True, tick_font=7, text_font=5):
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
                ax.text(j, i, m[i, j], ha="center", va="center", fontsize=text_font, color="black")


def render_intra_node_matrices(node_inputs, output_path: Path):
    parsed = []
    for node_name, topo_path in node_inputs:
        gpus, m = parse_topology_file(Path(topo_path))
        parsed.append((node_name, gpus, m))

    n = len(parsed)
    cols = min(3, n)
    rows = int(math.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4.8 * rows))
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = np.array([axes])
    elif cols == 1:
        axes = np.array([[a] for a in axes])

    for i, (node_name, gpus, m) in enumerate(parsed):
        r = i // cols
        c = i % cols
        ax = axes[r, c]
        draw_single_matrix(ax, gpus, m, f"{node_name} intra-node (GPUxGPU)", annotate=True, tick_font=8, text_font=6)

    for i in range(n, rows * cols):
        r = i // cols
        c = i % cols
        axes[r, c].axis("off")

    fig.suptitle("Intra-node connectivity matrices", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def render_inter_node_matrix(inter_path: Path, output_path: Path):
    nodes, m = parse_inter_node_matrix(inter_path)
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    draw_single_matrix(ax, nodes, m, "Inter-node route/fabric matrix (node x node)", annotate=True, tick_font=10, text_font=8)
    ax.set_xlabel("Destination node")
    ax.set_ylabel("Source node")
    plt.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def summarize_internal_switching(node_inputs):
    """Best-effort summary of intra-node link types from matrices."""
    summary = {}
    for node_name, topo_path in node_inputs:
        gpus, m = parse_topology_file(Path(topo_path))
        counts = {}
        for i in range(len(gpus)):
            for j in range(i + 1, len(gpus)):
                v = m[i, j]
                if v == "X":
                    continue
                counts[v] = counts.get(v, 0) + 1
        summary[node_name] = counts
    return summary


def collect_system_info(input_dir: Path, node_inputs, inter_path: Path):
    """
    Collect inventory and topology hints:
    - CPU / RAM
    - GPUs
    - NIC/RDMA interfaces
    - intra-node switch/path hints from matrices
    - inter-node fabric labels from 1_8.in
    """
    data = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "paths": {
            "input_dir": str(input_dir),
            "inter_node_file": str(inter_path),
        },
    }

    # CPU
    ok, out, err = run_cmd(["lscpu"])
    data["lscpu_raw"] = out if ok else f"ERROR: {err}"
    cpu = {}
    if ok:
        for line in out.splitlines():
            if ":" not in line:
                continue
            k, v = [x.strip() for x in line.split(":", 1)]
            if k in {
                "Architecture",
                "CPU(s)",
                "Model name",
                "Thread(s) per core",
                "Core(s) per socket",
                "Socket(s)",
                "NUMA node(s)",
            }:
                cpu[k] = v
    data["cpu"] = cpu

    # RAM
    ok, out, err = run_cmd(["free", "-h"])
    data["free_h_raw"] = out if ok else f"ERROR: {err}"
    data["memory"] = {}
    if ok:
        for line in out.splitlines():
            if line.lower().startswith("mem:"):
                t = line.split()
                if len(t) >= 7:
                    data["memory"] = {
                        "total": t[1],
                        "used": t[2],
                        "free": t[3],
                        "shared": t[4],
                        "buff_cache": t[5],
                        "available": t[6],
                    }

    # GPU inventory
    ok, out, err = run_cmd(["nvidia-smi", "-L"])
    data["nvidia_smi_L_raw"] = out if ok else f"ERROR: {err}"
    data["gpus"] = out.splitlines() if ok else []

    # Local GPU topo raw
    ok, out, err = run_cmd(["nvidia-smi", "topo", "-m"])
    data["nvidia_smi_topo_raw"] = out if ok else f"ERROR: {err}"

    # PCI/NIC inventory
    ok, out, err = run_cmd(["lspci"])
    data["lspci_raw"] = out if ok else f"ERROR: {err}"
    if ok:
        nic = []
        pcie = []
        for line in out.splitlines():
            low = line.lower()
            if "ethernet" in low or "infiniband" in low or "mellanox" in low:
                nic.append(line)
            if "pci bridge" in low or "pcie" in low:
                pcie.append(line)
        data["nic_devices"] = nic
        data["pcie_switch_hints"] = pcie
    else:
        data["nic_devices"] = []
        data["pcie_switch_hints"] = []

    # RDMA/IB info
    for cmd_name, cmd in [
        ("ibv_devinfo", ["ibv_devinfo"]),
        ("ibstat", ["ibstat"]),
        ("rdma_link", ["rdma", "link"]),
        ("ip_br_link", ["ip", "-br", "link"]),
    ]:
        ok, out, err = run_cmd(cmd)
        data[cmd_name] = out if ok else f"ERROR: {err}"

    # Intra switch/path summary derived from input matrices
    data["intra_node_link_summary"] = summarize_internal_switching(node_inputs)

    # Inter-node labels present in 1_8.in
    inter_nodes, inter_m = parse_inter_node_matrix(inter_path)
    labels = sorted({inter_m[i, j] for i in range(len(inter_nodes)) for j in range(len(inter_nodes)) if i != j})
    data["inter_node_route_labels"] = labels

    # Notes about switch visibility
    data["notes"] = [
        "Intra-node values (NV/PIX/PHB/...) represent local path classes from nvidia-smi topo.",
        "Inter-node labels represent route/fabric class, not direct GPU-to-GPU physical links.",
        "External switch topology (leaf/spine/fat-tree details) is often not fully visible without admin tooling.",
    ]
    return data


def save_gpu_info_compat(out_dir: Path, info: dict) -> None:
    """
    Same outputs as the former gpu_info (C++) binary: raw nvidia-smi dumps,
    gpu_info.json, gpu_info_report.txt.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    nvidia_smi_L_raw = info.get("nvidia_smi_L_raw", "")
    nvidia_smi_topo_raw = info.get("nvidia_smi_topo_raw", "")
    (out_dir / "nvidia_smi_L.txt").write_text(nvidia_smi_L_raw, encoding="utf-8")
    (out_dir / "nvidia_smi_topo_m.txt").write_text(nvidia_smi_topo_raw, encoding="utf-8")

    gpu_lines = [ln for ln in nvidia_smi_L_raw.splitlines() if ln.startswith("GPU ")]
    payload = {
        "timestamp_utc": info.get("timestamp_utc", ""),
        "hostname": info.get("hostname", ""),
        "gpu_count": len(gpu_lines),
        "nvidia_smi_L_raw": nvidia_smi_L_raw,
        "nvidia_smi_topo_m_raw": nvidia_smi_topo_raw,
    }
    (out_dir / "gpu_info.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    report_lines = [
        f"Timestamp UTC: {payload['timestamp_utc']}",
        f"Hostname: {payload['hostname']}",
        "",
        "== GPU list (nvidia-smi -L) ==",
        f"GPU count: {payload['gpu_count']}",
    ]
    for gl in gpu_lines:
        report_lines.append(f"- {gl}")
    if not gpu_lines:
        report_lines.append("(no GPUs detected via nvidia-smi -L)")
    report_lines.extend(
        [
            "",
            "== Notes ==",
            "- For topology class information (NVLink/PCIe/PHB/PIX/...), see nvidia-smi topo -m output.",
        ]
    )
    (out_dir / "gpu_info_report.txt").write_text("\n".join(report_lines) + "\n", encoding="utf-8")


def collect_gpu_info_only() -> dict:
    """Minimal snapshot for --gpu-info-only (replaces gpu_info executable)."""
    data = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
    }
    ok, out, err = run_cmd(["nvidia-smi", "-L"])
    data["nvidia_smi_L_raw"] = out if ok else f"ERROR: {err}"
    ok, out, err = run_cmd(["nvidia-smi", "topo", "-m"])
    data["nvidia_smi_topo_m_raw"] = out if ok else f"ERROR: {err}"
    return data


def save_system_info(out_dir: Path, info: dict):
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "system_info.json"
    txt_path = out_dir / "system_info_report.txt"
    json_path.write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = []
    lines.append(f"Timestamp UTC: {info.get('timestamp_utc', '')}")
    lines.append(f"Hostname: {info.get('hostname', '')}")
    lines.append("")
    lines.append("== CPU ==")
    for k, v in info.get("cpu", {}).items():
        lines.append(f"- {k}: {v}")
    lines.append("")
    lines.append("== Memory ==")
    for k, v in info.get("memory", {}).items():
        lines.append(f"- {k}: {v}")
    lines.append("")
    lines.append("== GPUs (nvidia-smi -L) ==")
    for row in info.get("gpus", []):
        lines.append(f"- {row}")
    lines.append("")
    lines.append("== NIC / Fabric devices (from lspci) ==")
    for row in info.get("nic_devices", []):
        lines.append(f"- {row}")
    lines.append("")
    lines.append("== Intra-node link summary (from matrices) ==")
    for node, counts in info.get("intra_node_link_summary", {}).items():
        lines.append(f"- {node}: {counts}")
    lines.append("")
    lines.append("== Inter-node route labels (from inter-node matrix) ==")
    lines.append(f"- {info.get('inter_node_route_labels', [])}")
    lines.append("")
    lines.append("== Notes ==")
    for n in info.get("notes", []):
        lines.append(f"- {n}")
    lines.append("")
    lines.append("== Raw command snippets ==")
    for k in ["ibv_devinfo", "ibstat", "rdma_link", "ip_br_link"]:
        lines.append(f"\n[{k}]")
        lines.append(info.get(k, ""))

    txt_path.write_text("\n".join(lines), encoding="utf-8")
    return json_path, txt_path


def render_global_matrix(node_inputs, inter_path: Path, output_path: Path):
    per_node = []
    labels = []
    offsets = []
    cur = 0
    for node_name, topo_path in node_inputs:
        gpus, m = parse_topology_file(Path(topo_path))
        per_node.append((node_name, gpus, m))
        offsets.append(cur)
        cur += len(gpus)
        labels.extend([f"{node_name}:{g}" for g in gpus])

    total = sum(len(g) for _, g, _ in per_node)
    global_m = np.full((total, total), "NET_PATH", dtype=object)

    node_ranges = {}
    for (node_name, gpus, m), off in zip(per_node, offsets):
        n = len(gpus)
        node_ranges[node_name] = (off, off + n)
        global_m[off : off + n, off : off + n] = m

    inter_nodes, inter_m = parse_inter_node_matrix(inter_path)
    node_idx = {n: i for i, n in enumerate(inter_nodes)}
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

    used = []
    seen = set()
    for row in global_m:
        for code in row:
            if code not in seen:
                seen.add(code)
                used.append(code)
    handles = []
    legend_labels = []
    for code in used:
        idx_color = code_to_idx.get(code, code_to_idx["UNK"])
        handles.append(
            plt.Line2D(
                [0], [0],
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
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(
        description="GPU inventory, topology matrices, and system info (nvidia-smi, plots, reports)."
    )
    parser.add_argument(
        "--net-map",
        dest="net_map",
        default=None,
        help="Path to inter-node matrix file (e.g. input/1_8.in).",
    )
    parser.add_argument(
        "--input-dir",
        dest="input_dir",
        default=None,
        help="Directory for intra-node topo inputs and live snapshots (default: ./input from CWD).",
    )
    parser.add_argument(
        "--out-dir",
        dest="out_dir",
        default=None,
        help="Directory for generated images and system_info.* files (default: ./output from CWD).",
    )
    parser.add_argument(
        "--node-topo",
        dest="node_topos",
        action="append",
        default=[],
        help="One node topology in format nodeX:path/to/topo.in (can be repeated).",
    )
    parser.add_argument(
        "--gpu-info-only",
        dest="gpu_info_only",
        action="store_true",
        help="Only write gpu_info.* and nvidia_smi_*.txt (like former gpu_info binary), then exit.",
    )
    parser.add_argument(
        "node_inputs",
        nargs="*",
        help="Positional node inputs in format nodeX:path/to/topo.in.",
    )
    return parser.parse_args()


def parse_node_inputs(raw_items):
    node_inputs = []
    for a in raw_items:
        if ":" not in a:
            raise SystemExit(f"Expected node:path format, got: {a}")
        node, p = a.split(":", 1)
        node_inputs.append((node, p))
    return node_inputs


def main():
    base = Path(__file__).resolve().parent
    args = parse_args()

    # Defaults follow the working directory (where mpirun/sbatch runs from).
    input_dir = Path(args.input_dir) if args.input_dir else (Path.cwd() / "input")
    out_dir = Path(args.out_dir) if args.out_dir else (Path.cwd() / "output")

    if args.gpu_info_only:
        info = collect_gpu_info_only()
        save_gpu_info_compat(out_dir, info)
        print(f"Saved GPU info to: {out_dir.resolve()}")
        print("  - gpu_info.json")
        print("  - gpu_info_report.txt")
        print("  - nvidia_smi_L.txt")
        print("  - nvidia_smi_topo_m.txt")
        return

    node_inputs = parse_node_inputs(args.node_inputs + args.node_topos)

    if not node_inputs:
        files = sorted(input_dir.glob("*node*.in"))
        if files:
            for p in files:
                m = re.search(r"(node\d+)", p.name)
                node_name = m.group(1) if m else p.stem
                node_inputs.append((node_name, str(p)))
        else:
            # Live fallback (single-node): use `nvidia-smi topo -m` directly.
            ok, out, err = run_cmd(["nvidia-smi", "topo", "-m"])
            if not ok:
                raise SystemExit(
                    "No input files found and failed to query live topology via nvidia-smi topo -m."
                )
            parse_topology_lines(out.splitlines())
            input_dir.mkdir(parents=True, exist_ok=True)
            live_node = "node0"
            live_file = input_dir / "live_node0.in"
            # Save live snapshot in topo-like text.
            raw_lines = out.splitlines()
            live_file.write_text("\n".join(raw_lines) + "\n", encoding="utf-8")
            node_inputs.append((live_node, str(live_file)))
            # Create minimal inter-node 1x1 matrix if absent.
            inter_auto = input_dir / "1_8.in"
            if not inter_auto.exists():
                inter_auto.write_text(
                    "        node0\n"
                    "node0     X\n\n"
                    "Legend:\n\n"
                    "  X = same node\n",
                    encoding="utf-8",
                )

    if args.net_map:
        inter_path = Path(args.net_map)
    else:
        inter_path = input_dir / "1_8.in"
    if not inter_path.exists():
        raise SystemExit(
            "Inter-node matrix file not found. Pass it explicitly via --net-map <path>."
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    out_intra = out_dir / "gpu_intra_node_matrices.png"
    out_inter = out_dir / "gpu_inter_node_matrix.png"

    render_intra_node_matrices(node_inputs, out_intra)
    render_inter_node_matrix(inter_path, out_inter)

    # Collect and save system inventory/report.
    info = collect_system_info(input_dir, node_inputs, inter_path)
    info_json, info_txt = save_system_info(out_dir, info)
    save_gpu_info_compat(out_dir, info)

    print(f"Saved: {out_intra}")
    print(f"Saved: {out_inter}")
    print(f"Saved: {info_json}")
    print(f"Saved: {info_txt}")
    print(f"Saved: {out_dir / 'gpu_info.json'} (and gpu_info_report.txt, nvidia_smi_*.txt)")


if __name__ == "__main__":
    main()
