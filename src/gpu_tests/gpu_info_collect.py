#!/usr/bin/env python3
"""
Collect GPU topology and host hints (stdlib only; no matplotlib/numpy).
Writes gpu_snapshot.json, system_info.json / system_info.txt, and nvidia_smi.txt (topo -m then -L legend; same strings are in the JSON).

Rendering is separate:

  python3 gpu_info_render.py output/gpu_snapshot.json --out-dir output
"""
from __future__ import annotations

import argparse
import json
import re
import socket
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def run_cmd(cmd: list[str]) -> tuple[bool, str, str]:
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, check=False)
        return p.returncode == 0, p.stdout.strip(), p.stderr.strip()
    except FileNotFoundError:
        return False, "", f"command not found: {' '.join(cmd)}"


def parse_topology_lines_plain(lines: list[str]) -> tuple[list[str], list[list[str]]]:
    """Parse nvidia-smi topo -m; matrix as nested lists (no numpy)."""
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
    m = [["UNK"] * n for _ in range(n)]
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
            m[row][col] = tokens[col + 1]
        row += 1
    return gpus, m


def parse_inter_node_plain(lines: list[str]) -> tuple[list[str], list[list[str]]]:
    """Parse node-level matrix file (input/1_8.in style), plain lists."""
    header_idx = None
    for i, line in enumerate(lines):
        if "node0" in line or "node1" in line or re.search(r"\bnode\d+\b", line):
            header_idx = i
            break
    if header_idx is None:
        raise RuntimeError("Cannot find node header in inter-node text")

    header_tokens = lines[header_idx].split()
    nodes = [t for t in header_tokens if t.startswith("node")]
    n = len(nodes)
    m = [["UNK"] * n for _ in range(n)]
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
            m[r][c] = tokens[c + 1]
    return nodes, m


def summarize_link_counts(labels: list[str], matrix: list[list[str]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    n = len(labels)
    for i in range(n):
        for j in range(i + 1, n):
            v = matrix[i][j]
            if v == "X":
                continue
            counts[v] = counts.get(v, 0) + 1
    return counts


def summarize_internal_switching(
    node_inputs: list[tuple[str, Path]],
) -> dict[str, dict[str, int]]:
    summary: dict[str, dict[str, int]] = {}
    for node_name, topo_path in node_inputs:
        lines = topo_path.read_text(encoding="utf-8", errors="replace").splitlines()
        gpus, m = parse_topology_lines_plain(lines)
        summary[node_name] = summarize_link_counts(gpus, m)
    return summary


def inter_route_labels_from_matrix(nodes: list[str], matrix: list[list[str]]) -> list[str]:
    labels: set[str] = set()
    n = len(nodes)
    for i in range(n):
        for j in range(n):
            if i != j:
                labels.add(matrix[i][j])
    return sorted(labels)


def collect_system_info_payload(
    input_dir: Path,
    node_inputs: list[tuple[str, Path]],
    inter_path: Path,
) -> dict:
    """Inventory and topology hints (same role as former gpu_info.collect_system_info)."""
    data: dict = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "paths": {
            "input_dir": str(input_dir),
            "inter_node_file": str(inter_path),
        },
    }

    ok, out, err = run_cmd(["lscpu"])
    data["lscpu_raw"] = out if ok else f"ERROR: {err}"
    cpu: dict[str, str] = {}
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

    ok, out, err = run_cmd(["nvidia-smi", "-L"])
    data["nvidia_smi_L_raw"] = out if ok else f"ERROR: {err}"
    data["gpus"] = out.splitlines() if ok else []

    ok, out, err = run_cmd(["nvidia-smi", "topo", "-m"])
    data["nvidia_smi_topo_raw"] = out if ok else f"ERROR: {err}"

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

    for cmd_name, cmd in [
        ("ibv_devinfo", ["ibv_devinfo"]),
        ("ibstat", ["ibstat"]),
        ("rdma_link", ["rdma", "link"]),
        ("ip_br_link", ["ip", "-br", "link"]),
    ]:
        o, out, err = run_cmd(cmd)
        data[cmd_name] = out if o else f"ERROR: {err}"

    data["intra_node_link_summary"] = summarize_internal_switching(node_inputs)

    inter_lines = inter_path.read_text(encoding="utf-8", errors="replace").splitlines()
    inter_nodes, inter_m = parse_inter_node_plain(inter_lines)
    data["inter_node_route_labels"] = inter_route_labels_from_matrix(inter_nodes, inter_m)

    data["notes"] = [
        "Intra-node values (NV/PIX/PHB/...) represent local path classes from nvidia-smi topo.",
        "Inter-node labels represent route/fabric class, not direct GPU-to-GPU physical links.",
        "External switch topology (leaf/spine/fat-tree details) is often not fully visible without admin tooling.",
    ]
    return data


def collect_gpu_info_only_payload() -> dict:
    """Minimal fields for --gpu-info-only (compat with former gpu_info binary)."""
    data = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
    }
    ok, out, err = run_cmd(["nvidia-smi", "-L"])
    data["nvidia_smi_L_raw"] = out if ok else f"ERROR: {err}"
    ok, out, err = run_cmd(["nvidia-smi", "topo", "-m"])
    data["nvidia_smi_topo_m_raw"] = out if ok else f"ERROR: {err}"
    return data


def format_nvidia_smi_sidecar_text(info: dict) -> str:
    """Single text file: topo -m matrix first, then nvidia-smi -L as GPU list legend."""
    nvidia_smi_L_raw = info.get("nvidia_smi_L_raw", "")
    nvidia_smi_topo_raw = info.get("nvidia_smi_topo_m_raw") or info.get("nvidia_smi_topo_raw", "")
    return (
        "=== nvidia-smi topo -m ===\n"
        f"{nvidia_smi_topo_raw.rstrip()}\n\n"
        "=== nvidia-smi -L (GPU list, legend) ===\n"
        f"{nvidia_smi_L_raw.rstrip()}\n"
    )


def save_gpu_info_compat(out_dir: Path, info: dict) -> None:
    """Write nvidia_smi.txt (topo -m then -L; fields also in gpu_snapshot.json)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "nvidia_smi.txt").write_text(
        format_nvidia_smi_sidecar_text(info),
        encoding="utf-8",
    )


def save_system_info(out_dir: Path, info: dict) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "system_info.json"
    txt_path = out_dir / "system_info.txt"
    json_path.write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")

    lines: list[str] = []
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
        lines.append(str(info.get(k, "")))

    txt_path.write_text("\n".join(lines), encoding="utf-8")
    return json_path, txt_path


def discover_live_topos(input_dir: Path) -> list[tuple[str, Path]]:
    files = list(input_dir.glob("live_node*.in"))
    if not files:
        return []

    def sort_key(p: Path) -> tuple[int, str]:
        m = re.search(r"live_node(\d+)", p.name)
        return (int(m.group(1)), p.name) if m else (0, p.name)

    files.sort(key=sort_key)
    out: list[tuple[str, Path]] = []
    for p in files:
        m = re.search(r"live_node(\d+)", p.name)
        name = f"node{m.group(1)}" if m else p.stem
        out.append((name, p))
    return out


def collect_snapshot_v1(inter_node_path: Path | None) -> dict:
    ts = datetime.now(timezone.utc).isoformat()
    host = socket.gethostname()

    ok, out_l, err_l = run_cmd(["nvidia-smi", "-L"])
    nvidia_smi_L_raw = out_l if ok else f"ERROR: {err_l}"

    ok, out_topo, err_topo = run_cmd(["nvidia-smi", "topo", "-m"])
    nvidia_smi_topo_m_raw = out_topo if ok else f"ERROR: {err_topo}"

    intra: dict | None = None
    if ok and out_topo:
        try:
            labels, matrix = parse_topology_lines_plain(out_topo.splitlines())
            intra = {"gpu_labels": labels, "matrix": matrix}
        except RuntimeError as e:
            intra = {"error": str(e)}

    extras: dict[str, str] = {}
    for key, cmd in [
        ("lscpu_raw", ["lscpu"]),
        ("free_h_raw", ["free", "-h"]),
        ("lspci_raw", ["lspci"]),
        ("ibv_devinfo_raw", ["ibv_devinfo"]),
        ("ibstat_raw", ["ibstat"]),
        ("rdma_link_raw", ["rdma", "link"]),
        ("ip_br_link_raw", ["ip", "-br", "link"]),
    ]:
        o, out, err = run_cmd(cmd)
        extras[key] = out if o else f"ERROR: {err}"

    snap: dict = {
        "format_version": 1,
        "timestamp_utc": ts,
        "hostname": host,
        "nvidia_smi_L_raw": nvidia_smi_L_raw,
        "nvidia_smi_topo_m_raw": nvidia_smi_topo_m_raw,
        "intra": intra,
        "extras": extras,
    }

    if inter_node_path is not None and inter_node_path.is_file():
        snap["inter_node"] = {
            "filename": inter_node_path.name,
            "raw": inter_node_path.read_text(encoding="utf-8", errors="replace"),
        }

    return snap


def _ensure_minimal_inter_1x1(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text(
            "        node0\n"
            "node0     X\n\n"
            "Legend:\n\n"
            "  X = same node\n",
            encoding="utf-8",
        )
    return path


def attach_system_to_snapshot_v1(snap: dict, inter_arg: Path | None, input_dir: Path) -> None:
    """Fill snap['system'] using local host commands and optional inter matrix."""
    input_dir.mkdir(parents=True, exist_ok=True)
    ok, out_topo, _ = run_cmd(["nvidia-smi", "topo", "-m"])
    if not ok or not out_topo:
        if inter_arg is not None and inter_arg.is_file():
            inter = inter_arg
        else:
            inter = _ensure_minimal_inter_1x1(input_dir / "1_8.in")
        snap["system"] = collect_system_info_payload(input_dir, [], inter)
        return

    live = input_dir / "live_node0.in"
    live.write_text("\n".join(out_topo.splitlines()) + "\n", encoding="utf-8")
    node_inputs: list[tuple[str, Path]] = [("node0", live)]

    if inter_arg is not None and inter_arg.is_file():
        inter = inter_arg
    else:
        inter = _ensure_minimal_inter_1x1(input_dir / "1_8.in")
    snap["system"] = collect_system_info_payload(input_dir, node_inputs, inter)


def collect_multi_snapshot(input_dir: Path, net_map: Path) -> dict:
    """Aggregate per-node live_node*.in + inter-node map (format_version 2)."""
    input_dir = input_dir.resolve()
    net_map = net_map.resolve()
    if not net_map.is_file():
        raise SystemExit(f"Inter-node matrix not found: {net_map}")

    node_inputs = discover_live_topos(input_dir)
    if not node_inputs:
        raise SystemExit(f"No live_node*.in files under {input_dir}")

    ts = datetime.now(timezone.utc).isoformat()
    host = socket.gethostname()

    nodes: list[dict] = []
    for name, topo_path in node_inputs:
        raw = topo_path.read_text(encoding="utf-8", errors="replace")
        intra: dict | None = None
        try:
            labels, matrix = parse_topology_lines_plain(raw.splitlines())
            intra = {"gpu_labels": labels, "matrix": matrix}
        except RuntimeError as e:
            intra = {"error": str(e)}
        nodes.append(
            {
                "name": name,
                "nvidia_smi_topo_m_raw": raw,
                "intra": intra,
            }
        )

    inter_raw = net_map.read_text(encoding="utf-8", errors="replace")

    ok, out_l, err_l = run_cmd(["nvidia-smi", "-L"])
    nvidia_smi_L_raw = out_l if ok else f"ERROR: {err_l}"
    ok, out_topo, err_topo = run_cmd(["nvidia-smi", "topo", "-m"])
    nvidia_smi_topo_m_raw = out_topo if ok else f"ERROR: {err_topo}"

    system = collect_system_info_payload(input_dir, node_inputs, net_map)

    return {
        "format_version": 2,
        "timestamp_utc": ts,
        "hostname": host,
        "nvidia_smi_L_raw": nvidia_smi_L_raw,
        "nvidia_smi_topo_m_raw": nvidia_smi_topo_m_raw,
        "nodes": nodes,
        "inter_node": {"filename": net_map.name, "raw": inter_raw},
        "system": system,
    }


def write_snapshot_bundle(out_dir: Path, snap: dict) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "gpu_snapshot.json"
    json_path.write_text(
        json.dumps(snap, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    sys_info = snap.get("system")
    if isinstance(sys_info, dict) and sys_info:
        save_system_info(out_dir, sys_info)
        merged = dict(sys_info)
        merged["nvidia_smi_L_raw"] = snap.get("nvidia_smi_L_raw", "")
        merged["nvidia_smi_topo_m_raw"] = snap.get("nvidia_smi_topo_m_raw", "")
        save_gpu_info_compat(out_dir, merged)
    else:
        save_gpu_info_compat(out_dir, snap)

    return json_path


def main() -> None:
    p = argparse.ArgumentParser(description="Collect GPU snapshot (stdlib only).")
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("output"),
        help="Directory for gpu_snapshot.json and companion files",
    )
    p.add_argument(
        "--inter-node",
        type=Path,
        default=None,
        help="Optional inter-node matrix file (1_8.in style); embedded in JSON as raw text",
    )
    p.add_argument(
        "--input-dir",
        type=Path,
        default=None,
        help="With --net-map: directory with live_node*.in (multi-node aggregate)",
    )
    p.add_argument(
        "--net-map",
        type=Path,
        default=None,
        help="Inter-node matrix file (required for multi-node aggregate)",
    )
    p.add_argument(
        "--gpu-info-only",
        action="store_true",
        help="Only write nvidia_smi.txt (topo -m then -L)",
    )
    args = p.parse_args()

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.gpu_info_only:
        info = collect_gpu_info_only_payload()
        save_gpu_info_compat(out_dir, info)
        print(f"Saved nvidia-smi dump to: {out_dir.resolve()}")
        print("  - nvidia_smi.txt")
        return

    if args.input_dir is not None and args.net_map is not None:
        snap = collect_multi_snapshot(args.input_dir, args.net_map)
        json_path = write_snapshot_bundle(out_dir, snap)
        print(f"Wrote {json_path} (format_version=2)")
        print("Render PNGs:")
        print(f"  python3 src/gpu_tests/gpu_info_render.py {json_path} --out-dir {out_dir}")
        return

    snap = collect_snapshot_v1(args.inter_node)
    gpu_dir = Path(__file__).resolve().parent
    input_dir = gpu_dir / "input"
    attach_system_to_snapshot_v1(snap, args.inter_node, input_dir)
    json_path = write_snapshot_bundle(out_dir, snap)
    print(f"Wrote {json_path}")
    print("Render PNGs:")
    print(f"  python3 src/gpu_tests/gpu_info_render.py {json_path} --out-dir {out_dir}")


if __name__ == "__main__":
    main()
