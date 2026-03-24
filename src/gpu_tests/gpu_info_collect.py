#!/usr/bin/env python3
"""
Сбор минимального снимка узла в ОДИН файл: gpu_snapshot.json.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import socket
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def run_cmd(cmd: list[str]) -> tuple[bool, str, str]:
    """Запуск внешней команды: (ok, stdout, stderr)."""
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, check=False)
        return p.returncode == 0, p.stdout.strip(), p.stderr.strip()
    except FileNotFoundError:
        return False, "", f"command not found: {' '.join(cmd)}"


def strip_ansi_escapes(s: str) -> str:
    """Удалить цветовые ANSI-последовательности из текста команд."""
    return re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", s)


def collect_mpi_gpudirect_hints() -> dict[str, str]:
    """
    Эвристики для CUDA-aware MPI и GPUDirect (не доказывают RDMA на сети).

    В приложении нет отдельного «if rdma» — RDMA выбирает стек (MPI/UCX/драйвер),
    если вы передаёте указатели на GPU и окружение собрано (peermem, P2P и т.д.).
    Здесь только то, что можно снять без запуска MPI-программы.
    """
    hints: dict[str, str] = {}
    ok, out, err = run_cmd(
        ["sh", "-c", "lsmod 2>/dev/null | grep -E '^nvidia_peermem|^nv_peer_mem' || true"]
    )
    hints["nvidia_peermem_lsmod"] = out if ok else f"ERROR: {err}"

    ok2, out2, err2 = run_cmd(
        ["sh", "-c", "ompi_info --parsable 2>/dev/null | grep -i cuda | head -30 || true"]
    )
    hints["ompi_info_cuda_lines"] = out2 if ok2 else f"ERROR: {err2}"

    ok3, out3, _ = run_cmd(["sh", "-c", "mpirun --version 2>/dev/null | head -4 || true"])
    hints["mpirun_version_head"] = out3 if ok3 else ""

    ok4, out4, _ = run_cmd(["sh", "-c", "ucx_info -v 2>/dev/null | head -10 || true"])
    hints["ucx_info_head"] = out4 if ok4 else ""

    return hints


# cn4, compute12, worker3 — типичные имена compute-узлов
_HOST_COMPUTE_NUM_RE = re.compile(r"\b(?:cn|compute|worker)(\d+)\b", re.IGNORECASE)


def host_compute_index(hostname: str) -> int | None:
    """Номер узла из hostname (cnN / computeN / workerN) или None."""
    m = _HOST_COMPUTE_NUM_RE.search(hostname)
    return int(m[1]) if m else None


def resolve_output_node_id(cli: int | None, hostname: str) -> int:
    """
    node_id для префикса node{n}_*:
      1) --node-id
      2) GPU_SNAPSHOT_NODE_ID
      3) SLURMD_NODENAME / hostname (cnN, computeN, workerN)
      4) SLURM_NODEID
      5) 0
    """
    if cli is not None:
        return int(cli)
    env_snap = os.environ.get("GPU_SNAPSHOT_NODE_ID", "").strip()
    if env_snap.isdigit():
        return int(env_snap)

    for h in (os.environ.get("SLURMD_NODENAME", "").strip(), hostname):
        if not h:
            continue
        idx = host_compute_index(h)
        if idx is not None:
            return idx

    envn = os.environ.get("SLURM_NODEID", "").strip()
    if envn.isdigit():
        return int(envn)
    return 0


def parse_topology_device_matrices(lines: list[str]) -> dict[str, object]:
    """
    Разбор nvidia-smi topo -m.
    Возвращает полный набор блоков:
      - gpu_labels
      - nic_labels
      - matrix_gpu_gpu
      - matrix_gpu_nic
      - matrix_nic_nic
    """
    header_idx = next((i for i, line in enumerate(lines) if re.search(r"\bGPU0\b", line)), None)
    if header_idx is None:
        raise RuntimeError("Cannot find GPU header in nvidia-smi topo output")

    header_tokens = lines[header_idx].split()
    gpus = [tok for tok in header_tokens if re.fullmatch(r"GPU\d+", tok)]
    nics = [tok for tok in header_tokens if re.fullmatch(r"NIC\d+", tok)]
    devices = gpus + nics
    if not gpus:
        raise RuntimeError("No GPU labels in topo header")

    n_dev = len(devices)
    row_map: dict[str, list[str]] = {}
    for line in lines[header_idx + 1 :]:
        if not line.strip() or line.lstrip().startswith("Legend"):
            break
        tokens = line.split()
        if not tokens or tokens[0] not in devices:
            continue
        name = tokens[0]
        if name in row_map:
            continue
        vals = tokens[1 : 1 + n_dev]
        while len(vals) < n_dev:
            vals.append("UNK")
        row_map[name] = vals[:n_dev]

    missing = [d for d in devices if d not in row_map]
    if missing:
        raise RuntimeError(f"Missing topo rows for devices: {missing[:8]}")

    mat = [row_map[d] for d in devices]
    ng = len(gpus)
    nn = len(nics)
    gpu_gpu = [row[:ng] for row in mat[:ng]]
    gpu_nic: list[list[str]] = []
    nic_nic: list[list[str]] = []
    if nn:
        gpu_nic = [row[ng : ng + nn] for row in mat[:ng]]
        nic_nic = [row[ng : ng + nn] for row in mat[ng : ng + nn]]

    return {
        "gpu_labels": gpus,
        "nic_labels": nics,
        "matrix_gpu_gpu": gpu_gpu,
        "matrix_gpu_nic": gpu_nic,
        "matrix_nic_nic": nic_nic,
    }


def collect_snapshot_v1(inter_node_path: Path | None, node_id: int = 0) -> dict:
    """Собрать единый снимок v1."""
    ts = datetime.now(timezone.utc).isoformat()
    host = socket.gethostname()

    ok_l, out_l, err_l = run_cmd(["nvidia-smi", "-L"])
    if ok_l:
        out_l = strip_ansi_escapes(out_l)
    nvidia_smi_l_raw = out_l if ok_l else f"ERROR: {err_l}"

    ok_t, out_topo, err_topo = run_cmd(["nvidia-smi", "topo", "-m"])
    if ok_t:
        out_topo = strip_ansi_escapes(out_topo)
    nvidia_smi_topo_m_raw = out_topo if ok_t else f"ERROR: {err_topo}"

    # intra используется рендером PNG; при ошибке кладём текст ошибки вместо матриц.
    intra: dict | None = None
    if ok_t and out_topo:
        try:
            intra = parse_topology_device_matrices(out_topo.splitlines())
        except RuntimeError as e:
            intra = {"error": str(e)}

    # extras — только сырые diagnostics (один источник, без дублей в других файлах).
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
        ok_cmd, out, err = run_cmd(cmd)
        extras[key] = out if ok_cmd else f"ERROR: {err}"

    snap: dict = {
        "format_version": 1,
        "node_id": node_id,
        "timestamp_utc": ts,
        "hostname": host,
        "nvidia_smi_L_raw": nvidia_smi_l_raw,
        "nvidia_smi_topo_m_raw": nvidia_smi_topo_m_raw,
        "intra": intra,
        "mpi_gpudirect_hints": collect_mpi_gpudirect_hints(),
        "extras": extras,
    }

    if inter_node_path is not None and inter_node_path.is_file():
        snap["inter_node"] = {
            "filename": inter_node_path.name,
            "raw": inter_node_path.read_text(encoding="utf-8", errors="replace"),
        }

    return snap


def write_snapshot(out_dir: Path, snap: dict) -> Path:
    """Записать снимок в node{n}_snapshot.json."""
    out_dir.mkdir(parents=True, exist_ok=True)
    node_id = int(snap.get("node_id", 0))
    json_path = out_dir / f"node{node_id}_snapshot.json"
    json_path.write_text(json.dumps(snap, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return json_path


def main() -> None:
    p = argparse.ArgumentParser(description="Collect node{n}_snapshot.json (single output file).")
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("output"),
        help="Directory for node{n}_snapshot.json",
    )
    p.add_argument(
        "--inter-node",
        type=Path,
        default=None,
        help="Optional inter-node matrix file (net_map.in style); embedded in JSON as raw text",
    )
    p.add_argument(
        "--node-id",
        type=int,
        default=None,
        help="Force index n for node{n}_* (default: cn<N>/compute<N>/worker<N> or SLURM_NODEID or 0)",
    )
    args = p.parse_args()

    host = socket.gethostname()
    nid = resolve_output_node_id(args.node_id, host)
    snap = collect_snapshot_v1(args.inter_node, node_id=nid)
    json_path = write_snapshot(args.out_dir, snap)
    print(f"Wrote {json_path}")
    print("Render PNGs:")
    print(f"  python3 src/gpu_tests/gpu_info_render.py {json_path} --out-dir {args.out_dir}")


if __name__ == "__main__":
    main()
