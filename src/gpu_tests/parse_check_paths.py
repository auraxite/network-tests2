#!/usr/bin/env python3
"""Разбор результатов check_paths.sh: какой путь прошла каждая проверка.

    python parse_check_paths.py results/paths_<задание> [-o paths_check.csv]

Для каждого теста из perf.csv: код выхода, задержка ucx_perftest (медиана, мкс; со знаком
минус — тест не закончился, взят последний промежуточный отчёт),
конфигурация UCX из журнала (cfg#N tag(...)) и приращения счётчиков на обоих узлах:
байты NVLink (Tx всех каналов, по GPU), байты RDMA адаптеров (port_xmit_data × 4,
для Ethernet ещё tx_vport_rdma_unicast_bytes из ethtool). Объём одного теста —
ITERS × 2 × SIZE (ping-pong): тест передал 4 МБ × 100 в каждую сторону.
"""
import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path

PREFIX = re.compile(r"^\[[^\]]*\] \[[^\]]*\] *")
CFG = re.compile(r"(?:intra|inter)-node cfg#\d+ tag\(([^)]*)\)")
FINAL = re.compile(r"^Final:\s+(\d+)\s+([\d.]+)\s+([\d.]+)")
STAGE = re.compile(r"^\[thread \d+\]\s+(\d+)\s+([\d.]+)\s+([\d.]+)")   # промежуточный отчёт
ERRS = ("Transport retry count exceeded", "Destination is unreachable", "Endpoint timeout")


def counters(path):
    """{ключ: значение} из снимка счётчиков: nic/<dev>, rdma/<dev>, nvl/<gpu>."""
    out = defaultdict(int)
    if not path.exists():
        return None
    gpu, ndev = None, {}
    for line in path.read_text(errors="replace").splitlines():
        f = line.split()
        if line.startswith("nic ") and f[-1].isdigit():
            out[f"nic/{f[1]}"] += int(f[-1]) * 4
        elif line.startswith("netdev "):
            ndev[f[2]] = f[1]
        elif line.startswith("ethtool ") and "tx_vport_rdma_unicast_bytes" in line:
            out[f"rdma/{ndev.get(f[1], f[1])}"] += int(f[-1])
        elif line.startswith("GPU "):
            gpu = line.split(":")[0].split()[1]
        elif "Data Tx:" in line and gpu is not None:
            out[f"nvl/GPU{gpu}"] += int(f[-2]) * 1024
    return out


def delta(cdir, node, name):
    a = counters(cdir / f"{node}_{name}_before.txt")
    b = counters(cdir / f"{node}_{name}_after.txt")
    if a is None or b is None:
        return {}
    return {k: b[k] - a.get(k, 0) for k in b if b[k] - a.get(k, 0) > 0}


def ucx_log(path):
    cfg, err = "", ""
    lat = stage = None
    if not path.exists():
        return cfg, lat, err
    for raw in path.read_text(errors="replace").splitlines():
        line = PREFIX.sub("", raw)
        m = CFG.search(line)
        if m and not cfg:
            cfg = m.group(1)
        m = FINAL.match(line)
        if m:
            lat = float(m.group(2))
        m = STAGE.match(line)
        if m and lat is None:
            stage = float(m.group(2))
        for e in ERRS:
            if e in line and not err:
                err = e
    if lat is None and stage is not None:
        lat = -stage   # тест не закончился: медиана из последнего промежуточного отчёта, со знаком минус
    return cfg, lat, err


def fmt(d, total):
    """Приращения крупнее 5% объёма теста, в долях объёма."""
    return " ".join(f"{k}={v / total:.2f}" for k, v in sorted(d.items()) if v > 0.05 * total)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("-o", "--out")
    ap.add_argument("--size", type=int, default=4194304)
    ap.add_argument("--iters", type=int, default=100)
    a = ap.parse_args()
    d = Path(a.dir)
    total = a.size * a.iters   # в каждую сторону
    rows = []
    with open(d / "perf.csv", newline="") as f:
        for r in csv.DictReader(f):
            name = r["name"]
            cfg_c, lat, err_c = ucx_log(d / "perf" / f"{name}.client")
            cfg_s, _, err_s = ucx_log(d / "perf" / f"{name}.server")
            nodes = sorted({r["server"], r["client"]})
            cnt = {n: fmt(delta(d / "counters", n, name), total) for n in nodes}
            rows.append({
                "name": name, "exit": r["exit"], "numa": r.get("numa", ""),
                "lat_us": "" if lat is None else f"{lat:.1f}",
                "cfg": cfg_c or cfg_s, "error": err_c or err_s,
                **{f"cnt_{n}": cnt[n] for n in nodes},
            })
    keys = list(dict.fromkeys(k for r in rows for k in r))
    if a.out:
        with open(a.out, "w", newline="") as f:
            w = csv.DictWriter(f, keys)
            w.writeheader()
            w.writerows(rows)
    for r in rows:
        print(" | ".join(str(r.get(k, "")) for k in keys))


if __name__ == "__main__":
    main()
