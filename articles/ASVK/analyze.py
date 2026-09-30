# -*- coding: utf-8 -*-
"""Сводка медианных задержек для статьи в кафедральный сборник.

Для каждого порога UCX_RNDV_THRESH (default, 0, 32M), среды (auto, host),
режима (one_to_one, all_to_all) и типа пары (NVLink внутри G8600 V7,
PCIe внутри G5500 V7, PCIe + InfiniBand между узлами G5500 V7) печатает
медиану по парам от медианных задержек пар, минимум и максимум по парам.
"""
import re
import statistics
from pathlib import Path

ARTICLES = Path(__file__).resolve().parents[1]
LAST = ARTICLES / "LAST_RESULTS"
SOURCES = {
    "default": (LAST / "results_default_PCI5", LAST / "results_default_SXM4"),
    "0": (LAST / "results_0_PCI5", LAST / "results_0_SXM4"),
    "32M": (LAST / "results_32M_PCI5", LAST / "results_32M_SXM4"),
}
PAIR_RE = re.compile(r"^pair (\d+)\.(\d+) -> (\d+)\.(\d+) .*?med_us=([\d.]+)")


def collect(folder, env, mode):
    """{размер: {'intra': [...], 'inter': [...]}}"""
    out = {}
    for f in folder.glob(f"*_{env}_{mode}_b*.txt"):
        size = int(re.search(r"_b(\d+)_", f.name).group(1))
        b = out.setdefault(size, {"intra": [], "inter": []})
        for line in f.read_text(encoding="utf-8", errors="replace").splitlines():
            m = PAIR_RE.match(line)
            if not m:
                continue
            sn, sg, dn, dg, med = m.groups()
            if (sn, sg) == (dn, dg):
                continue
            b["intra" if sn == dn else "inter"].append(float(med))
    return out


def main():
    print("thresh env mode link size median min max n GBps")
    for thresh, (pci, sxm) in SOURCES.items():
        for env in ("auto", "host"):
            for mode in ("one_to_one", "all_to_all"):
                sets = [("nvlink", collect(sxm, env, mode), "intra"),
                        ("pcie", collect(pci, env, mode), "intra"),
                        ("ib", collect(pci, env, mode), "inter")]
                for link, data, kind in sets:
                    for size in sorted(data):
                        v = data[size][kind]
                        if not v:
                            continue
                        med = statistics.median(v)
                        print(f"{thresh} {env} {mode} {link} {size} "
                              f"{med:.2f} {min(v):.2f} {max(v):.2f} {len(v)} "
                              f"{size / med / 1e3:.2f}")


if __name__ == "__main__":
    main()
