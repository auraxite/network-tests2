# -*- coding: utf-8 -*-
"""Рис. 1: медианная задержка в зависимости от размера сообщения.

Запуск: python build_fig_latency.py <auto|host> <default|0|32M>
Три кривые (режим one_to_one):
  NVLink, внутри узла    — sxm4 (g8600), все пары на одном узле;
  PCIe, внутри узла      — pci5 (g5500), пары на одном узле;
  PCIe + InfiniBand        — pci5 (g5500), пары на разных узлах.
Для каждого размера берётся медиана по парам от медианных задержек пар.
"""
import math
import re
import statistics
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, NullLocator, ScalarFormatter

DESKTOP = Path(__file__).resolve().parents[2]
LAST = Path(__file__).resolve().parent / "LAST_RESULTS"
ENV = sys.argv[1] if len(sys.argv) > 1 else "auto"
# Порог UCX_RNDV_THRESH: default — не задан (данные в ~/Desktop/pci5, sxm4),
# 0 и 32M — повторные запуски из LAST_RESULTS.
THRESH = sys.argv[2] if len(sys.argv) > 2 else "default"
VARIANTS = {
    "default": (DESKTOP / "pci5", DESKTOP / "sxm4", "default",
                "UCX_RNDV_THRESH по умолчанию"),
    "0": (LAST / "results_0_PCI5", LAST / "results_0_SXM4", "thresh0",
          "UCX_RNDV_THRESH=0"),
    "32M": (LAST / "results_32M_PCI5", LAST / "results_32M_SXM4", "thresh32M",
            "UCX_RNDV_THRESH=32M"),
}
PCI_DIR, SXM_DIR, TAG, TITLE_SUFFIX = VARIANTS[THRESH]
OUT = (Path(__file__).resolve().parent / "output"
       / f"fig_latency_{ENV}_one_to_one_{TAG}.png")

PAIR_RE = re.compile(r"^pair (\d+)\.(\d+) -> (\d+)\.(\d+) .*?med_us=([\d.]+)")
NAME_RE = re.compile(r"_one_to_one_b(\d+)_")


def collect(folder, env):
    """{размер: {'intra': [med_us...], 'inter': [...]}} для среды env."""
    data = {}
    for f in folder.glob(f"*_{env}_one_to_one_b*.txt"):
        size = int(NAME_RE.search(f.name).group(1))
        bucket = data.setdefault(size, {"intra": [], "inter": []})
        for line in f.read_text(encoding="utf-8", errors="replace").splitlines():
            m = PAIR_RE.match(line)
            if not m:
                continue
            sn, sg, dn, dg, med = m.groups()
            if (sn, sg) == (dn, dg):
                continue
            bucket["intra" if sn == dn else "inter"].append(float(med))
    return data


def curve(data, kind):
    sizes = sorted(s for s in data if data[s][kind])
    return sizes, [statistics.median(data[s][kind]) for s in sizes]


sxm4 = collect(SXM_DIR, ENV)
pci5 = collect(PCI_DIR, ENV)

IB = "PCIe + InfiniBand, внутри и между узлами"
PCIE = "PCIe, внутри узла"
NVLINK = "NVLink, внутри узла"
series = [
    (IB, curve(pci5, "inter"), "#d62728"),
    (PCIE, curve(pci5, "intra"), "#2a78d6"),
    (NVLINK, curve(sxm4, "intra"), "#1baf7a"),
]
# При пороге 32M кривые PCIe и InfiniBand почти совпадают: InfiniBand
# рисуется толстой линией снизу, PCIe тонкой сверху, чтобы были видны обе.
style = {label: dict(lw=1.6, ms=5, zorder=2) for label, _, _ in series}
if THRESH == "32M" and ENV == "auto":
    style[IB] = dict(lw=4.0, ms=9, zorder=2)
    style[PCIE] = dict(lw=1.2, ms=4, zorder=3)
# В среде host на крупных сообщениях так же совпадают PCIe и NVLink.
if ENV == "host":
    style[NVLINK] = dict(lw=3.4, ms=8, zorder=2)
    style[PCIE] = dict(lw=1.2, ms=4, zorder=3)

plt.rcParams.update({"font.family": "Times New Roman", "font.size": 11})
# Высота подобрана так, чтобы оба рисунка тезисов (auto и host) помещались
# на одной странице вместе с текстом результатов.
fig, ax = plt.subplots(figsize=(6.3, 2.7), dpi=300)
for label, (x, y), color in series:
    ax.plot(x, y, "-", color=color, marker="o", mfc=color, mec=color, mew=0,
            label=label, **style[label])
    print(f"{label}: " + ", ".join(f"{s}:{v:.2f}" for s, v in zip(x, y)))

ax.set_title(f"среда {ENV}, режим one_to_one, " + TITLE_SUFFIX, fontsize=11)

ax.set_xscale("log", base=2)
ax.set_yscale("log", base=2)
ticks = [1000, 4000, 16000, 64000, 256000, 1024000, 4096000, 16384000]
ax.xaxis.set_major_locator(FixedLocator(ticks))
ax.xaxis.set_minor_locator(NullLocator())
ax.set_xticklabels(["1 КБ", "4 КБ", "16 КБ", "64 КБ", "256 КБ",
                    "1 МБ", "4 МБ", "16 МБ"])
# Верх оси — ближайшая степень двойки над максимумом данных.
YMAX = 2 ** math.ceil(math.log2(max(max(y) for _, (_, y), _ in series)))
yt = [2 ** k for k in range(3, int(math.log2(YMAX)) + 1, 2)]
ax.yaxis.set_major_locator(FixedLocator(yt))
ax.yaxis.set_major_formatter(ScalarFormatter())
ax.yaxis.set_minor_locator(NullLocator())
ax.set_ylim(8, YMAX)
ax.set_xlabel("Размер сообщения (log шкала)")
ax.set_ylabel("Медианная задержка, мкс")
ax.grid(True, which="major", color="#d9d9d9", lw=0.6)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.legend(frameon=False, loc="upper left")
fig.tight_layout()
OUT.parent.mkdir(exist_ok=True)
fig.savefig(OUT)
print("saved", OUT)
