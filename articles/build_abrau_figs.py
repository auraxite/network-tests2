# -*- coding: utf-8 -*-
"""Рисунки для статьи на конференцию «Научный сервис в сети Интернет»
(output/Статья.docx).

Те же графики, что и для сборника АСВК (ASVK/build_asvk_figs.py), но под
Word: шрифт Times New Roman, как в основном тексте шаблона, растр PNG 300 dpi
на ширину полосы набора (16 см).
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "ASVK"))
import build_asvk_figs as b  # noqa: E402

OUT = Path(__file__).resolve().parent / "output" / "figs"
b.plt.rcParams.update({"font.family": "Times New Roman", "font.size": 9,
                       "legend.fontsize": 8.5})
b.WIDTH_IN = 160 * b.MM
b.PLOT_HEIGHT_IN = 2.2

# Сборник конференции только электронный, поэтому рисунки цветные.
# Цвета типов пар те же, что в графиках тезисов: PCIe + InfiniBand — красный,
# PCIe — синий, NVLink — зелёный.
RED, BLUE, GREEN = "#d62728", "#2a78d6", "#1baf7a"
b.LINKS = [
    ("ib", "PCIe + InfiniBand, между узлами G5500 V7",
     dict(color=RED, ls="-", marker="o", mfc=RED)),
    ("pcie", "PCIe, внутри узла G5500 V7",
     dict(color=BLUE, ls="-", marker="s", mfc=BLUE)),
    ("nvlink", "NVLink, внутри узла G8600 V7",
     dict(color=GREEN, ls="-", marker="^", mfc=GREEN)),
]
b.THRESHOLDS = [
    ("default", "auto", "auto, порог по умолчанию",
     dict(color="#e07b00", ls="-", marker="o", mfc="#e07b00")),
    ("0", "auto", "auto, порог 0",
     dict(color=BLUE, ls="-", marker="s", mfc=BLUE)),
    ("32M", "auto", "auto, порог 32M",
     dict(color="#8e44ad", ls="-", marker="D", mfc="#8e44ad")),
    ("0", "host", "host",
     dict(color="0.35", ls="--", marker="^", mfc="white")),
]
b.FILL.update({"GPU": "#d4f0e3", "RAM": "#fde2c4", "NIC": "#d6e6f7"})


def save(fig, name, tight=True):
    OUT.mkdir(parents=True, exist_ok=True)
    if tight:
        fig.tight_layout(pad=0.2)
    fig.savefig(OUT / f"{name}.png", dpi=300)
    b.plt.close(fig)
    print("saved", name)


b.save = save

if __name__ == "__main__":
    b.fig_paths()
    b.fig_thresh("ib", "thresh_ib")
    b.fig_links("auto")
    # В среде host кривые PCIe и NVLink совпадают: NVLink толще и снизу,
    # PCIe тоньше и сверху, чтобы были видны обе.
    b.LINKS[1][2].update(lw=1.0, ms=3.0, zorder=3)
    b.LINKS[2][2].update(lw=3.2, ms=6.5, zorder=2)
    b.fig_links("host")
