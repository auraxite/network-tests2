# -*- coding: utf-8 -*-
"""Рисунки для статьи в кафедральный сборник АСВК.

Требования сборника: иллюстрации чёрно-белые и векторные, каждая в двух
форматах (PDF для pdflatex и EPS для latex), имена файлов — латиница в нижнем
регистре с фамилией и номером группы. Линии различаются оттенком серого,
типом линии и маркером, чтобы оставаться различимыми при печати.

Шрифт — CMU Serif (Computer Modern Unicode из пакета MiKTeX cm-unicode),
тот же, что и в тексте статьи. Размер рисунка равен ширине полосы набора A5
(110 мм), поэтому кегль на рисунке не масштабируется при вставке.

Запуск: python build_asvk_figs.py [--preview DIR] — с --preview дополнительно
сохраняются PNG для просмотра.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.legend_handler import HandlerLine2D
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

from analyze import SOURCES, collect, statistics

OUT = Path(__file__).resolve().parent / "kibizov_421"
PREFIX = "kibizov_421_"
MM = 1 / 25.4
WIDTH_IN = 110 * MM
PLOT_HEIGHT_IN = 1.75
PREVIEW = Path(sys.argv[sys.argv.index("--preview") + 1]) if "--preview" in sys.argv else None

CMU_DIR = Path(r"C:\Users\user\AppData\Local\Programs\MiKTeX\fonts\opentype\public\cm-unicode")
for name in ("cmunrm", "cmunti", "cmunbx"):
    font_manager.fontManager.addfont(str(CMU_DIR / f"{name}.otf"))

plt.rcParams.update({
    "font.family": "CMU Serif",
    "font.size": 8,
    "legend.fontsize": 7.5,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.linewidth": 0.5,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "ytick.minor.width": 0.4,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "ytick.minor.size": 1.5,
    "lines.linewidth": 1.0,
    "lines.markersize": 3.2,
    "lines.markeredgewidth": 0.7,
})

BLACK, DARK, MID = "0.0", "0.3", "0.5"

LINKS = [
    ("ib", "между узлами G5500 V7",
     dict(color=BLACK, ls="-", marker="o", mfc=BLACK)),
    ("pcie", "внутри узла G5500 V7",
     dict(color=DARK, ls="--", marker="s", mfc="white")),
    ("nvlink", "внутри узла G8600 V7",
     dict(color=MID, ls="-.", marker="^", mfc=MID)),
]
# Кривая порога по умолчанию до 256 КБ совпадает с кривой порога выше 16 МБ,
# а после 512 КБ — с кривой порога 0. Поэтому она рисуется поверх остальных
# тонкой линией с мелкими закрашенными маркерами, а остальные — крупными
# полыми: в местах совпадения закрашенный кружок виден внутри полого маркера.
THRESHOLDS = [
    ("default", "auto", "auto, порог по умолчанию",
     dict(color=BLACK, ls="-", lw=0.8, marker="o", ms=2.4, mfc=BLACK, zorder=5)),
    ("0", "auto", "auto, порог 0",
     dict(color=DARK, ls="--", lw=1.1, marker="s", ms=4.4, mfc="white", zorder=3)),
    ("32M", "auto", "auto, порог выше 16 МБ",
     dict(color=MID, ls="-.", lw=1.1, marker="D", ms=4.4, mfc="white", zorder=2)),
    ("0", "host", "host, порог 0",
     dict(color=BLACK, ls=":", lw=1.1, marker="^", ms=4.0, mfc="white", zorder=4)),
]
SIZE_TICKS = [1000, 4000, 16000, 64000, 256000, 1024000, 4096000, 16384000]
SIZE_LABELS = ["1 КБ", "4 КБ", "16 КБ", "64 КБ", "256 КБ", "1 МБ", "4 МБ", "16 МБ"]


def curve(thresh, env, link):
    pci, sxm = SOURCES[thresh]
    data = collect(sxm if link == "nvlink" else pci, env, "one_to_one")
    kind = "inter" if link == "ib" else "intra"
    sizes = sorted(s for s in data if data[s][kind])
    return sizes, [statistics.median(data[s][kind]) for s in sizes]


def setup(ax, ymax):
    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=10)
    ax.xaxis.set_major_locator(FixedLocator(SIZE_TICKS))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xticklabels(SIZE_LABELS)
    ax.set_xlim(800, 20_000_000)

    top = 10 ** len(str(int(ymax)))
    ax.set_ylim(8, top)
    ax.yaxis.set_major_locator(FixedLocator([10 ** k for k in range(1, len(str(top)))]))
    ax.yaxis.set_major_formatter(
        FuncFormatter(lambda v, _: f"{int(v):,}".replace(",", "\u2009")))
    ax.yaxis.set_minor_locator(NullLocator())

    ax.set_xlabel("Размер сообщения")
    ax.set_ylabel("Медианная задержка, мкс")
    ax.grid(True, which="major", color="0.85", lw=0.4)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def save(fig, name, tight=True):
    OUT.mkdir(exist_ok=True)
    if tight:
        fig.tight_layout(pad=0.2)
    for ext in ("pdf", "eps"):
        fig.savefig(OUT / f"{PREFIX}{name}.{ext}")
    if PREVIEW:
        fig.savefig(PREVIEW / f"{PREFIX}{name}.png", dpi=200)
    plt.close(fig)
    print("saved", name)


class SymmetricLine(HandlerLine2D):
    """Образец линии в легенде: две половины от маркера наружу.

    Штриховой узор начинается от первой точки линии, поэтому при обычной
    отрисовке он обрезается по краям несимметрично. Половины, идущие от
    центра в разные стороны, начинаются с одной фазы и зеркальны.
    """

    def create_artists(self, legend, orig_handle, xdescent, ydescent,
                       width, height, fontsize, trans):
        y = (height - ydescent) / 2
        x0, x1 = -xdescent, -xdescent + width
        xc = (x0 + x1) / 2
        artists = []
        for sign in (-1, 1):
            half = Line2D([xc, xc + sign * (x1 - xc)], [y, y])
            self.update_prop(half, orig_handle, legend)
            half.set_marker("")
            half.set_transform(trans)
            # Обрезать половину по концу целого штриха, чтобы на краю
            # образца не оставался обрубок.
            _, seq = half._dash_pattern
            if seq:
                length, pos, i = x1 - xc, 0.0, 0
                end = length
                while pos < length:
                    on = seq[i % len(seq)]
                    if pos + on <= length:
                        end = pos + on
                    pos += on + seq[(i + 1) % len(seq)]
                    i += 2
                half.set_xdata([xc, xc + sign * end])
            artists.append(half)
        mark = Line2D([xc], [y])
        self.update_prop(mark, orig_handle, legend)
        mark.set_linestyle("None")
        mark.set_transform(trans)
        artists.append(mark)
        return artists


def legend(ax):
    ax.legend(frameon=False, loc="upper left", handlelength=3.2,
              borderaxespad=0.2, labelspacing=0.3,
              handler_map={Line2D: SymmetricLine()})


def plot(lines, name):
    fig, ax = plt.subplots(figsize=(WIDTH_IN, PLOT_HEIGHT_IN))
    ymax = 0
    for (x, y), label, style in lines:
        ymax = max(ymax, max(y))
        ax.plot(x, y, label=label, **style)
    setup(ax, ymax)
    legend(ax)
    save(fig, name)


def fig_links(env):
    """Три типа пар в одной среде при UCX_RNDV_THRESH=0."""
    plot([(curve("0", env, key), label, style) for key, label, style in LINKS], env)


def fig_thresh(link, name):
    """Среда auto при трёх порогах UCX и среда host для одного типа пар."""
    plot([(curve(t, env, link), label, style) for t, env, label, style in THRESHOLDS], name)


def fig_speedup():
    """Отношение задержки host к задержке auto при UCX_RNDV_THRESH=0.

    Значение больше 1 — выигрыш передачи без промежуточного копирования.
    """
    fig, ax = plt.subplots(figsize=(WIDTH_IN, PLOT_HEIGHT_IN))
    for key, label, style in LINKS:
        xa, ya = curve("0", "auto", key)
        xh, yh = curve("0", "host", key)
        assert xa == xh
        ax.plot(xa, [h / a for h, a in zip(yh, ya)], label=label, **style)
    ax.axhline(1.0, color="0.4", lw=0.6, ls=(0, (1, 1.5)), zorder=1)

    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=10)
    ax.xaxis.set_major_locator(FixedLocator(SIZE_TICKS))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xticklabels(SIZE_LABELS)
    ax.set_xlim(800, 20_000_000)
    ax.set_ylim(0.5, 25)
    ax.yaxis.set_major_locator(FixedLocator([0.5, 1, 2, 5, 10, 20]))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.set_xlabel("Размер сообщения")
    ax.set_ylabel("Во сколько раз\nauto быстрее host")
    note = dict(fontsize=6.5, style="italic", color="0.3", ha="left")
    ax.text(40_000, 1.1, "выше 1: быстрее auto", va="bottom", **note)
    ax.text(40_000, 0.555, "ниже 1: быстрее host", va="center", **note)
    ax.grid(True, which="major", color="0.85", lw=0.4)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    legend(ax)
    save(fig, "speedup")


# --- Схема путей передачи ---------------------------------------------------
# Шесть колонок: GPU, RAM, NIC узла 1 и NIC, RAM, GPU узла 2. Каждая строка —
# один путь; блоки выровнены по колонкам, чтобы пути было легко сравнить.
# Все подписи на схеме английские: смесь «GPU», «PCIe» и русских сокращений
# читалась плохо.
BOX_W, BOX_H = 11.0, 6.0
GAP_BETWEEN_NODES = 18.0  # вмещает подпись «InfiniBand»
GAP = (110 - 6 * BOX_W - GAP_BETWEEN_NODES) / 4
COL_X = [BOX_W / 2 + i * (BOX_W + GAP) + (GAP_BETWEEN_NODES - GAP if i >= 3 else 0)
         for i in range(6)]
FILL = {"GPU": "white", "RAM": "0.82", "NIC": "white"}

# Межузловая сеть: в измерениях InfiniBand, но схема общая.
NET = "InfiniBand\nили Ethernet"

PATHS = [
    ("a) внутри узла, без промежуточного копирования",
     [(0, "GPU"), (2, "GPU")], ["NVLink или PCIe"]),
    ("b) внутри узла, через оперативную память",
     [(0, "GPU"), (1, "RAM"), (2, "GPU")], ["PCIe", "PCIe"]),
    ("c) между узлами, без промежуточного копирования (GPUDirect RDMA)",
     [(0, "GPU"), (2, "NIC"), (3, "NIC"), (5, "GPU")], ["PCIe", NET, "PCIe"]),
    ("d) между узлами, через оперативную память",
     [(0, "GPU"), (1, "RAM"), (2, "NIC"), (3, "NIC"), (4, "RAM"), (5, "GPU")],
     ["PCIe", "PCIe", NET, "PCIe", "PCIe"]),
]
ROW_H = 10.5


def fig_paths():
    height = 8 + ROW_H * len(PATHS)
    fig = plt.figure(figsize=(WIDTH_IN, height * MM))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 110)
    ax.set_ylim(0, height)
    ax.axis("off")

    top = height - 1.5
    ax.text((COL_X[0] + COL_X[2]) / 2, top, "Узел 1", ha="center", va="top",
            fontsize=8, fontweight="bold")
    ax.text((COL_X[3] + COL_X[5]) / 2, top, "Узел 2", ha="center", va="top",
            fontsize=8, fontweight="bold")

    for row, (title, boxes, links) in enumerate(PATHS):
        y0 = height - 8 - ROW_H * row
        ax.text(0.3, y0 - 0.5, title, ha="left", va="top", fontsize=7.5,
                style="italic")
        yc = y0 - 4.5 - BOX_H / 2
        for col, label in boxes:
            x = COL_X[col]
            ax.add_patch(FancyBboxPatch(
                (x - BOX_W / 2, yc - BOX_H / 2), BOX_W, BOX_H,
                boxstyle="round,pad=0,rounding_size=0.8",
                fc=FILL[label], ec="black", lw=0.6))
            ax.text(x, yc, label, ha="center", va="center", fontsize=8)
        for (c1, _), (c2, _), link in zip(boxes, boxes[1:], links):
            x1, x2 = COL_X[c1] + BOX_W / 2, COL_X[c2] - BOX_W / 2
            ib = link.startswith("InfiniBand")
            ax.add_patch(FancyArrowPatch(
                (x1, yc), (x2, yc), arrowstyle="-|>", mutation_scale=6,
                lw=1.4 if ib else 0.7, color="black", shrinkA=0.4, shrinkB=0.4))
            above, _, below = link.partition("\n")
            ax.text((x1 + x2) / 2, yc + 0.8, above, ha="center", va="bottom",
                    fontsize=6.5)
            if below:
                ax.text((x1 + x2) / 2, yc - 0.8, below, ha="center", va="top",
                        fontsize=6.5)
    save(fig, "paths", tight=False)


# --- Архитектура узла с несколькими GPU --------------------------------------
# Обобщённая схема по мотивам схем узлов NVIDIA с GPU: два процессора с
# межпроцессорным соединением, коммутаторы PCIe, GPU, связанные NVLink, и
# сетевые адаптеры, выходящие на коммутатор InfiniBand или Ethernet. Не
# описывает конкретный узел кластера. Подписи блоков английские, как на
# схеме путей.
def fig_arch():
    width, height = 110.0, 58.0
    fig = plt.figure(figsize=(WIDTH_IN, height * MM))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.axis("off")

    PCIE = dict(color="black", lw=0.6)
    NVL = dict(color="0.45", lw=2.6, solid_capstyle="butt")
    UPI = dict(color="black", lw=0.8, ls=(0, (3, 2)))
    NET_STYLE = dict(color="black", lw=1.4, ls=(0, (5, 1.5, 1, 1.5)))

    def box(x, y, w, h, label, fc="white", fs=7.5):
        ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                                    boxstyle="round,pad=0,rounding_size=0.8",
                                    fc=fc, ec="black", lw=0.6, zorder=3))
        ax.text(x, y, label, ha="center", va="center", fontsize=fs, zorder=4)

    def line(xs, ys, style):
        ax.plot(xs, ys, zorder=2, **style)

    # Рамка узла
    ax.add_patch(FancyBboxPatch((1, 10), 83, 44, boxstyle="round,pad=0,rounding_size=1.5",
                                fc="none", ec="0.35", lw=0.6, ls=(0, (2, 1.5)), zorder=1))
    ax.text(43, 52.6, "Узел", ha="center", va="top", fontsize=8, fontweight="bold")

    y_cpu, y_sw, y_gpu, y_nvl = 46, 35, 24, 15.5
    x_ram0, x_cpu0, x_cpu1, x_ram1 = 11, 31, 55, 75
    gpus = [16, 32, 52, 68]

    # Процессоры, память, межпроцессорное соединение
    box(x_ram0, y_cpu, 11, 6, "RAM", fc="0.82")
    box(x_ram1, y_cpu, 11, 6, "RAM", fc="0.82")
    box(x_cpu0, y_cpu, 11, 6, "CPU")
    box(x_cpu1, y_cpu, 11, 6, "CPU")
    line([x_ram0 + 5.5, x_cpu0 - 5.5], [y_cpu, y_cpu], dict(color="0.4", lw=1.0))
    line([x_cpu1 + 5.5, x_ram1 - 5.5], [y_cpu, y_cpu], dict(color="0.4", lw=1.0))
    line([x_cpu0 + 5.5, x_cpu1 - 5.5], [y_cpu, y_cpu], UPI)

    # Коммутаторы PCIe и сетевые адаптеры
    for xc, xn in ((x_cpu0, x_ram0), (x_cpu1, x_ram1)):
        box(xc, y_sw, 13, 6.5, "PCIe\nswitch", fs=6.5)
        line([xc, xc], [y_cpu - 3, y_sw + 3.25], PCIE)
        box(xn, y_sw, 11, 6, "NIC")
        side = 1 if xn > xc else -1
        line([xc + side * 6.5, xn - side * 5.5], [y_sw, y_sw], PCIE)

    # GPU: каждый подключён к коммутатору PCIe своего процессора
    for i, xg in enumerate(gpus):
        box(xg, y_gpu, 10, 6, "GPU")
        xs = x_cpu0 if i < 2 else x_cpu1
        line([xg, xg, xs + (xg - xs) * 0.35, xs + (xg - xs) * 0.35],
             [y_gpu + 3, y_gpu + 6.5, y_gpu + 6.5, y_sw - 3.25], PCIE)
        line([xg, xg], [y_gpu - 3, y_nvl], NVL)
    line([gpus[0] - 1.3, gpus[-1] + 1.3], [y_nvl, y_nvl], NVL)
    ax.text((gpus[1] + gpus[2]) / 2, y_nvl - 1.3, "NVLink или NVSwitch",
            ha="center", va="top", fontsize=6.5)

    # Сеть: оба адаптера выходят на коммутатор InfiniBand / Ethernet
    x_net, y_net = 98, y_sw
    box(x_net, y_net, 19, 9, "InfiniBand\nили Ethernet\nswitch", fs=6)
    line([x_ram1 + 5.5, x_net - 9.5], [y_sw, y_sw], NET_STYLE)
    line([x_ram0 - 5.5, 3.2, 3.2, x_net, x_net],
         [y_sw, y_sw, 56.5, 56.5, y_net + 4.5], NET_STYLE)
    ax.add_patch(FancyArrowPatch((x_net, y_net - 4.5), (x_net, 21),
                                 arrowstyle="-|>", mutation_scale=6, lw=1.4,
                                 color="black"))
    ax.text(x_net, 20, "к другим\nузлам", ha="center", va="top", fontsize=6.5)

    # Легенда
    items = [("NVLink", NVL), ("PCIe", PCIE), ("межпроцессорный\nинтерфейс", UPI),
             ("InfiniBand /\nEthernet", NET_STYLE)]
    for x, (label, style) in zip((2, 24, 44, 78), items):
        line([x, x + 7], [4, 4], style)
        ax.text(x + 8.5, 4, label, ha="left", va="center", fontsize=6.5)
    save(fig, "arch", tight=False)


if __name__ == "__main__":
    # Графики среды host и порогов для NVLink в статью не вошли, их цифры
    # приведены в тексте. Функции оставлены для отдельного построения.
    fig_arch()
    fig_paths()
    fig_thresh("ib", "thresh_ib")
    fig_speedup()
    # fig_links("auto")  # рис. с задержкой в auto исключён из статьи ради объёма
