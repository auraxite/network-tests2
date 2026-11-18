#!/usr/bin/env python3
"""Подбор транспорта UCX по измерению (эксперимент experiments/autotransport_method.conf).

    python3 analysis/autotransport.py results/hyp_<1> [results/hyp_<2> ...] [--default all] [--tol 1.1] [--plot файл.png]

Вход — результаты заданий, где один и тот же набор размеров измерен в нескольких блоках; блоки различаются
набором разрешённых транспортов UCX (all — как в custom, noipc, nic, nogdr; см. описание эксперимента).
Порог протокола в каждом блоке — по умолчанию UCX.

Пары делятся на классы, как в autothresh.py: (между узлами | внутри узла, GPU одного процессора) × GPU
отправителя — транспорт и порог UCX выбирает отправитель. Пары GPU разных процессоров внутри узла пропускаются:
в custom модуль сам ведёт их через RAM, и транспорт для памяти GPU к ним не относится.

Для каждого класса и размера — медиана медиан пар (по всем парам и заданиям) в каждом блоке. Дальше:
  лучший набор — блок с наименьшей задержкой; если он быстрее эталона (--default) меньше чем в --tol раз,
  лучшим считается эталон (разница в пределах шума);
  выигрыш      — задержка эталона / задержка лучшего набора.
Итог по классу — диапазоны размеров, где лучший набор не эталон, и рекомендуемый для них набор.
Запуски с ненулевым кодом возврата (runs.csv, rc) перечисляются: набор мог не подойти для части пар.
"""
import argparse
import csv
import re
import sys
from collections import defaultdict
from pathlib import Path
from statistics import median

NAME_RE = re.compile(r"_([A-Za-z0-9]+)_rep\d+_s\d+_(\w+?)_one_to_one_b(\d+)_w\d+_i\d+\.txt$")
PAIR_RE = re.compile(r"^pair (\S+) -> (\S+) .*?med_us=([0-9.eE+-]+)")

SETS = {   # пояснения к блокам autotransport_method.conf
	"all": "как в custom: cuda_copy, cuda_ipc, InfiniBand, cma, sm, self; GPUDirect RDMA",
	"noipc": "без cuda_ipc: внутри узла через RAM или адаптер",
	"nic": "cuda_copy, InfiniBand, self: внутри узла через адаптер",
	"nogdr": "без GPUDirect RDMA: между узлами через RAM",
}


def gpu_of(label):
	node, _, gpu = label.rpartition(".")
	return node, int(gpu)


def pair_class(src, dst, gps):
	sn, sg = gpu_of(src)
	dn, dg = gpu_of(dst)
	if sn != dn:
		return "между узлами"
	if sg // gps == dg // gps:
		return "внутри узла"
	return None


def fmt_bytes(b):
	for unit, k in (("МБ", 1 << 20), ("КБ", 1 << 10)):
		if b >= k:
			return f"{b / k:.3g} {unit}".replace(".", ",")
	return f"{b} Б"


def fmt_us(x):
	return f"{x:.1f}".replace(".", ",") if x < 1000 else f"{x / 1000:.2f} мс".replace(".", ",")


def load(jobs, gps):
	"""curves[(класс, GPU отправителя)][блок][размер] = медиана медиан пар; failed — упавшие запуски."""
	raw = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
	failed = []
	for job in jobs:
		for f in sorted(job.glob("*.txt")):
			m = NAME_RE.search(f.name)
			if not m:
				continue
			block, size = m.group(1), int(m.group(3))
			for line in f.read_text(encoding="utf-8", errors="replace").splitlines():
				p = PAIR_RE.match(line)
				if not p or p.group(1) == p.group(2):
					continue
				cls = pair_class(p.group(1), p.group(2), gps)
				if cls:
					raw[(cls, gpu_of(p.group(1))[1])][block][size].append(float(p.group(3)))
		runs = job / "runs.csv"
		if runs.exists():
			with runs.open(encoding="utf-8") as fh:
				for r in csv.DictReader(fh):
					if r.get("rc", "0") not in ("0", ""):
						failed.append(f"{job.name}: {r['block']} {fmt_bytes(int(r['bytes']))} rc={r['rc']}")
	curves = {k: {b: {s: median(v) for s, v in by.items()} for b, by in blocks.items()} for k, blocks in raw.items()}
	return curves, failed


def ranges(sizes, best, default):
	"""Слить подряд идущие размеры с одним и тем же лучшим набором (кроме эталона)."""
	out, cur = [], None
	for s in sizes:
		b = best[s]
		if b == default:
			cur = None
			continue
		if cur and cur[2] == b:
			cur[1] = s
		else:
			cur = [s, s, b]
			out.append(cur)
	return out


def main():
	ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	ap.add_argument("job", type=Path, nargs="+", help="каталоги результатов заданий (объединяются)")
	ap.add_argument("--default", default="all", help="эталонный блок (как в среде custom)")
	ap.add_argument("--tol", type=float, default=1.1, help="лучший набор считается лучше эталона, только если быстрее в tol раз")
	ap.add_argument("--gpus-per-socket", type=int, default=2)
	ap.add_argument("--plot", type=Path, help="записать графики задержки по наборам (нужен matplotlib)")
	args = ap.parse_args()

	curves, failed = load(args.job, args.gpus_per_socket)
	if not curves:
		print(f"нет результатов в {[str(j) for j in args.job]}", file=sys.stderr)
		return 1
	blocks = sorted({b for c in curves.values() for b in c}, key=lambda b: (b != args.default, b))
	if args.default not in blocks:
		print(f"нет эталонного блока {args.default!r}; есть: {', '.join(blocks)}", file=sys.stderr)
		return 1
	print(f"{', '.join(j.name for j in args.job)}: блоки {', '.join(blocks)}; эталон — {args.default}\n")
	for b in blocks:
		if b in SETS:
			print(f"- {b}: {SETS[b]}")
	print()

	summary = []
	for key in sorted(curves):
		c = curves[key]
		sizes = sorted(set.intersection(*(set(c[b]) for b in blocks if b in c)))
		print(f"### {key[0]}, отправитель GPU{key[1]}")
		print("| Размер | " + " | ".join(f"{b}, мкс" for b in blocks) + " | Лучший | Выигрыш к эталону |")
		print("|---" * (len(blocks) + 3) + "|")
		best = {}
		for s in sizes:
			vals = {b: c[b][s] for b in blocks if b in c and s in c[b]}
			b_min = min(vals, key=vals.get)
			gain = vals[args.default] / vals[b_min]
			best[s] = b_min if gain >= args.tol else args.default
			cells = " | ".join(fmt_us(vals[b]) if b in vals else "—" for b in blocks)
			g = f"{gain:.2f}".replace(".", ",") if best[s] != args.default else "—"
			print(f"| {fmt_bytes(s)} | {cells} | {best[s]} | {g} |")
		print()
		for lo, hi, b in ranges(sizes, best, args.default):
			span = fmt_bytes(lo) if lo == hi else f"{fmt_bytes(lo)} … {fmt_bytes(hi)}"
			summary.append(f"| {key[0]} | GPU{key[1]} | {span} | {b} |")

	print("## Итог: где быстрее не эталон")
	if summary:
		print("| Класс | GPU отпр. | Размеры | Лучший набор |\n|---|---|---|---|")
		print("\n".join(summary))
	else:
		print(f"на всех размерах эталон ({args.default}) не медленнее остальных наборов (с точностью {args.tol})")
	if failed:
		print("\nЗапуски с ошибкой (набор, вероятно, не подходит части пар):\n- " + "\n- ".join(failed))

	if args.plot:
		import matplotlib
		matplotlib.use("Agg")
		import matplotlib.pyplot as plt
		keys = sorted(curves)
		fig, axs = plt.subplots(1, len(keys), figsize=(4.2 * len(keys), 3.8), squeeze=False)
		for ax, key in zip(axs[0], keys):
			for b in blocks:
				pts = sorted(curves[key].get(b, {}).items())
				if pts:
					ax.plot([p[0] for p in pts], [p[1] for p in pts], "o-" if b == args.default else ".--", ms=3, label=b)
			ax.set_xscale("log", base=2)
			ax.set_yscale("log")
			ax.set_title(f"{key[0]}, GPU{key[1]}", fontsize=11)
			ax.set_xlabel("байт")
			ax.grid(alpha=0.4)
		axs[0][0].set_ylabel("медиана задержки, мкс")
		axs[0][0].legend()
		fig.tight_layout()
		fig.savefig(args.plot, dpi=150)
		print(f"\nграфик: {args.plot}")
	return 0


if __name__ == "__main__":
	sys.exit(main())
