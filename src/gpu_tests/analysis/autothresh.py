#!/usr/bin/env python3
"""Автоматический поиск порога eager/rendezvous по измерению (эксперимент experiments/autothresh.conf).

    python3 analysis/autothresh.py results/hyp_<задание> [results/hyp_<уточнение> ...] [--plot autothresh.png] [--refine experiments/autothresh_sizes.txt]

Вход — результаты задания с блоками auto (порог выбирает UCX), eager (UCX_RNDV_THRESH=inf) и rndv
(UCX_RNDV_THRESH=0). Для каждого класса пар — GPU отправителя × (между узлами | внутри узла, GPU одного
процессора) — по каждому размеру берётся медиана медиан пар. Дальше:

  1. Проверка, что принудительный порог подействовал: кривые eager и rndv должны различаться хотя бы на
     части размеров. Если нет — UCX_RNDV_THRESH не влияет на выбор (результат не годится).
  2. Лучший порог по измерению — наименьший размер m*, начиная с которого rndv не медленнее eager на всех
     бо́льших размерах сетки. Точное место пересечения оценивается линейной интерполяцией log(rndv/eager)
     по log(размер) между соседними точками.
  3. Порог, который выбирает UCX, — первый размер, на котором кривая auto ближе к rndv, чем к eager.
  4. Выигрыш — на размерах между двумя порогами: во сколько раз auto медленнее лучшего из eager и rndv.

Пары GPU разных процессоров одного узла в среде custom идут через RAM силами модуля (--sys-via-host),
порог UCX на них не влияет — они не рассматриваются.
Если пересечение попало между точками сетки с шагом больше --tol, пишется файл размеров для
уточняющего прогона (SIZES_FILE=... в autothresh.conf).
"""

from __future__ import annotations

import argparse
import math
import re
import sys
from collections import defaultdict
from pathlib import Path
from statistics import median

NAME_RE = re.compile(r"_(auto|eager|rndv)_rep\d+_s\d+_(\w+?)_one_to_one_b(\d+)_w\d+_i\d+\.txt$")
PAIR_RE = re.compile(r"^pair (\S+) -> (\S+) .*?med_us=([0-9.eE+-]+)")
MODES = ("auto", "eager", "rndv")


def gpu_of(label: str) -> tuple[str, int]:
	"""Метка ранга вида <узел>.<GPU> (как в .txt модуля): узел и номер GPU."""
	node, _, gpu = label.rpartition(".")
	return node, int(gpu)


def pair_class(src: str, dst: str, gpus_per_socket: int) -> str | None:
	sn, sg = gpu_of(src)
	dn, dg = gpu_of(dst)
	if sn != dn:
		return "между узлами"
	if sg // gpus_per_socket == dg // gpus_per_socket:
		return "внутри узла"
	return None   # GPU разных процессоров — через RAM, порог UCX не действует


def load(jobs: list[Path], gpus_per_socket: int):
	"""curves[(класс, GPU отправителя)][режим][размер] = медиана медиан пар."""
	raw = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
	files = 0
	for f in sorted(f for job in jobs for f in job.glob("*.txt")):
		m = NAME_RE.search(f.name)
		if not m:
			continue
		mode, size = m.group(1), int(m.group(3))
		files += 1
		for line in f.read_text(encoding="utf-8", errors="replace").splitlines():
			p = PAIR_RE.match(line)
			if not p or p.group(1) == p.group(2):
				continue
			cls = pair_class(p.group(1), p.group(2), gpus_per_socket)
			if cls is None:
				continue
			key = (cls, gpu_of(p.group(1))[1])
			raw[key][mode][size].append(float(p.group(3)))
	curves = {k: {md: {sz: median(v) for sz, v in by.items()} for md, by in modes.items()} for k, modes in raw.items()}
	return curves, files


def crossover(eager: dict[int, float], rndv: dict[int, float]):
	"""Наименьший размер сетки, начиная с которого rndv <= eager до конца сетки; и оценка точки пересечения."""
	sizes = sorted(set(eager) & set(rndv))
	if not sizes:
		return None
	d = [math.log(rndv[s] / eager[s]) for s in sizes]
	k = len(sizes)
	while k > 0 and d[k - 1] <= 0:
		k -= 1
	if k == len(sizes):
		return {"grid": None, "lo": sizes[-1], "hi": None, "est": None}       # rndv хуже везде
	if k == 0:
		return {"grid": sizes[0], "lo": None, "hi": sizes[0], "est": sizes[0]}  # rndv лучше везде
	lo, hi = sizes[k - 1], sizes[k]
	a, b = d[k - 1], d[k]   # a > 0 >= b
	t = a / (a - b) if a != b else 0.5
	est = math.exp(math.log(lo) + t * (math.log(hi) - math.log(lo)))
	return {"grid": hi, "lo": lo, "hi": hi, "est": int(round(est))}


def ucx_threshold(auto: dict[int, float], eager: dict[int, float], rndv: dict[int, float]):
	"""Порог, который выбрал UCX: (последний размер, где auto идёт как eager; первый размер, начиная с которого
	auto идёт как rndv до конца сетки). Ищется с конца сетки: на малых размерах eager и rndv могут отличаться
	на уровне шума, и случайная точка, где auto чуть ближе к rndv, не должна приниматься за порог.
	Размеры, где eager и rndv отличаются меньше чем на 5%, не решают ничего и пропускаются."""
	sizes = sorted(set(auto) & set(eager) & set(rndv))
	nxt = None
	for s in reversed(sizes):
		distinct = abs(math.log(rndv[s] / eager[s])) > 0.05
		near_eager = abs(math.log(auto[s] / eager[s])) <= abs(math.log(auto[s] / rndv[s]))
		if distinct and near_eager:
			return s, nxt
		if distinct:
			nxt = s
	return None, nxt


def fmt_bytes(b: int | None) -> str:
	if b is None:
		return "—"
	for unit, k in (("МБ", 1 << 20), ("КБ", 1 << 10)):
		if b >= k:
			return f"{b / k:.3g} {unit}".replace(".", ",")
	return f"{b} Б"


def main() -> int:
	ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	ap.add_argument("job", type=Path, nargs="+", help="каталоги результатов заданий autothresh (сетки объединяются)")
	ap.add_argument("--gpus-per-socket", type=int, default=2, help="GPU на процессор (G5500 V7: 2)")
	ap.add_argument("--tol", type=float, default=1.1, help="уточнять, если соседние точки сетки дальше, чем в tol раз")
	ap.add_argument("--refine", type=Path, help="записать сюда размеры для уточняющего прогона")
	ap.add_argument("--plot", type=Path, help="записать график кривых по классам (нужен matplotlib)")
	args = ap.parse_args()

	curves, files = load(args.job, args.gpus_per_socket)
	if not curves:
		print(f"нет результатов autothresh в {[str(j) for j in args.job]} (файлы *_auto|eager|rndv_rep*_..._b*_w*_i*.txt)", file=sys.stderr)
		return 1
	print(f"{', '.join(str(j) for j in args.job)}: файлов {files}, классов пар {len(curves)}\n")

	rows, refine = [], set()
	for key in sorted(curves):
		c = curves[key]
		if not all(m in c for m in MODES):
			print(f"{key}: нет всех трёх режимов, пропуск")
			continue
		e, r, a = c["eager"], c["rndv"], c["auto"]
		common = sorted(set(e) & set(r))
		diff = max(abs(math.log(r[s] / e[s])) for s in common)
		forced_ok = diff > 0.05
		x = crossover(e, r)
		u_lo, u_hi = ucx_threshold(a, e, r)
		gain = None
		if x and x["est"] and u_hi:
			lo, hi = sorted((x["est"], u_hi))
			between = [s for s in a if lo <= s < hi]
			if between:
				gain = max(a[s] / min(e[s], r[s]) for s in between if s in e and s in r)
		rows.append((key, forced_ok, x, (u_lo, u_hi), gain))
		# уточнять нужно только лучший порог: между соседними точками сетки или ниже её начала
		if x and x["lo"] and x["hi"] and x["hi"] / x["lo"] > args.tol:
			lo, hi = x["lo"], x["hi"]
			for i in range(1, 8):
				refine.add(int(round(lo * (hi / lo) ** (i / 8))))
		elif x and x["lo"] is None:          # rndv не медленнее уже с первой точки — пересечение ниже сетки
			b = 1
			while b < x["hi"]:
				refine.add(b)
				b *= 2

	print("| Класс пар | GPU отпр. | Режимы различаются | Лучший порог по измерению | Порог UCX (между) | Выигрыш между порогами |")
	print("|---|---|---|---|---|---|")
	for (cls, gpu), ok, x, (u_lo, u_hi), gain in rows:
		if x is None:
			best = "—"
		elif x["grid"] is None:
			best = f"rndv медленнее на всей сетке (до {fmt_bytes(x['lo'])})"
		elif x["lo"] is None:
			best = f"rndv не медленнее уже с {fmt_bytes(x['hi'])}"
		else:
			best = f"≈ {fmt_bytes(x['est'])} ({x['est']} Б; между {fmt_bytes(x['lo'])} и {fmt_bytes(x['hi'])})"
		ucx = f"{fmt_bytes(u_lo)} … {fmt_bytes(u_hi)}" if u_hi else f"не переключился до {fmt_bytes(u_lo)}"
		g = f"до {gain:.2g} раз".replace(".", ",") if gain and gain > 1.05 else "—"
		print(f"| {cls} | GPU{gpu} | {'да' if ok else 'НЕТ — порог не подействовал'} | {best} | {ucx} | {g} |")

	if any(not ok for _, ok, *_ in rows):
		print("\nВНИМАНИЕ: у части классов eager и rndv совпадают — UCX_RNDV_THRESH, похоже, не меняет выбор протокола"
		      " (см. таблицы в ucx/*proto_* этого задания).")
	print("\nРекомендуемое значение для класса — UCX_RNDV_THRESH=<лучший порог>; порог задаётся на процесс,"
	      " у нас один процесс на GPU, поэтому значение можно задавать по GPU.")

	if refine and args.refine:
		args.refine.write_text("\n".join(str(s) for s in sorted(refine)) + "\n", encoding="utf-8")
		print(f"\nуточняющая сетка: {len(refine)} размеров → {args.refine}"
		      f"\n  запуск: SIZES_FILE={args.refine} bash submit.sh ... task.sbatch autothresh")
	elif refine:
		print(f"\nпересечение попало между редкими точками сетки — для уточнения: --refine experiments/autothresh_sizes.txt")

	if args.plot:
		import matplotlib
		matplotlib.use("Agg")
		import matplotlib.pyplot as plt
		keys = sorted(curves)
		fig, axs = plt.subplots(1, len(keys), figsize=(4.2 * len(keys), 3.8), squeeze=False)
		for ax, key in zip(axs[0], keys):
			for mode, style in (("eager", "o-"), ("rndv", "s-"), ("auto", "--")):
				pts = sorted(curves[key].get(mode, {}).items())
				if pts:
					ax.plot([p[0] for p in pts], [p[1] for p in pts], style, ms=3, label=mode)
			ax.set_xscale("log", base=2)
			ax.set_yscale("log")
			ax.set_title(f"{key[0]}, GPU{key[1]}", fontsize=11)
			ax.set_xlabel("байт")
			ax.grid(alpha=0.4)
		axs[0][0].set_ylabel("медиана задержки, мкс")
		axs[0][0].legend()
		fig.tight_layout()
		fig.savefig(args.plot, dpi=150)
		print(f"график: {args.plot}")
	return 0


if __name__ == "__main__":
	sys.exit(main())
