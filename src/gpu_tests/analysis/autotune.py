#!/usr/bin/env python3
"""Совместный подбор набора транспортов UCX и порога протокола (эксперимент experiments/autotune_method.conf).

    python3 analysis/autotune.py results/hyp_<1> [results/hyp_<2> ...] [--default all] [--gpus-per-socket 2]

Блоки называются <набор><протокол>: E — eager (UCX_RNDV_THRESH=inf), R — rendezvous (=0), A — порог UCX.
Пары делятся на классы, как в autothresh.py: (между узлами | внутри узла, GPU одного процессора) × GPU отправителя;
пары GPU разных процессоров внутри узла пропускаются (модуль ведёт их через RAM).

Алгоритм:
  1. Для каждого набора T и класса — кривые E_T(m), R_T(m): медиана медиан пар по всем парам и заданиям.
  2. Лучший порог набора t*(T) — размер, начиная с которого R_T не медленнее E_T на всех бо́льших размерах
     (как в autothresh.py; между точками сетки — оценка интерполяцией). Задержка набора при его лучшем пороге:
     L_T(m) = E_T(m) на размерах сетки ниже порога, R_T(m) — начиная с него.
  3. L_best(m) — минимум L_T(m) по наборам (по каждому классу и размеру).
  4. Оценка набора — среднее геометрическое L_T(m) / L_best(m) по всем классам и размерам (1 — везде лучший);
     дополнительно — наихудшее отношение (минимакс).
  5. Ответ — набор с наименьшей оценкой и его пороги: UCX_TLS=… UCX_RNDV_THRESH=intra:…,inter:…
     (порог по классу — медиана оценок по GPU отправителя: порог UCX задаётся на класс, а не на GPU).
Для сравнения — оценка эталона с порогом UCX (блок <default>A): во сколько раз подобранная настройка быстрее.
"""
import argparse
import csv
import math
import re
import sys
from collections import defaultdict
from pathlib import Path
from statistics import median

sys.path.insert(0, str(Path(__file__).resolve().parent))
from autothresh import crossover, fmt_bytes, gpu_of, pair_class  # noqa: E402

NAME_RE = re.compile(r"_([a-z]+)([EAR])_rep\d+_s\d+_(\w+?)_one_to_one_b(\d+)_w\d+_i\d+\.txt$")
PAIR_RE = re.compile(r"^pair (\S+) -> (\S+) .*?med_us=([0-9.eE+-]+)")
SETS = {
	"all": "cuda_copy, cuda_ipc, InfiniBand, cma, sm, self; GPUDirect RDMA (как custom)",
	"noipc": "без cuda_ipc",
	"nic": "cuda_copy, InfiniBand, self",
	"nogdr": "как all, без GPUDirect RDMA",
}


def load(jobs, gps):
	"""curves[(класс, GPU)][набор][протокол][размер] = медиана медиан пар."""
	raw = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))
	failed = []
	for job in jobs:
		for f in sorted(job.glob("*.txt")):
			m = NAME_RE.search(f.name)
			if not m:
				continue
			tset, proto, size = m.group(1), m.group(2), int(m.group(4))
			for line in f.read_text(encoding="utf-8", errors="replace").splitlines():
				p = PAIR_RE.match(line)
				if not p or p.group(1) == p.group(2):
					continue
				cls = pair_class(p.group(1), p.group(2), gps)
				if cls:
					raw[(cls, gpu_of(p.group(1))[1])][tset][proto][size].append(float(p.group(3)))
		runs = job / "runs.csv"
		if runs.exists():
			with runs.open(encoding="utf-8") as fh:
				for r in csv.DictReader(fh):
					if r.get("rc", "0") not in ("0", ""):
						failed.append(f"{job.name}: {r['block']} {fmt_bytes(int(r['bytes']))} rc={r['rc']}")
	curves = {k: {t: {pr: {s: median(v) for s, v in by.items()} for pr, by in ps.items()} for t, ps in ts.items()}
	          for k, ts in raw.items()}
	return curves, failed


def tuned_curve(e, r):
	"""Задержка при лучшем пороге (по точкам сетки) и сам порог: (кривая, оценка порога, подпись)."""
	x = crossover(e, r)
	sizes = sorted(set(e) & set(r))
	if x is None:
		return {}, None, "—"
	if x["grid"] is None:                     # rendezvous медленнее везде
		return {s: e[s] for s in sizes}, None, "∞ (только eager)"
	if x["lo"] is None:                       # rendezvous не медленнее с первой точки
		return {s: r[s] for s in sizes}, 0, f"0 (rendezvous с {fmt_bytes(x['hi'])})"
	return ({s: (e[s] if s < x["hi"] else r[s]) for s in sizes}, x["est"],
	        f"≈ {fmt_bytes(x['est'])} (между {fmt_bytes(x['lo'])} и {fmt_bytes(x['hi'])})")


def gmean(v):
	return math.exp(sum(math.log(x) for x in v) / len(v)) if v else float("nan")


def num(x, nd=2):
	return f"{x:.{nd}f}".replace(".", ",")


def main():
	ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	ap.add_argument("job", type=Path, nargs="+")
	ap.add_argument("--default", default="all", help="эталонный набор (с блоком <набор>A — порог UCX)")
	ap.add_argument("--gpus-per-socket", type=int, default=2)
	args = ap.parse_args()

	curves, failed = load(args.job, args.gpus_per_socket)
	if not curves:
		print(f"нет результатов autotune в {[str(j) for j in args.job]}", file=sys.stderr)
		return 1
	sets = sorted({t for c in curves.values() for t in c})
	print(f"{', '.join(j.name for j in args.job)}: наборы {', '.join(sets)}\n")

	tuned, thr = {}, defaultdict(dict)         # tuned[(key, T)] = кривая; thr[T][key] = (оценка, подпись)
	for key, c in curves.items():
		for t in sets:
			if "E" in c.get(t, {}) and "R" in c.get(t, {}):
				curve, est, label = tuned_curve(c[t]["E"], c[t]["R"])
				tuned[(key, t)] = curve
				thr[t][key] = (est, label)

	best = {}
	for key in curves:
		for t in sets:
			for s, v in tuned.get((key, t), {}).items():
				best[(key, s)] = min(best.get((key, s), v), v)

	scores = {}
	for t in sets:
		ratios = [v / best[(key, s)] for key in curves for s, v in tuned.get((key, t), {}).items()]
		if ratios:
			scores[t] = (gmean(ratios), max(ratios), len(ratios))
	full = max(n for _, _, n in scores.values())

	print("## Наборы при своём лучшем пороге (1 — везде лучший)")
	print("| Набор | Средний проигрыш | Наихудший | Точек | Состав |\n|---|---|---|---|---|")
	order = sorted(scores, key=lambda t: (scores[t][2] < full, scores[t][0]))
	for t in order:
		g, w, n = scores[t]
		note = "" if n == full else " (неполные данные)"
		print(f"| {t} | {num(g)} | {num(w)} | {n}{note} | {SETS.get(t, '')} |")

	win = order[0]
	print(f"\n## Лучший порог для каждого набора (по GPU отправителя)")
	for t in order:
		print(f"- **{t}**: " + "; ".join(f"{k[0]}, GPU{k[1]}: {thr[t][k][1]}" for k in sorted(thr[t])))

	rec = {}
	for cls, name in (("внутри узла", "intra"), ("между узлами", "inter")):
		ests = [thr[win][k][0] for k in thr[win] if k[0] == cls]
		if not ests:
			continue
		if any(e is None for e in ests):
			rec[name] = "inf"
		else:
			rec[name] = str(int(round(median(ests))))
	tls = {"all": "(как в custom)", "noipc": "cuda_copy,<IB>,cma,sm,self", "nic": "cuda_copy,<IB>,self",
	       "nogdr": "(как в custom) UCX_IB_GPU_DIRECT_RDMA=n"}.get(win, win)
	print(f"\n## Ответ\nнабор **{win}** — UCX_TLS={tls}  UCX_RNDV_THRESH="
	      + ",".join(f"{k}:{v}" for k, v in rec.items()))

	ref = [(key, s, v) for key, c in curves.items() for s, v in c.get(args.default, {}).get("A", {}).items()]
	if ref:
		r_def = [v / best[(key, s)] for key, s, v in ref if (key, s) in best]
		gain = [v / tuned[(key, win)][s] for key, s, v in ref if s in tuned.get((key, win), {})]
		print(f"\nЭталон {args.default} с порогом UCX: средний проигрыш {num(gmean(r_def))}, наихудший {num(max(r_def))}. "
		      f"Подобранная настройка быстрее эталона в среднем в {num(gmean(gain))} раза, максимум в {num(max(gain))}.")
	if failed:
		print("\nЗапуски с ошибкой:\n- " + "\n- ".join(failed))
	return 0


if __name__ == "__main__":
	sys.exit(main())
