#!/usr/bin/env python3
"""Нужен ли прогрев передачами (эксперимент experiments/warmup_method.conf).

    python3 analysis/warmup.py results/hyp_<1> [results/hyp_<2> ...] [--gpus-per-socket 2]

По каждому блоку (meth — по методике, bare — без нагрузки CPU и диагностики), размеру и классу пар
(между узлами / внутри узла, один процессор / внутри узла, разные процессоры) печатает для прогрева
0 / 100 / 1000 передач:
  медиана      — медиана задержки пары (из .txt), медиана по парам в задании, затем медиана по заданиям;
  к w0         — во сколько раз медиана больше, чем без прогрева (медиана по парам и заданиям отношений);
  1-я итер.    — первая замеренная итерация пары / установившийся уровень (медиана итераций 100…конец);
  итер. 1–10   — медиана первых 10 итераций / установившийся уровень;
  1-я пара     — медиана первой пары запуска / медиана остальных пар того же класса в этом запуске.
Если прогрев полезен, при w0 «1-я итер.» и «итер. 1–10» заметно больше 1, а с ростом прогрева падают к 1.
"""
import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path
from statistics import median

NAME_RE = re.compile(r"_(\w+?)_rep\d+_s\d+_(\w+?)_one_to_one_b(\d+)_w(\d+)_i\d+\.txt$")
PAIR_RE = re.compile(r"^pair (\S+) -> (\S+) .*?med_us=([0-9.eE+-]+)")
ORDER_RE = re.compile(r"^pair_time (\S+) -> (\S+) index=(\d+)")
STEADY_FROM = 100


def gpu_of(label):
	node, _, gpu = label.rpartition(".")
	return node, int(gpu)


def pair_class(src, dst, gps):
	sn, sg = gpu_of(src)
	dn, dg = gpu_of(dst)
	if sn != dn:
		return "между узлами"
	return "внутри узла, один процессор" if sg // gps == dg // gps else "внутри узла, разные процессоры"


def raw_series(path):
	vals = []
	for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
		if line.startswith("#"):
			continue
		parts = line.split()
		if len(parts) >= 2:
			vals.append(float(parts[1]))
	return vals


def fmt(x, nd=2):
	return "—" if x is None else f"{x:.{nd}f}".replace(".", ",")


def med_or_none(v):
	return median(v) if v else None


def main():
	ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	ap.add_argument("job", type=Path, nargs="+")
	ap.add_argument("--gpus-per-socket", type=int, default=2)
	args = ap.parse_args()

	# (блок, размер, класс, прогрев) -> показатели
	per_job_med = defaultdict(lambda: defaultdict(list))   # key -> job -> [медианы пар]
	pair_med = {}                                           # (job, блок, размер, прогрев, src, dst) -> медиана
	first_it = defaultdict(list)
	head10 = defaultdict(list)
	first_pair = defaultdict(list)
	files = 0
	for job in args.job:
		for f in sorted(job.glob("*.txt")):
			m = NAME_RE.search(f.name)
			if not m:
				continue
			files += 1
			block, size, w = m.group(1), int(m.group(3)), int(m.group(4))
			text = f.read_text(encoding="utf-8", errors="replace").splitlines()
			meds = {}
			order = {}
			for line in text:
				p = PAIR_RE.match(line)
				if p and p.group(1) != p.group(2):
					meds[(p.group(1), p.group(2))] = float(p.group(3))
				o = ORDER_RE.match(line)
				if o:
					order[(o.group(1), o.group(2))] = int(o.group(3))
			for (s, d), v in meds.items():
				cls = pair_class(s, d, args.gpus_per_socket)
				key = (block, size, cls, w)
				per_job_med[key][job.name].append(v)
				pair_med[(job.name, block, size, w, s, d)] = v
				raw = job / "raw" / f"{f.stem}_src{s}_dst{d}.raw"
				if raw.exists():
					x = raw_series(raw)
					if len(x) > STEADY_FROM + 10:
						st = median(x[STEADY_FROM:])
						first_it[key].append(x[0] / st)
						head10[key].append(median(x[:10]) / st)
			if order:
				first = min(order, key=order.get)
				if first in meds:
					cls = pair_class(*first, args.gpus_per_socket)
					rest = [v for k, v in meds.items() if k != first and pair_class(*k, args.gpus_per_socket) == cls]
					if rest:
						first_pair[(block, size, cls, w)].append(meds[first] / median(rest))

	if not per_job_med:
		print(f"нет результатов warmup_method в {[str(j) for j in args.job]}", file=sys.stderr)
		return 1
	print(f"{', '.join(j.name for j in args.job)}: файлов {files}\n")

	keys = sorted({k[:3] for k in per_job_med})
	ws = sorted({k[3] for k in per_job_med})
	for block, size, cls in keys:
		print(f"### {block}, {size} Б, {cls}")
		print("| прогрев | медиана, мкс | к w0 | 1-я итер. | итер. 1–10 | 1-я пара запуска |")
		print("|---|---|---|---|---|---|")
		for w in ws:
			key = (block, size, cls, w)
			if key not in per_job_med:
				continue
			m = median(median(v) for v in per_job_med[key].values())
			ratios = []
			for (j, b, s, ww, src, dst), v in pair_med.items():
				if (b, s, ww) == (block, size, w) and pair_class(src, dst, args.gpus_per_socket) == cls:
					base = pair_med.get((j, b, s, 0, src, dst))
					if base:
						ratios.append(v / base)
			print(f"| {w} | {fmt(m, 1)} | {fmt(med_or_none(ratios), 3)} | {fmt(med_or_none(first_it[key]))} "
			      f"| {fmt(med_or_none(head10[key]))} | {fmt(med_or_none(first_pair[key]))} (n={len(first_pair[key])}) |")
		print()
	return 0


if __name__ == "__main__":
	sys.exit(main())
