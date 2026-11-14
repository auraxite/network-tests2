#!/usr/bin/env python3
"""MPI_Send против MPI_Isend+MPI_Wait в one_to_one (эксперимент experiments/p2pcheck.conf).

    python3 analysis/p2pcheck.py results/hyp_<задание> [--gpus-per-socket 2]

Для каждого класса пар (между узлами | внутри узла, GPU одного процессора | GPU разных процессоров) и размера —
медиана медиан пар для blk (MPI_Send/MPI_Recv) и nb (MPI_Isend/MPI_Irecv + MPI_Wait) и отношение nb / blk:
меньше 1 — неблокирующий вызов быстрее (UCX выбрал другой протокол).
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path
from statistics import median

NAME_RE = re.compile(r"_(blk|nb)_rep\d+_s\d+_\w+?_one_to_one_b(\d+)_w\d+_i\d+\.txt$")
PAIR_RE = re.compile(r"^pair (\S+) -> (\S+) .*?med_us=([0-9.eE+-]+)")
CLASSES = ("между узлами", "внутри узла", "GPU разных процессоров")


def pair_class(src: str, dst: str, per_socket: int) -> str:
	sn, _, sg = src.rpartition(".")
	dn, _, dg = dst.rpartition(".")
	if sn != dn:
		return CLASSES[0]
	return CLASSES[1] if int(sg) // per_socket == int(dg) // per_socket else CLASSES[2]


def main() -> int:
	ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	ap.add_argument("job", type=Path, nargs="+")
	ap.add_argument("--gpus-per-socket", type=int, default=2)
	args = ap.parse_args()
	vals = defaultdict(list)
	for f in sorted(f for j in args.job for f in j.glob("*.txt")):
		m = NAME_RE.search(f.name)
		if not m:
			continue
		var, size = m.group(1), int(m.group(2))
		for line in f.read_text(encoding="utf-8", errors="replace").splitlines():
			p = PAIR_RE.match(line)
			if p and p.group(1) != p.group(2):
				vals[(pair_class(p.group(1), p.group(2), args.gpus_per_socket), size, var)].append(float(p.group(3)))
	if not vals:
		print("нет файлов *_blk|nb_rep*_one_to_one_b*_w*_i*.txt", file=sys.stderr)
		return 1
	sizes = sorted({k[1] for k in vals})
	print("| Класс пар | " + " | ".join(f"{s} Б" for s in sizes) + " |")
	print("|---|" + "---|" * len(sizes))
	for cls in CLASSES:
		cells = []
		for s in sizes:
			b, n = vals.get((cls, s, "blk")), vals.get((cls, s, "nb"))
			if b and n:
				mb, mn = median(b), median(n)
				cells.append(f"{mb:.1f} / {mn:.1f} ({mn / mb:.2f})".replace(".", ","))
			else:
				cells.append("—")
		print(f"| {cls} | " + " | ".join(cells) + " |")
	print("\nВ ячейке: MPI_Send / MPI_Isend+Wait, мкс (отношение). Меньше 1 — неблокирующий вызов быстрее.")
	return 0


if __name__ == "__main__":
	sys.exit(main())
