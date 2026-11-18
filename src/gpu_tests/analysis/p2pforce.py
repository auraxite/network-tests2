#!/usr/bin/env python3
"""MPI_Send против MPI_Isend+MPI_Wait при одном и том же протоколе (эксперимент experiments/p2pforce_method.conf).

    python3 analysis/p2pforce.py results/hyp_<1> [results/hyp_<2> ...] [--gpus-per-socket 2]

Блоки <вызов><протокол>: вызов blk (MPI_Send/MPI_Recv) или nb (MPI_Isend/MPI_Irecv + MPI_Wait); протокол
A — порог UCX по умолчанию, E — только eager (UCX_RNDV_THRESH=inf), R — только rendezvous (=0).
Для каждого класса пар (между узлами | внутри узла, GPU одного процессора | разных процессоров) и размера —
медиана медиан пар (все пары всех заданий) в каждом блоке и отношение nb / blk при каждом протоколе:
меньше 1 — MPI_Isend быстрее. Если при E и при R отношение близко к 1, а при A — нет, выигрыш MPI_Isend
объясняется только тем, что при пороге по умолчанию UCX выбирает для него другой протокол.
"""
import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path
from statistics import median

sys.path.insert(0, str(Path(__file__).resolve().parent))
from autothresh import fmt_bytes, gpu_of  # noqa: E402

NAME_RE = re.compile(r"_(blk|nb)([AER])_rep\d+_s\d+_(\w+?)_one_to_one_b(\d+)_w\d+_i\d+\.txt$")
PAIR_RE = re.compile(r"^pair (\S+) -> (\S+) .*?med_us=([0-9.eE+-]+)")
PROTO = {"A": "порог UCX", "E": "только eager", "R": "только rendezvous"}


def pair_class(src, dst, gps):
	sn, sg = gpu_of(src)
	dn, dg = gpu_of(dst)
	if sn != dn:
		return "между узлами"
	return "внутри узла, один процессор" if sg // gps == dg // gps else "внутри узла, разные процессоры"


def num(x, nd=2):
	return f"{x:.{nd}f}".replace(".", ",")


def main():
	ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	ap.add_argument("job", type=Path, nargs="+")
	ap.add_argument("--gpus-per-socket", type=int, default=2)
	args = ap.parse_args()

	raw = defaultdict(list)   # (класс, вызов, протокол, размер) -> медианы пар
	for job in args.job:
		for f in sorted(job.glob("*.txt")):
			m = NAME_RE.search(f.name)
			if not m:
				continue
			call, proto, size = m.group(1), m.group(2), int(m.group(4))
			for line in f.read_text(encoding="utf-8", errors="replace").splitlines():
				p = PAIR_RE.match(line)
				if p and p.group(1) != p.group(2):
					raw[(pair_class(p.group(1), p.group(2), args.gpus_per_socket), call, proto, size)].append(
						float(p.group(3)))
	if not raw:
		print(f"нет результатов p2pforce в {[str(j) for j in args.job]}", file=sys.stderr)
		return 1
	med = {k: median(v) for k, v in raw.items()}
	classes = sorted({k[0] for k in med})
	sizes = sorted({k[3] for k in med})
	print(f"{', '.join(j.name for j in args.job)}: медиана медиан пар, мкс; nb/blk < 1 — MPI_Isend быстрее\n")
	for cls in classes:
		print(f"### {cls}")
		print("| Размер | " + " | ".join(f"Send / Isend, {PROTO[p]}" for p in "AER")
		      + " | " + " | ".join(f"Isend/Send, {PROTO[p]}" for p in "AER") + " |")
		print("|---" * 7 + "|")
		for s in sizes:
			cells, ratios = [], []
			for p in "AER":
				b, n = med.get((cls, "blk", p, s)), med.get((cls, "nb", p, s))
				cells.append(f"{num(b, 1)} / {num(n, 1)}" if b and n else "—")
				ratios.append(num(n / b) if b and n else "—")
			print(f"| {fmt_bytes(s)} | " + " | ".join(cells) + " | " + " | ".join(ratios) + " |")
		print()
	return 0


if __name__ == "__main__":
	sys.exit(main())
