#!/usr/bin/env python3
"""Порог rendezvous в одновременных режимах (эксперимент experiments/rndvmodes.conf).

    python3 analysis/rndvmodes.py results/hyp_<задание> [ещё задания ...]

Для каждого режима, класса пар (между узлами | внутри узла) и размера — медиана медиан пар в вариантах
auto (порог UCX) и t65 (UCX_RNDV_THRESH=65) и их отношение t65 / auto: меньше 1 — порог 65 Б быстрее.
"""

from __future__ import annotations

import re
import sys
from collections import defaultdict
from pathlib import Path
from statistics import median

NAME_RE = re.compile(r"_(auto|t65)_rep\d+_s\d+_\w+?_(one_to_one|all_to_all|one_to_all|all_to_one)_b(\d+)_w\d+_i\d+\.txt$")
PAIR_RE = re.compile(r"^pair (\S+) -> (\S+) .*?med_us=([0-9.eE+-]+)")
MODES = ("one_to_one", "all_to_all", "one_to_all", "all_to_one")


def node(label: str) -> str:
	return label.rpartition(".")[0]


def main() -> int:
	jobs = [Path(a) for a in sys.argv[1:]]
	if not jobs:
		print(__doc__)
		return 1
	vals = defaultdict(list)   # (режим, класс, размер, вариант) -> медианы пар
	for f in sorted(f for j in jobs for f in j.glob("*.txt")):
		m = NAME_RE.search(f.name)
		if not m:
			continue
		var, mode, size = m.group(1), m.group(2), int(m.group(3))
		for line in f.read_text(encoding="utf-8", errors="replace").splitlines():
			p = PAIR_RE.match(line)
			if not p or p.group(1) == p.group(2):
				continue
			cls = "между узлами" if node(p.group(1)) != node(p.group(2)) else "внутри узла"
			vals[(mode, cls, size, var)].append(float(p.group(3)))
	if not vals:
		print("нет файлов *_auto|t65_rep*_<режим>_b*_w*_i*.txt", file=sys.stderr)
		return 1
	sizes = sorted({k[2] for k in vals})
	for cls in ("между узлами", "внутри узла"):
		print(f"\n## {cls}: медиана по парам, мкс — auto / t65 (t65 ÷ auto)\n")
		print("| Режим | " + " | ".join(f"{s} Б" for s in sizes) + " |")
		print("|---|" + "---|" * len(sizes))
		for mode in MODES:
			cells = []
			for s in sizes:
				a, t = vals.get((mode, cls, s, "auto")), vals.get((mode, cls, s, "t65"))
				if a and t:
					ma, mt = median(a), median(t)
					cells.append(f"{ma:.1f} / {mt:.1f} ({mt / ma:.2f})".replace(".", ","))
				else:
					cells.append("—")
			print(f"| {mode} | " + " | ".join(cells) + " |")
	print("\nМеньше 1 в скобках — порог 65 Б быстрее порога UCX.")
	return 0


if __name__ == "__main__":
	sys.exit(main())
