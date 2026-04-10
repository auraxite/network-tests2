#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path


def process_file(path: Path) -> tuple[Path, int]:
	values = []
	for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
		s = line.strip()
		if not s or s.startswith("#"):
			continue
		values.append(float(s))
	values.sort()
	dst = path.with_name("sorted_" + path.name)
	with dst.open("w", encoding="utf-8") as f:
		for v in values:
			f.write(f"{v:.3f}\n")
	return dst, len(values)


def main() -> int:
	parser = argparse.ArgumentParser(
		description=(
			"Sort raw sample files (.raw) numerically into new files only.\n"
			"Output file names are prefixed with 'sorted_'."
		)
	)
	parser.add_argument("input", type=Path, help="Input .raw file or directory")
	parser.add_argument(
		"--recursive",
		action="store_true",
		help="When input is a directory, include subdirectories",
	)
	args = parser.parse_args()

	if args.input.is_file():
		files = [args.input]
	elif args.input.is_dir():
		pattern = "**/*.raw" if args.recursive else "*.raw"
		files = sorted(p for p in args.input.glob(pattern) if p.is_file())
	else:
		files = []

	if not files:
		print(f"sort_raw_samples: no files found for {args.input}")
		return 1

	ok = 0
	for path in files:
		try:
			dst, n = process_file(path)
			print(f"{path} -> {dst} ({n} values)")
			ok += 1
		except Exception as e:
			print(f"{path}: error: {e}")

	return 0 if ok == len(files) else 2


if __name__ == "__main__":
	raise SystemExit(main())

