#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path


def parse_values(path: Path) -> list[float]:
	values: list[float] = []
	for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
		text = line.strip()
		if not text:
			continue
		values.append(float(text))
	return values


def out_path_for(src: Path) -> Path:
	return src.with_name("sorted_" + src.name)


def process_file(path: Path) -> tuple[Path, int]:
	values = parse_values(path)
	values.sort()
	dst = out_path_for(path)
	with dst.open("w", encoding="utf-8") as f:
		for v in values:
			f.write(f"{v:.3f}\n")
	return dst, len(values)


def collect_files(path: Path, recursive: bool) -> list[Path]:
	if path.is_file():
		return [path]
	if path.is_dir():
		pattern = "**/*.raw" if recursive else "*.raw"
		return sorted(p for p in path.glob(pattern) if p.is_file())
	return []


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

	files = collect_files(args.input, args.recursive)
	if not files:
		print(f"sort_raw_samples: no files found for {args.input}")
		return 1

	ok = 0
	for path in files:
		try:
			dst, n = process_file(path)
			print(f"{path} -> {dst} ({n} values)")
			ok += 1
		except Exception as e:  # noqa: BLE001
			print(f"{path}: error: {e}")

	return 0 if ok == len(files) else 2


if __name__ == "__main__":
	raise SystemExit(main())

