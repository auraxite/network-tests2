#!/usr/bin/env python3
from __future__ import annotations

"""Convert gpu_benchmark .txt (+.raw) output files to NetCDF4.

Single file:   gpu_netcdf.py run.txt [-o run.nc]
Directory:     gpu_netcdf.py results/ [-o merged.nc]
               → merged .nc with a `bytes` dimension (one slice per .txt).

Layout in the resulting .nc:
  Dimensions:  src_rank × dst_rank form an N×N grid — directly usable as a
               heatmap in ncview.  `bytes` is the record dimension (animation
               axis).  `iter` holds raw per-iteration latencies.
  Variables:   avg_us / med_us / min_us / max_us / var_us / std_us
               → (bytes, src_rank, dst_rank)  [or (src_rank, dst_rank) for one file]
               latency_us → (bytes, src_rank, dst_rank, iter)
               rank_label → (rank, label_len)  S1 (fixed UTF-8 bytes; ncview-friendly)

Raw samples are read from a raw/ subdirectory next to each .txt file,
using filenames produced by gpu_common.cpp:
    raw/{base}_src{src_label}_dst{dst_label}.raw
"""

import argparse
import math
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import netCDF4 as nc4
import numpy as np

import gpu_heatmap as gh

# Strips per-run sweep tags from a stem to get the logical "series" key:
# cn_4_43_rep1_auto_one_to_one_b1000_w5_i2000 → cn_4_43_rep1_auto_one_to_one
_SERIES_TAG_STRIP_RE = re.compile(r"_(?:b|w|i)\d+", re.IGNORECASE)


def _series_key(stem: str) -> str:
    return _SERIES_TAG_STRIP_RE.sub("", stem)


def _group_by_series(paths: list[Path]) -> dict[str, list[Path]]:
    groups: dict[str, list[Path]] = defaultdict(list)
    for p in paths:
        groups[_series_key(p.stem)].append(p)
    return dict(groups)


def _shared_int_meta(runs: list["RunData"], key: str) -> int:
    values = {
        int(v)
        for r in runs
        if isinstance((v := r.meta.get(key)), int)
    }
    if not values:
        return 0
    if len(values) == 1:
        return next(iter(values))
    return -1


def _meta_value_list(runs: list["RunData"], key: str) -> str:
    values = sorted(
        {
            int(v)
            for r in runs
            if isinstance((v := r.meta.get(key)), int)
        }
    )
    return ",".join(str(v) for v in values)

# --------------------------------------------------------------------------- #
# Raw file helpers                                                             #
# --------------------------------------------------------------------------- #

def _sanitize(label: str) -> str:
    return "".join(c if (c.isalnum() or c in "._-") else "_" for c in label) or "nolabel"


def _raw_path(txt_path: Path, src_label: str, dst_label: str) -> Path:
    raw_dir = txt_path.parent / "raw"
    return raw_dir / f"{txt_path.stem}_src{_sanitize(src_label)}_dst{_sanitize(dst_label)}.raw"


def _load_raw(txt_path: Path, src_label: str, dst_label: str) -> list[float]:
    p = _raw_path(txt_path, src_label, dst_label)
    if not p.is_file():
        return []
    samples: list[float] = []
    with p.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            try:
                samples.append(float(s))
            except ValueError:
                pass
    return samples


# --------------------------------------------------------------------------- #
# Parse one .txt file into a structured RunData                               #
# --------------------------------------------------------------------------- #

_STAT_KEYS = ("avg_us", "med_us", "min_us", "max_us", "var_us", "std_us")


class RunData:
    """Everything from one .txt + its raw/ files."""

    def __init__(self, txt_path: Path) -> None:
        self.txt_path = txt_path
        text = txt_path.read_text(encoding="utf-8", errors="replace")
        self.meta, self._pairs = gh.parse_gpu_one_to_one_text(text)

        # Rank labels: prefer hostname-based labels from the meta dict
        n = self.meta.get("ranks", 0)
        rank_hosts = gh.rank_hosts_from_meta(self.meta, n) if isinstance(n, int) else None
        if rank_hosts:
            self.rank_labels: list[str] = gh.axis_labels_by_node(rank_hosts)
        else:
            max_rank = max((max(s, d) for s, d, _ in self._pairs), default=-1)
            self.rank_labels = [str(r) for r in range(max_rank + 1)]

        self.n_ranks = len(self.rank_labels)

        # nbytes: from header or filename tag
        nbytes = self.meta.get("bytes")
        if not isinstance(nbytes, int) or nbytes <= 0:
            m = gh.BYTES_TAG_RE.search(txt_path.stem)
            nbytes = int(m.group(1)) if m else 0
        self.nbytes: int = nbytes

        # stat matrix: shape (n_ranks, n_ranks, 6) filled with NaN
        self.stat = np.full((self.n_ranks, self.n_ranks, len(_STAT_KEYS)), np.nan)
        for src_r, dst_r, vals in self._pairs:
            if 0 <= src_r < self.n_ranks and 0 <= dst_r < self.n_ranks:
                for j, v in enumerate(vals[: len(_STAT_KEYS)]):
                    self.stat[src_r, dst_r, j] = v

        # raw samples: dict (src_r, dst_r) → list[float]
        self.raw: dict[tuple[int, int], list[float]] = {}
        for src_r, dst_r, _ in self._pairs:
            if 0 <= src_r < self.n_ranks and 0 <= dst_r < self.n_ranks:
                sl = self.rank_labels[src_r]
                dl = self.rank_labels[dst_r]
                samples = _load_raw(txt_path, sl, dl)
                if samples:
                    self.raw[(src_r, dst_r)] = samples

    def max_iter(self) -> int:
        return max((len(s) for s in self.raw.values()), default=0)


# --------------------------------------------------------------------------- #
# Write helpers                                                               #
# --------------------------------------------------------------------------- #

def _write_common_coords(ds: nc4.Dataset, rank_labels: list[str]) -> None:
    """Store rank labels as a global attribute — ncview crashes on S1 char variables."""
    if rank_labels:
        ds.rank_labels = ",".join(rank_labels)


def _write_stat_vars(
    ds: nc4.Dataset,
    dims: tuple[str, ...],
) -> dict[str, nc4.Variable]:
    units_map = {"var_us": "us^2"}
    svars: dict[str, nc4.Variable] = {}
    for sn in _STAT_KEYS:
        v = ds.createVariable(sn, "f8", dims, fill_value=np.nan, zlib=True, complevel=4)
        v.units = units_map.get(sn, "us")
        v.long_name = gh.METRIC_TITLE.get(sn, sn)
        svars[sn] = v
    return svars


def _write_raw_var(
    ds: nc4.Dataset,
    dims: tuple[str, ...],
) -> nc4.Variable:
    v = ds.createVariable(
        "latency_us", "f8", dims, fill_value=np.nan, zlib=True, complevel=4
    )
    v.units = "us"
    v.long_name = "raw latency per iteration"
    return v


def _set_global_attrs(ds: nc4.Dataset, meta: dict[str, Any], source: str) -> None:
    ds.Conventions = "CF-1.8"
    ds.env = str(meta.get("env", ""))
    ds.mode = str(meta.get("mode", ""))
    ds.transport = str(meta.get("transport", ""))
    ds.timer = str(meta.get("timer", ""))
    ds.bytes = int(meta.get("bytes", 0)) if isinstance(meta.get("bytes"), int) else 0
    ds.warmup = int(meta.get("warmup", 0)) if isinstance(meta.get("warmup"), int) else 0
    ds.iters = int(meta.get("iters", 0)) if isinstance(meta.get("iters"), int) else 0
    ds.ranks = int(meta.get("ranks", 0)) if isinstance(meta.get("ranks"), int) else 0
    if "total_elapsed_s" in meta:
        ds.total_elapsed_s = float(meta["total_elapsed_s"])
    ds.source = source
    ds.created = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _set_merged_global_attrs(ds: nc4.Dataset, runs: list["RunData"], source: str) -> None:
    _set_global_attrs(ds, runs[0].meta, source)
    ds.bytes = 0
    ds.warmup = _shared_int_meta(runs, "warmup")
    ds.iters = _shared_int_meta(runs, "iters")
    ds.bytes_values = ",".join(str(r.nbytes) for r in runs)
    ds.warmup_values = _meta_value_list(runs, "warmup")
    ds.iters_values = _meta_value_list(runs, "iters")


# --------------------------------------------------------------------------- #
# Single-file writer                                                          #
# --------------------------------------------------------------------------- #

def write_single_nc(txt_path: Path, nc_path: Path) -> None:
    run = RunData(txt_path)
    n = run.n_ranks
    n_iter = run.max_iter()

    with nc4.Dataset(str(nc_path), "w", format="NETCDF4") as ds:
        ds.createDimension("src_rank", n)
        ds.createDimension("dst_rank", n)
        if n_iter > 0:
            ds.createDimension("iter", n_iter)

        _write_common_coords(ds, run.rank_labels)

        # stat variables: (src_rank, dst_rank)
        stat_dims = ("src_rank", "dst_rank")
        svars = _write_stat_vars(ds, stat_dims)
        for j, sn in enumerate(_STAT_KEYS):
            svars[sn][:, :] = run.stat[:, :, j]

        _set_global_attrs(ds, run.meta, str(txt_path.resolve()))

    # raw latency → separate file so ncview var list stays clean
    if n_iter > 0:
        raw_path = nc_path.with_name(nc_path.stem + "_raw" + nc_path.suffix)
        raw_arr = np.full((n, n, n_iter), np.nan)
        for (sr, dr), samples in run.raw.items():
            raw_arr[sr, dr, : len(samples)] = samples
        with nc4.Dataset(str(raw_path), "w", format="NETCDF4") as ds:
            ds.createDimension("iter", n_iter)
            ds.createDimension("src_rank", n)
            ds.createDimension("dst_rank", n)
            _write_common_coords(ds, run.rank_labels)
            # dim order: (iter, src_rank, dst_rank) → ncview defaults to src×dst heatmap
            v_raw = _write_raw_var(ds, ("iter", "src_rank", "dst_rank"))
            v_raw[:, :, :] = raw_arr.transpose(2, 0, 1)
            _set_global_attrs(ds, run.meta, str(txt_path.resolve()))
        print(raw_path)


# --------------------------------------------------------------------------- #
# Multi-file (bytes sweep) writer                                             #
# --------------------------------------------------------------------------- #

def write_merged_nc(txt_paths: list[Path], nc_path: Path) -> None:
    runs: list[RunData] = []
    for p in txt_paths:
        r = RunData(p)
        if r.nbytes <= 0:
            print(f"gpu_netcdf: [{p.name}] no Bytes header and no _bNNN_ tag — skip", file=sys.stderr)
            continue
        runs.append(r)

    if not runs:
        print("gpu_netcdf: no usable .txt files.", file=sys.stderr)
        sys.exit(1)

    runs.sort(key=lambda r: r.nbytes)

    # Use rank labels from the first run; warn if ranks differ
    rank_labels = runs[0].rank_labels
    n = len(rank_labels)
    for r in runs[1:]:
        if r.n_ranks != n or r.rank_labels != rank_labels:
            print(
                f"gpu_netcdf: [{r.txt_path.name}] rank layout differs from first file "
                f"({r.n_ranks} vs {n}); proceeding with first layout.",
                file=sys.stderr,
            )

    n_bytes = len(runs)
    n_iter = max((r.max_iter() for r in runs), default=0)
    bytes_vals = np.array([r.nbytes for r in runs], dtype=np.int64)

    with nc4.Dataset(str(nc_path), "w", format="NETCDF4") as ds:
        # bytes is the "record" dimension — ncview will animate over it
        ds.createDimension("bytes", n_bytes)
        ds.createDimension("src_rank", n)
        ds.createDimension("dst_rank", n)
        if n_iter > 0:
            ds.createDimension("iter", n_iter)

        # coordinate variable: same name as dimension → ncview treats it as axis, not var button
        v_bytes = ds.createVariable("bytes", "i8", ("bytes",))
        v_bytes.long_name = "message size"
        v_bytes.units = "bytes"
        v_bytes[:] = bytes_vals

        _write_common_coords(ds, rank_labels)

        # stat variables: (bytes, src_rank, dst_rank)
        stat_dims = ("bytes", "src_rank", "dst_rank")
        svars = _write_stat_vars(ds, stat_dims)

        stat_data = {sn: np.full((n_bytes, n, n), np.nan) for sn in _STAT_KEYS}
        for bi, run in enumerate(runs):
            nr = min(run.n_ranks, n)
            for j, sn in enumerate(_STAT_KEYS):
                stat_data[sn][bi, :nr, :nr] = run.stat[:nr, :nr, j]

        for sn in _STAT_KEYS:
            svars[sn][:, :, :] = stat_data[sn]

        _set_merged_global_attrs(ds, runs, str(nc_path.parent.resolve()))
        ds.bytes_count = n_bytes
        ds.source_files = " ".join(r.txt_path.name for r in runs)

    # raw latency → separate file so ncview var list stays clean
    if n_iter > 0:
        raw_path = nc_path.with_name(nc_path.stem + "_raw" + nc_path.suffix)
        raw_arr = np.full((n_bytes, n, n, n_iter), np.nan)
        for bi, run in enumerate(runs):
            nr = min(run.n_ranks, n)
            for (sr, dr), samples in run.raw.items():
                if sr < nr and dr < nr:
                    raw_arr[bi, sr, dr, : len(samples)] = samples
        with nc4.Dataset(str(raw_path), "w", format="NETCDF4") as ds:
            ds.createDimension("bytes", n_bytes)
            ds.createDimension("iter", n_iter)
            ds.createDimension("src_rank", n)
            ds.createDimension("dst_rank", n)
            v_bytes2 = ds.createVariable("bytes", "i8", ("bytes",))
            v_bytes2.long_name = "message size"
            v_bytes2.units = "bytes"
            v_bytes2[:] = bytes_vals
            _write_common_coords(ds, rank_labels)
            # dim order: (bytes, iter, src_rank, dst_rank) → ncview defaults to src×dst heatmap
            v_raw = _write_raw_var(ds, ("bytes", "iter", "src_rank", "dst_rank"))
            v_raw[:, :, :, :] = raw_arr.transpose(0, 3, 1, 2)
            _set_merged_global_attrs(ds, runs, str(nc_path.parent.resolve()))
        print(raw_path)


# --------------------------------------------------------------------------- #
# CLI                                                                         #
# --------------------------------------------------------------------------- #

def main() -> int:
    p = argparse.ArgumentParser(
        description="Convert gpu_benchmark .txt (+.raw) to NetCDF4.",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=(
            "Examples:\n"
            "  gpu_netcdf.py run_b4096.txt              → run_b4096.nc\n"
            "  gpu_netcdf.py run_b4096.txt -o out.nc    → out.nc\n"
            "  gpu_netcdf.py results/                   → results/cn_4_43_rep1_auto_one_to_one.nc\n"
            "                                              results/cn_4_43_rep1_host_one_to_one.nc\n"
            "  gpu_netcdf.py results/ -o nc/            → nc/{series}.nc  (выходная папка)\n"
            "  ncview results/cn_4_43_rep1_auto_one_to_one.nc\n"
        ),
    )
    p.add_argument("input", help=".txt file or directory with .txt files")
    p.add_argument(
        "-o", "--out", dest="out", type=Path,
        help="for single file: output .nc path; for directory: output directory",
    )
    args = p.parse_args()

    path = Path(args.input)
    if not path.exists():
        print(f"gpu_netcdf: not found: {path}", file=sys.stderr)
        return 1

    if path.is_file():
        nc_path = args.out or path.with_suffix(".nc")
        write_single_nc(path, nc_path)
        print(nc_path)
        return 0

    if path.is_dir():
        txts = sorted(path.glob("*.txt"))
        if not txts:
            print(f"gpu_netcdf: no *.txt files in {path}", file=sys.stderr)
            return 1

        # Group by logical series (drop _bNNN_, _wNNN_, _iNNN_); auto and host stay separate.
        groups = _group_by_series(txts)
        out_dir = args.out if args.out else path
        out_dir.mkdir(parents=True, exist_ok=True)

        for key, group_txts in sorted(groups.items()):
            nc_path = out_dir / f"{key}.nc"
            if len(group_txts) == 1:
                write_single_nc(group_txts[0], nc_path)
            else:
                write_merged_nc(group_txts, nc_path)
            print(nc_path)
        return 0

    print(f"gpu_netcdf: not a file or directory: {path}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
