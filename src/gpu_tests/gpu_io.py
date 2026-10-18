#!/usr/bin/env python3
"""Чтение результатов gpu_tests: общее для gpu_heatmap.py, gpu_plot.py и gpu_netcdf.py.

Формат .txt пишет gpu_benchmark.cpp: шапка (Env:, Mode:, Bytes:, Iters:, Ranks:,
строки «r<k> hostname=...»), затем строки
    pair <src> -> <dst> avg_us=… med_us=… min_us=… max_us=… var_us=… std_us=…
(или только одна величина при --stat) и TotalTimeSec:.
Сырые задержки лежат в raw/<stem>_src<src>_dst<dst>.raw рядом с .txt.
"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Any

METRIC_KEYS = ("avg_us", "med_us", "min_us", "max_us", "var_us", "std_us")
METRIC_FILE_STEM = {k: k[:-3] for k in METRIC_KEYS}
METRIC_CBAR_LABEL = {k: ("мкс²" if k == "var_us" else "мкс") for k in METRIC_KEYS}
METRIC_TITLE = {
	"avg_us": "Среднее арифметическое задержек",
	"med_us": "Медиана задержек",
	"min_us": "Минимум задержек",
	"max_us": "Максимум задержек",
	"var_us": "Выборочная дисперсия задержек",
	"std_us": "Стандартное отклонение задержек",
}

# Теги в именах файлов task.sbatch: <узлы>_<env>_<mode>_b<байт>_w<warmup>_i<iters>.txt
REP_TAG_RE = re.compile(r"(?:^|_)rep(\d+)(?:_|$)", re.I)
ENV_TAG_RE = re.compile(r"(?:^|_)(auto|host)(?:_|$)", re.I)
MODE_TAG_RE = re.compile(
	r"(?:^|_)(cuda_one_to_one|cuda_all_to_all|one_to_one_v2|one_to_one|all_to_all|one_to_all|all_to_one)(?:_|$)", re.I
)
BYTES_TAG_RE = re.compile(r"(?:^|_)b(\d+)(?:_|$)", re.I)
WARMUP_TAG_RE = re.compile(r"(?:^|_)w(\d+)(?:_|$)", re.I)
ITERS_TAG_RE = re.compile(r"(?:^|_)i(\d+)(?:_|$)", re.I)

_PAIR_HEAD_RE = re.compile(r"^pair(?:_(?:mpi|cpu|cuda|gpu))? (\S+) -> (\S+) ")
_NUMBER_RE = re.compile(r"[0-9.eE+-]+")
_RANK_MAP_RE = re.compile(r"^r(\d+)\s+hostname=(\S+)")
_INT_HEADERS = {"Bytes": "bytes", "Warmup": "warmup", "Iters": "iters"}
_STR_HEADERS = {"Env": "env", "Mode": "mode", "Transport": "transport", "Timer": "timer"}


def short_host(hostname: str) -> str:
	return hostname.split(".", 1)[0].strip() or hostname


def rank_hosts_from_meta(meta: dict[str, Any], n: int) -> list[str] | None:
	"""Узел каждого ранга из строк Rank map, None если хотя бы одного нет."""
	hosts_map = meta.get("rank_hosts")
	if not isinstance(hosts_map, dict):
		return None
	hosts = [hosts_map.get(r) for r in range(n)]
	return hosts if all(isinstance(h, str) and h for h in hosts) else None


def axis_labels_by_node(rank_hosts: list[str]) -> list[str]:
	"""Метки «<цифры имени узла>.<номер ранга на узле>», как в gpu_common.cpp."""
	seen: dict[str, int] = {}
	labels: list[str] = []
	for h in map(short_host, rank_hosts):
		local_idx = seen.get(h, 0)
		seen[h] = local_idx + 1
		m = re.search(r"(\d+)$", h)
		labels.append(f"{m.group(1) if m else h}.{local_idx}")
	return labels


def pair_label_to_rank(label: str, meta: dict[str, Any]) -> int | None:
	"""Номер ранга по метке пары («28.0» или просто номер)."""
	if re.fullmatch(r"\d+", label):
		return int(label)
	cache = meta.get("_pair_label_to_rank")
	if not isinstance(cache, dict):
		n = meta.get("ranks")
		rank_hosts = rank_hosts_from_meta(meta, n) if isinstance(n, int) else None
		if not rank_hosts:
			return None
		cache = {lab: i for i, lab in enumerate(axis_labels_by_node(rank_hosts))}
		meta["_pair_label_to_rank"] = cache
	v = cache.get(label)
	return v if isinstance(v, int) else None


def _pair_values(rest: str) -> list[float] | None:
	"""Шесть величин из хвоста строки pair: все шесть по порядку или одна (--stat)."""
	items = [t.split("=", 1) for t in rest.split()]
	if not items or any(len(kv) != 2 or not _NUMBER_RE.fullmatch(kv[1]) for kv in items):
		return None
	keys = tuple(k for k, _ in items)
	if keys != METRIC_KEYS and not (len(keys) == 1 and keys[0] in METRIC_KEYS):
		return None
	vals = [float("nan")] * len(METRIC_KEYS)
	for k, v in items:
		vals[METRIC_KEYS.index(k)] = float(v)
	return vals


def parse_gpu_one_to_one_text(
	text: str,
) -> tuple[dict[str, Any], list[tuple[int, int, list[float]]]]:
	"""Разбор .txt в метаданные и список (src_rank, dst_rank, шесть величин).

	Если строки одной пары повторяются, побеждает последняя."""
	meta: dict[str, Any] = {}
	pairs: list[tuple[int, int, list[float]]] = []
	for raw in text.splitlines():
		line = raw.strip()
		key, sep, value = line.partition(":")
		if sep and key in _STR_HEADERS:
			meta[_STR_HEADERS[key]] = value.strip()
		elif sep and key in _INT_HEADERS and re.fullmatch(r"\s*\d+\s*", value):
			meta[_INT_HEADERS[key]] = int(value)
		elif sep and key == "Ranks" and (m := re.match(r"\s*(\d+)", value)):
			meta["ranks"] = int(m.group(1))
		elif sep and key == "TotalTimeSec" and re.fullmatch(r"\s*[0-9.eE+-]+\s*", value):
			meta["total_elapsed_s"] = float(value)
		elif m := _RANK_MAP_RE.match(line):
			meta.setdefault("rank_hosts", {})[int(m.group(1))] = m.group(2)
		elif m := _PAIR_HEAD_RE.match(line):
			vals = _pair_values(line[m.end():])
			if vals is None:
				continue
			src = pair_label_to_rank(m.group(1), meta)
			dst = pair_label_to_rank(m.group(2), meta)
			if src is not None and dst is not None:
				pairs.append((src, dst, vals))
	return meta, pairs


def generation_timestamp() -> str:
	"""Дата и время построения: дд/мм/гггг чч:мм:сс и смещение от UTC (+03:00)."""
	dt = datetime.now().astimezone()
	base = dt.strftime("%d/%m/%Y %H:%M:%S")
	raw = dt.strftime("%z")
	return f"{base}{raw[0]}{raw[1:3]}:{raw[3:5]}" if raw else base


# --------------------------------------------------------------------------- #
# Сырые задержки                                                              #
# --------------------------------------------------------------------------- #

def _sanitize(label: str) -> str:
	return "".join(c if (c.isalnum() or c in "._-") else "_" for c in label) or "nolabel"


def raw_path(txt_path: Path, src_label: str, dst_label: str) -> Path:
	"""Путь, по которому gpu_common.cpp пишет серию пары."""
	return txt_path.parent / "raw" / f"{txt_path.stem}_src{_sanitize(src_label)}_dst{_sanitize(dst_label)}.raw"


def load_raw(txt_path: Path, src_label: str, dst_label: str) -> list[float]:
	p = raw_path(txt_path, src_label, dst_label)
	if not p.is_file():
		return []
	samples: list[float] = []
	with p.open(encoding="utf-8", errors="replace") as f:
		for line in f:
			s = line.strip()
			if s and not s.startswith("#"):
				try:
					samples.append(float(s))
				except ValueError:
					pass
	return samples
