"""Разбор эксперимента paths (experiments/paths.conf): каким путём фактически шли пары GPU.

python parse_paths.py results/hyp_<job>

Пишет в каталог задания:
  paths_nodes.csv — по запуску и узлу: байты по NVLink, через сетевые адаптеры и по PCIe GPU,
                    сколько ожидалось от межузловых и внутриузловых пар узла; какая группа путей
                    подтверждена для пар внутри узла и между узлами и какие пути из enum_paths.py
                    в неё входят (если рядом лежит снимок env_<узел>.txt);
  paths_ucx.csv   — по запуску и рангу: транспорты из строк ep_cfg журнала UCX и строки
                    таблиц UCX_PROTO_INFO (внутри узла / между узлами, размеры, протокол, устройства).

Как различаются группы путей внутри узла: из измеренного вычитается то, что дают межузловые
пары, остаток сравнивается с объёмом внутриузловых пар:
  NVLink         — растёт счётчик NVLink;
  через адаптер  — адаптер отправил больше, чем межузловые пары;
  по PCIe GPU    — NVLink и адаптер не выросли (трафик PCIe — подтверждение): напрямую GPU → GPU (cuda_ipc)
                   или через ОЗУ (cuda_copy и sm/cma).
Между узлами счётчики не отличают GPUDirect RDMA от передачи через ОЗУ: байты в обоих
случаях проходят PCIe GPU и адаптер. Это различает только таблица UCX_PROTO_INFO (блок proto).
Внутренние вершины (какой коммутатор PCIe, ОЗУ какого процессора) счётчиками не видны.

Ограничение: журнал UCX не называет собеседника, поэтому пары различаются только как
«внутри узла» и «между узлами». Для двух GPU на узел этого достаточно.
"""
import csv
import re
import sys
from collections import defaultdict
from pathlib import Path

UNITS = {"B": 1, "KiB": 1024, "MiB": 1024 ** 2, "GiB": 1024 ** 3}
SIZE = {"": 1, "K": 1024, "M": 1024 ** 2, "G": 1024 ** 3}
SHARE = 0.5   # доля внутриузлового объёма, начиная с которой считаем путь подтверждённым


def read_txt(path):
	"""Ранги (узел, GPU) и пары из файла результатов программы."""
	ranks, pairs = {}, []
	for line in open(path, encoding="utf-8", errors="replace"):
		m = re.match(r"\s+r(\d+) hostname=(\S+) local_gpu=(\d+)", line)
		if m:
			host = m[2].split(".")[0]
			ranks[int(m[1])] = (host, f"{re.sub(r'.*-', '', host)}.{m[3]}")
		m = re.match(r"pair (\S+) -> (\S+) ", line)
		if m and m[1] != m[2]:
			pairs.append((m[1], m[2]))
	return ranks, pairs


def read_counters(path):
	"""Байты, отправленные адаптерами, и байты Tx по NVLink для одного узла."""
	nic, nvl = 0, 0
	if not path.exists():
		return None, None
	for line in open(path, encoding="utf-8", errors="replace"):
		m = re.match(r"nic \S+ \S+ xmit_words (\d+)", line)
		if m:
			nic += 4 * int(m[1])
		m = re.search(r"Data Tx:\s*([\d.]+)\s*(B|KiB|MiB|GiB)", line)
		if m:
			nvl += float(m[1]) * UNITS[m[2]]
	return nic, nvl


def read_pcie(path):
	"""Отсчёты nvidia-smi dmon -s t с временем эпохи: [(t, gpu, rx МБ/с, tx МБ/с)]."""
	out = []
	if not path.exists():
		return out
	for line in open(path, encoding="utf-8", errors="replace"):
		f = line.split()
		if len(f) >= 4 and not f[1].startswith("#"):
			try:
				out.append((float(f[0]), int(f[1]), float(f[2]), float(f[3])))
			except ValueError:
				pass
	return out


def pcie_tx(samples, t0, t1):
	"""Байты, переданные всеми GPU узла по PCIe за [t0, t1]: МБ/с × 1с на отсчёт."""
	return sum(tx * 1e6 for t, _, _, tx in samples if t0 <= t <= t1 + 1) if samples else None


def read_ucx(path):
	"""Конфигурации соединений и таблицы UCX_PROTO_INFO из журнала одного процесса.

	UCX 1.15 пишет в журнал уровня info строку на конфигурацию соединения:
	  «0x… intra-node cfg#2 tag(sysv/memory cma/memory rc_mlx5/mlx5_0:1 cuda_ipc/cuda)»
	(в других версиях — «ep_cfg[N]: tag(...)»). Каждая строка журнала начинается
	с «[время] [узел:pid:поток]», его снимаем.
	"""
	cfg, rows = [], []
	if not path.exists():
		return cfg, rows
	title = None
	for raw in open(path, encoding="utf-8", errors="replace"):
		line = re.sub(r"^\[[^\]]*\]\s*\[[^\]]*\]\s*", "", raw)
		m = re.search(r"\b((?:self|intra-node|inter-node) cfg#\d+) (tag\(.*\))", line) or \
			re.search(r"(ep_cfg\[\d+\]):\s*(.*)", line)
		if m and not line.lstrip().startswith("|"):
			cfg.append((m[1], m[2].strip()))
			continue
		if not line.lstrip().startswith("|"):
			continue
		cells = [c.strip() for c in line.strip().strip("|").split("|")]
		if any(k in line for k in ("intra-node", "inter-node", "self cfg")):
			title = " | ".join(c for c in cells if c)
		elif title and len(cells) >= 2 and re.match(r"^[\d.]+\w*\.\.", cells[0].replace(" ", "")):
			rows.append((title, cells[0], cells[1], cells[2] if len(cells) > 2 else ""))
	return cfg, rows


def in_range(sizes, B):
	"""Попадает ли размер B в диапазон таблицы UCX вида «8K..inf»."""
	m = re.match(r"^\s*([\d.]+)([KMG]?)\s*\.\.\s*(inf|[\d.]+)([KMG]?)", sizes.replace(" ", ""))
	if not m:
		return False
	lo = float(m[1]) * SIZE[m[2]]
	hi = float("inf") if m[3] == "inf" else float(m[3]) * SIZE[m[4]]
	return lo <= B <= hi


def classify(proto, lanes):
	"""Путь по строке таблицы UCX: протокол и транспорты.

	copy-in — отправитель копирует данные в промежуточный буфер в ОЗУ, copy-out — получатель
	копирует из такого буфера; zero-copy — устройство работает с памятью GPU само.
	"""
	p = proto.replace("(?)", "").strip()
	if "cuda_ipc" in lanes:
		return "напрямую GPU → GPU"
	if "copy-in" in p:
		return "через ОЗУ"
	if "zero-copy copy-out" in p:
		return "отправитель без ОЗУ, получатель через ОЗУ"
	if "zero-copy" in p:
		return "без ОЗУ"
	return p


def proto_verdict(rows, kind, B):
	"""Что таблицы UCX_PROTO_INFO говорят о передаче тегированных сообщений из памяти GPU
	для пар kind («intra-node» или «inter-node») и размера B. Вариантов вызова у UCX несколько
	(обычный, fast-completion, multi) — перечисляются все различающиеся."""
	sel = []
	for t, s, p, l in rows:
		if kind in t and "tagged message" in t and "from cuda" in t and in_range(s, B):
			item = f"{classify(p, l)} ({p.replace('(?)', '').strip()}, {l})"
			if item not in sel:
				sel.append(item)
	return "; ".join(sel)


def candidates(job, host, group):
	"""Пути из enum_paths.py, входящие в подтверждённую группу (нужен снимок env_<узел>.txt)."""
	env = job / f"env_{host}.txt"
	if not env.exists() or not group:
		return ""
	sys.path.insert(0, str(Path(__file__).resolve().parent))
	import enum_paths
	keys = {"NVLink": ("NVLink",), "через адаптер": ("через адаптер",),
	        "напрямую по PCIe": ("PCIe напрямую",), "через ОЗУ": ("через ОЗУ",),
	        "по PCIe GPU": ("PCIe напрямую", "через ОЗУ")}[group]
	rows = enum_paths.intra(enum_paths.Node(env))
	return "; ".join(k for k, _, _, ok in rows if k.startswith(keys) and ok is not False)


def main(job):
	job = Path(job)
	runs = list(csv.DictReader(open(job / "runs.csv", encoding="utf-8")))
	pcie = {}
	nodes_out, ucx_out = [], []
	for r in runs:
		txt = next(job.glob(f"*_s{r['seed']}_*.txt"), None)
		if txt is None:
			print("нет файла результатов для", r["seed"])
			continue
		ranks, pairs = read_txt(txt)
		host_of = {lab: host for host, lab in ranks.values()}
		msgs = int(r["iters"]) + int(r["warmup"])
		B = int(r["bytes"])
		exp_inter, exp_intra = defaultdict(int), defaultdict(int)
		for s, d in pairs:
			hs, hd = host_of.get(s), host_of.get(d)
			(exp_intra if hs == hd else exp_inter)[hs] += msgs * B

		# журнал UCX: строки по рангам и объединение по узлу
		node_cfg, node_rows = defaultdict(list), defaultdict(list)
		for rank, (host, lab) in sorted(ranks.items()):
			for ext in ("out", "err"):
				cfg, rows = read_ucx(job / "ucx" / f"{txt.stem}.r{rank}.{ext}")
				node_cfg[host] += [l for _, l in cfg]
				node_rows[host] += rows
				for label, lanes in cfg:
					ucx_out.append(dict(index=r["index"], block=r["block"], env=r["env"], bytes=B, rank=rank,
					                    gpu=lab, source="cfg", table=label, sizes="", protocol="",
					                    lanes=lanes))
				for title, sizes, proto, lanes in rows:
					ucx_out.append(dict(index=r["index"], block=r["block"], env=r["env"], bytes=B, rank=rank,
					                    gpu=lab, source="proto_info", table=title, sizes=sizes, protocol=proto,
					                    lanes=lanes))

		for host in sorted({h for h, _ in ranks.values()}):
			n0, v0 = read_counters(job / "counters" / f"{host}_{r['index']}_before.txt")
			n1, v1 = read_counters(job / "counters" / f"{host}_{r['index']}_after.txt")
			nic = n1 - n0 if None not in (n0, n1) else None
			nvl = v1 - v0 if None not in (v0, v1) else None
			if host not in pcie:
				pcie[host] = read_pcie(job / "counters" / f"pcie_{host}.txt")
			pci = pcie_tx(pcie[host], float(r["start"]), float(r["end"]))
			ei, ea = exp_inter[host], exp_intra[host]
			# доли внутриузлового объёма, которые пришлись на NVLink, адаптер и PCIe GPU
			f_nvl = nvl / ea if nvl is not None and ea else None
			f_nic = (nic - ei) / ea if nic is not None and ea else None
			f_pci = (pci - ei) / ea if pci is not None and ea else None
			# решают точные накопительные счётчики NVLink и адаптера; трафик PCIe (мгновенный
			# отсчёт раз в секунду) слишком шумный и в выводе не участвует
			if f_nvl is not None and f_nvl > SHARE:
				group = "NVLink"
			elif f_nic is not None and f_nic > SHARE:
				group = "через адаптер"
			elif f_nvl is not None and f_nic is not None:
				# ни NVLink, ни адаптер сверх межузловых пар — значит, по PCIe GPU
				group = "по PCIe GPU"
			else:
				group = ""
			intra_ucx = proto_verdict(node_rows[host], "intra-node", B)
			inter_ucx = proto_verdict(node_rows[host], "inter-node", B)
			nodes_out.append(dict(
				index=r["index"], block=r["block"], env=r["env"], bytes=B, node=host,
				expected_intra=ea, expected_inter=ei, nvlink_tx=nvl, nic_tx=nic, pcie_tx=pci,
				share_nvlink=f_nvl, share_nic=f_nic, share_pcie=f_pci,
				intra_by_counters=group or "не определить", intra_by_ucx=intra_ucx,
				intra_candidates=candidates(job, host, group),
				inter_by_ucx=inter_ucx or ("через ОЗУ: в среде host программа сама копирует данные в ОЗУ"
				                           if r["env"] == "host" else
				                           "в журнале UCX узла нет таблицы для этого размера" if r["block"] == "proto" else
				                           "нужен блок proto: счётчики GPUDirect RDMA и ОЗУ не различают")))

	for name, rows in (("paths_nodes.csv", nodes_out), ("paths_ucx.csv", ucx_out)):
		if rows:
			with open(job / name, "w", newline="", encoding="utf-8") as fh:
				w = csv.DictWriter(fh, fieldnames=list(rows[0]))
				w.writeheader()
				w.writerows(rows)
			print("saved", job / name, len(rows), "строк")
		else:
			print("пусто:", name)
	for n in nodes_out:
		print(f"{n['index']:>2} {n['block']:5} {n['env']:4} {n['bytes']:>8} {n['node']:9} "
		      f"внутри узла: {n['intra_by_counters']}" + (f" | UCX: {n['intra_by_ucx']}" if n["intra_by_ucx"] else "")
		      + f" | между узлами: {n['inter_by_ucx']}")
		if n["intra_candidates"]:
			print(f"   пути из графа: {n['intra_candidates']}")


if __name__ == "__main__":
	main(sys.argv[1] if len(sys.argv) > 1 else ".")
