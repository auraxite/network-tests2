"""Перечисление возможных путей передачи между GPU по снимку collect_env.sh.

python paths/paths_possible.py results/hyp_<job>/env_<узел>.txt [env_<второй узел>.txt]

Путь — последовательность вершин и рёбер графа коммуникационной среды. Граф строится по
кодам связи из `nvidia-smi topo -m`: код задаёт цепочку вершин между двумя устройствами
  NV#  — NVLink напрямую
  PIX  — через один коммутатор PCIe
  PXB  — через несколько коммутаторов PCIe
  PHB  — через контроллер PCIe процессора
  NODE — через разные контроллеры PCIe одного процессора
  SYS  — через межпроцессорный интерфейс между процессорами
Перечисляются все пути GPU → GPU внутри узла и между узлами. Путь состоит из стороны
отправителя, способа передачи и стороны получателя:
  способ: напрямую GPU → GPU; копирование между буферами в ОЗУ; через пару адаптеров,
          связанных одной сетью (внутри узла — в том числе через один адаптер и обратно);
  сторона: без буфера в ОЗУ или через буфер в ОЗУ любого процессора узла.
Для каждого пути указано, что нужно, чтобы он был допустим, и выполнено ли это по снимку.

Это возможные пути, а не выбранный: какой путь прошло сообщение, показывает эксперимент
paths (журнал UCX и счётчики адаптеров и NVLink).
"""
import ipaddress
import re
import sys
from pathlib import Path


def section(text, title):
	"""Текст раздела collect_env.sh: от «===== title» до следующего «=====»."""
	m = re.search(r"^===== " + re.escape(title) + r" =====\n(.*?)(?=^===== |\Z)", text, re.M | re.S)
	# nvidia-smi выделяет заголовок управляющими кодами терминала (ESC[4m … ESC[0m)
	return re.sub(r"\x1b\[[0-9;]*m", "", m.group(1)) if m else ""


def parse_topo(txt):
	"""Матрица кодов связи, привязка GPU к NUMA и имена адаптеров из nvidia-smi topo -m."""
	lines = [l.rstrip() for l in txt.splitlines() if l.strip()]
	# заголовок — строка со столбцами устройств; начинается с пустой ячейки (табуляции или пробелов)
	head = next((l for l in lines if re.search(r"\bGPU0\b", l) and l[:1] in ("\t", " ")), None)
	if head is None:
		return {}, {}, {}
	cols = [c for c in re.split(r"\t+|\s{2,}", head.strip()) if c]
	devs = [c for c in cols if re.match(r"^(GPU\d+|NIC\d+|mlx\d+_\d+)$", c)]
	rel, numa = {}, {}
	for l in lines:
		cells = [c.strip() for c in l.split("\t") if c.strip() != ""] if "\t" in l else l.split()
		if not cells or cells[0] not in devs:
			continue
		name, vals = cells[0], cells[1:]
		for d, v in zip(devs, vals):
			rel[(name, d)] = v
		if name.startswith("GPU") and len(vals) > len(devs) + 1 and vals[len(devs) + 1].isdigit():
			numa[name] = int(vals[len(devs) + 1])
	legend = dict(re.findall(r"^\s*(NIC\d+):\s*(\S+)", txt, re.M))
	return rel, numa, legend


def parse_p2p(txt):
	"""Разрешён ли прямой доступ GPU → GPU (nvidia-smi topo -p2p r): OK или нет."""
	ok = {}
	rows = [l.split() for l in txt.splitlines() if re.match(r"^\s*GPU\d+", l)]
	head = next((l.split() for l in txt.splitlines() if re.search(r"GPU0\s+GPU1", l)), [])
	for r in rows:
		for d, v in zip(head, r[1:]):
			ok[(r[0], d)] = v == "OK"
	return ok


def parse_nics(txt):
	"""Порты адаптеров: состояние и тип сети (ibv_devinfo)."""
	nics, cur = {}, None
	for l in txt.splitlines():
		m = re.match(r"^hca_id:\s*(\S+)", l)
		if m:
			cur = m[1]
			nics[cur] = {"active": False, "layer": "?"}
		if cur and "PORT_ACTIVE" in l:
			nics[cur]["active"] = True
		m = re.search(r"link_layer:\s*(\S+)", l)
		if cur and m:
			nics[cur]["layer"] = m[1]
	return nics


def parse_subnets(t):
	"""Подсети IPv4 каждого адаптера: его интерфейс и VLAN на нём (раздел sysfs и ip -br -4 addr)."""
	netdev = dict(re.findall(r"^nic (\S+) .*?netdev=(\S+) ", t, re.M))
	nets = {}
	for l in section(t, "ip -br -4 addr").splitlines():
		f = l.split()
		for cidr in f[2:]:
			try:
				nets.setdefault(f[0].split("@")[-1], set()).add(str(ipaddress.ip_interface(cidr).network))
			except ValueError:
				pass
	return {nic: nets.get(nd, set()) for nic, nd in netdev.items() if not nd.startswith("path=")}


class Node:
	def __init__(self, path):
		t = Path(path).read_text(encoding="utf-8", errors="replace")
		self.name = Path(path).stem.replace("env_", "")
		self.rel, self.numa, legend = parse_topo(section(t, "nvidia-smi topo -m"))
		self.p2p = parse_p2p(section(t, "nvidia-smi topo -p2p r"))
		self.nicinfo = parse_nics(section(t, "ibv_devinfo"))
		self.subnets = parse_subnets(t)
		self.tls = set(re.findall(r"Transport:\s*(\S+)", section(t, next(
			(k for k in re.findall(r"^===== (.*ucx_info -d) =====", t, re.M)), "ucx_info -d"))))
		self.peermem = bool(re.search(r"nvidia_peermem|nv_peer_mem", section(t, next(
			(k for k in re.findall(r"^===== (bash -c lsmod.*?) =====", t, re.M)), "lsmod"))))
		sockets = re.search(r"Socket\(s\):\s*(\d+)", section(t, "lscpu"))
		self.sockets = int(sockets[1]) if sockets else 2
		devs = sorted({a for a, _ in self.rel})
		self.gpus = [d for d in devs if d.startswith("GPU")]
		# адаптер: имя из легенды (NIC0 → mlx5_0) или сразу имя столбца
		self.nics = [d for d in devs if not d.startswith("GPU")]
		self.nicname = {d: legend.get(d, d) for d in self.nics}

	def socket_of(self, dev):
		"""NUMA-узел устройства: у GPU — из topo, у адаптера — по связи с ближайшим GPU."""
		if dev in self.numa:
			return self.numa[dev]
		for g in self.gpus:
			code = self.rel.get((dev, g), "")
			if code in ("PIX", "PXB", "PHB", "NODE") or code.startswith("NV"):
				return self.socket_of(g)
		for g in self.gpus:
			if self.rel.get((dev, g)) == "SYS":
				return 1 - self.socket_of(g) if self.sockets == 2 else None
		return 0

	def hop(self, a, b):
		"""Цепочка вершин между двумя устройствами узла по коду связи."""
		code = self.rel.get((a, b), "?")
		sa, sb = self.socket_of(a), self.socket_of(b)
		if code.startswith("NV"):
			return [f"={code}=>"], code
		return {
			"PIX": ["коммутатор PCIe"],
			"PXB": ["коммутатор PCIe", "коммутатор PCIe"],
			"PHB": [f"контроллер PCIe CPU{sa}"],
			"NODE": [f"контроллер PCIe CPU{sa}", f"CPU{sa}", f"контроллер PCIe CPU{sa}"],
			"SYS": [f"контроллер PCIe CPU{sa}", f"CPU{sa}", "межпроцессорный интерфейс", f"CPU{sb}",
			        f"контроллер PCIe CPU{sb}"],
		}.get(code, [f"?{code}"]), code

	def to_ram(self, dev, s):
		"""Цепочка от устройства до ОЗУ процессора s."""
		ls = self.socket_of(dev)
		chain = [f"контроллер PCIe CPU{ls}", f"CPU{ls}"]
		if s != ls:
			chain += ["межпроцессорный интерфейс", f"CPU{s}"]
		return chain + [f"ОЗУ{s}"]

	def active_nics(self):
		return [n for n in self.nics if self.nicinfo.get(self.nicname[n], {"active": True})["active"]]


def fmt(chain):
	return " → ".join(chain)


def ram_to(node, dev, s):
	"""Цепочка от ОЗУ процессора s до устройства (обратная к to_ram, без ОЗУ в начале)."""
	return node.to_ram(dev, s)[::-1][1:]


def sides(node, gpu, nic, sender):
	"""Варианты стороны пути между GPU и адаптером: без буфера в ОЗУ (адаптер сам обращается
	к видеопамяти, GPUDirect RDMA) или через буфер в ОЗУ любого процессора узла."""
	out = []
	direct = node.hop(gpu, nic)[0] if sender else node.hop(nic, gpu)[0][::-1]
	out.append(("без ОЗУ", direct, True))
	for s in range(node.sockets):
		if sender:   # GPU → ОЗУ s → адаптер
			chain = node.to_ram(gpu, s) + ram_to(node, nic, s)
		else:        # адаптер → ОЗУ s → GPU
			chain = node.to_ram(nic, s) + ram_to(node, gpu, s)
		out.append((f"через ОЗУ{s}", chain, False))
	return out


def connected(n1, x, n2, y):
	"""Есть ли сеть между адаптером x узла n1 и адаптером y узла n2 (условие связности):
	True, False или None, если по снимку это установить нельзя (нет адресов Ethernet)."""
	if n1 is n2 and x == y:
		return True   # петля через тот же адаптер
	lx = n1.nicinfo.get(n1.nicname[x], {}).get("layer", "?")
	ly = n2.nicinfo.get(n2.nicname[y], {}).get("layer", "?")
	if lx != ly:
		return False
	# Ethernet: адаптеры связаны, если у их интерфейсов есть общая подсеть
	if lx != "Ethernet":
		return True
	if not (n1.subnets and n2.subnets):
		return None
	return bool(n1.subnets.get(n1.nicname[x], set()) & n2.subnets.get(n2.nicname[y], set()))


def via_nics(n1, a, n2, b, prime=""):
	"""Пути через адаптеры: каждая сторона без ОЗУ или через буфер в ОЗУ любого процессора,
	адаптеры — любая пара, связанная одной сетью (внутри узла — в том числе один адаптер)."""
	rows = []
	has_copy = lambda n: ("cuda_copy" in n.tls) if n.tls else None
	for x in n1.active_nics():
		for y in n2.active_nics():
			link = connected(n1, x, n2, y)
			if link is False:
				continue
			nx, ny = n1.nicname[x], n2.nicname[y]
			layer = n1.nicinfo.get(nx, {}).get("layer", "?")
			mid = [nx, f"петля {layer}", nx] if (n1 is n2 and x == y) else [nx, f"сеть {layer}", ny]
			for sd, side_a, gdr_a in sides(n1, a, x, True):
				for rd, side_b, gdr_b in sides(n2, b, y, False):
					need, ok = [], (True if link else None)
					if gdr_a or gdr_b:
						need.append("GPUDirect RDMA (nvidia_peermem)")
						if (gdr_a and not n1.peermem) or (gdr_b and not n2.peermem):
							ok = False
					if not (gdr_a and gdr_b):
						need.append("cuda_copy")
						for n in (n1, n2):
							if has_copy(n) is False:
								ok = False
					rows.append((f"через адаптеры {nx} → {ny}: отправитель {sd}, получатель {rd}",
					             [f"VRAM({a})", a] + side_a + mid + side_b + [f"{b}{prime}", f"VRAM({b}{prime})"],
					             ", ".join(need) + ", сетевой транспорт UCX", ok))
	return rows


def intra(node):
	rows = []
	G = node.gpus
	for i, a in enumerate(G):
		for b in G[i + 1:]:
			code = node.rel.get((a, b), "?")
			p2p_ok = node.p2p.get((a, b))
			need_ipc = "cuda_ipc" in node.tls if node.tls else None
			# a) напрямую GPU → GPU
			if code.startswith("NV"):
				rows.append(("напрямую, NVLink", [f"VRAM({a})", a] + node.hop(a, b)[0] + [b, f"VRAM({b})"],
				             "прямой доступ GPU → GPU, транспорт cuda_ipc", p2p_ok and need_ipc))
				# PCIe между ними в topo скрыт кодом NV#: связь есть, но драйвер ведёт прямой доступ по NVLink
				rows.append(("напрямую, PCIe", [f"VRAM({a})", a, "PCIe (код скрыт NVLink)", b, f"VRAM({b})"],
				             "недопустим: при наличии NVLink драйвер ведёт прямой доступ по NVLink", False))
			else:
				chain = node.hop(a, b)[0]
				rows.append((f"напрямую, PCIe ({code})", [f"VRAM({a})", a] + chain + [b, f"VRAM({b})"],
				             "прямой доступ GPU → GPU, транспорт cuda_ipc", p2p_ok and need_ipc))
			# b) через ОЗУ: буфер отправителя в ОЗУ s, буфер получателя в ОЗУ t;
			# при s ≠ t копирование между буферами идёт через межпроцессорный интерфейс
			for s in range(node.sockets):
				for t in range(node.sockets):
					ra, rb = node.to_ram(a, s), node.to_ram(b, t)[::-1]
					tail = rb[1:] if s == t else [f"CPU{s}", "межпроцессорный интерфейс", f"CPU{t}"] + rb
					rows.append((f"через ОЗУ: буфер отправителя в ОЗУ{s}, получателя в ОЗУ{t}",
					             [f"VRAM({a})", a] + ra + tail + [b, f"VRAM({b})"],
					             "cuda_copy и копирование ОЗУ → ОЗУ (sm, cma)",
					             ("cuda_copy" in node.tls) if node.tls else None))
			# c) через адаптеры внутри узла
			rows += via_nics(node, a, node, b)
	return rows


def inter(n1, n2):
	"""d) Пути от GPU0 первого узла к GPU0 второго."""
	return via_nics(n1, n1.gpus[0], n2, n2.gpus[0], prime="'")


def show(title, rows):
	ok = sum(1 for r in rows if r[3])
	unk = sum(1 for r in rows if r[3] is None)
	print(f"\n=== {title}: {len(rows)} путей, допустимы по снимку {ok}" + (f", неизвестно {unk}" if unk else ""))
	for i, (kind, chain, need, ok) in enumerate(rows, 1):
		mark = {True: "возможен", False: "НЕТ", None: "?"}[ok]
		print(f"{i:>3}. [{mark}] {kind}\n     {fmt(chain)}\n     нужно: {need}")


def main(args):
	n1 = Node(args[0])
	n2 = Node(args[1]) if len(args) > 1 else n1
	print(f"узел {n1.name}: GPU {n1.gpus}, адаптеры {[n1.nicname[n] for n in n1.nics]}, "
	      f"работают {[n1.nicname[n] for n in n1.active_nics()]}, процессоров {n1.sockets}, "
	      f"GPUDirect RDMA: {'да' if n1.peermem else 'нет'}, транспорты UCX: {sorted(n1.tls) or '?'}")
	show(f"внутри узла {n1.name}", intra(n1))
	show(f"между узлами {n1.name} → {n2.name} (GPU0 → GPU0)", inter(n1, n2))


if __name__ == "__main__":
	main(sys.argv[1:])
