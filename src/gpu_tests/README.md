# GPU topology: collect + render

- `gpu_info_collect.py` — только стандартная библиотека, без pip.
- `gpu_info_render.py` — нужны **matplotlib** и **numpy**.

## Зависимости

### Вариант A — Pipenv (из корня `network-tests2`)

```bash
cd /path/to/network-tests2
pip install pipenv          # один раз, в user-site или через pipx
pipenv install
pipenv run python src/gpu_tests/gpu_info_render.py output/gpu_snapshot.json --out-dir output
```

Либо зайти в shell окружения:

```bash
pipenv shell
python src/gpu_tests/gpu_info_render.py output/gpu_snapshot.json --out-dir output
```

### Вариант B — venv + pip

```bash
cd /path/to/network-tests2
python3 -m venv .venv
source .venv/bin/activate
pip install -r src/gpu_tests/requirements.txt
```

### Вариант C — пакеты ОС (без PyPI)

Если нет сети / DNS к PyPI:

```bash
sudo apt install python3-matplotlib python3-numpy
python3 src/gpu_tests/gpu_info_render.py ...
```

---

## Ошибка: `Temporary failure in name resolution` / `No matching distribution found for numpy`

Это **не** «неподходящая версия numpy», а **нет доступа к PyPI** (DNS не резолвит `pypi.org`, или нет интернета). Pipenv при `pipenv install` сначала строит `Pipfile.lock`, для этого тоже нужен запрос к сети — без него список версий пустой (`from versions: none`).

**Что сделать:**

1. Проверить сеть и DNS, например: `ping -c1 pypi.org` или `getent hosts pypi.org`.
2. Починить Wi‑Fi / VPN / `/etc/resolv.conf` (на некоторых системах помогает `systemd-resolved`).
3. Пока PyPI недоступен — использовать **вариант C** (`apt install python3-matplotlib python3-numpy`) и обычный `python3` без venv/pipenv.
4. Когда интернет появится: снова `pipenv install` (или `pip install -r src/gpu_tests/requirements.txt` в venv).

`Pipfile.lock` в репозитории можно добавить позже с машины, где `pipenv lock` проходит — тогда коллегам проще повторять сборку, но **скачать колёса** всё равно нужен доступ к PyPI (или заранее подготовленный кэш pip).

## Сборка снимка

```bash
python3 src/gpu_tests/gpu_info_collect.py --out-dir output
```
