Задача: Измерение задержки передачи данных между двумя GPU (GPU↔GPU) через PCIe / NVLink.

Проверяет доступность P2P (cudaDeviceCanAccessPeer).
Активирует P2P (cudaDeviceEnablePeerAccess).
Измеряет round-trip latency при cudaMemcpyPeer.
Может запускаться с MPI для GPU↔GPU между узлами (через CUDA IPC + MPI).

Выходные данные:
- Таблица задержек между всеми парами GPU.
- Обнаружение «медленных» связей (разные PCIe или NVLink линии).