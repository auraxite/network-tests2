Задача: Вспомогательные функции для gpu_tests

Содержит:
- Проверку ошибок CUDA (checkCudaErrors).
- Функции измерения времени (GpuTimer wrapper).
- Перечисление устройств (cudaGetDeviceCount, cudaGetDeviceProperties).
- Подготовку буферов (allocateDeviceMemory, initializeData).
- Форматированный вывод результатов.