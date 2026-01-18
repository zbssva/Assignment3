//с использованием РАЗДЕЛЯЕМОЙ памяти
__global__ void multiply_shared(float* data, float k, int n) {

    // Разделяемая память — общая для всех потоков в блоке
    __shared__ float cache[256];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;  // Локальный индекс потока

    // Копируем данные из глобальной памяти в shared memory
    if (idx < n) {
        cache[tid] = data[idx];
    }

    // Ждём, пока ВСЕ потоки блока закончат копирование
    __syncthreads();

    // Выполняем вычисление и записываем результат
    if (idx < n) {
        data[idx] = cache[tid] * k;
    }
}
