#include "cuda_kernels.h"
#include <cuda_runtime.h>

// -------------------------
// 1. Редукция с использованием только глобальной памяти
// Каждый поток суммирует часть массива и добавляет результат в глобальный результат
// -------------------------
__global__ void reduce_global(int *arr, int *result, int N) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    int sum = 0;
    for(int i = tid; i < N; i += stride)
        sum += arr[i];

    // Используем атомарное сложение, чтобы избежать гонок
    atomicAdd(result, sum);
}

// -------------------------
// 2. Редукция с использованием shared memory
// Сначала суммируем элементы внутри блока в shared memory, затем первый поток блока пишет результат в глобальную память
// -------------------------
__global__ void reduce_shared(int *arr, int *result, int N) {
    extern __shared__ int sdata[]; // динамическая shared память

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Загружаем данные в shared memory
    sdata[tid] = (idx < N) ? arr[idx] : 0;
    __syncthreads();

    // Редукция внутри блока
    for(int s = blockDim.x / 2; s > 0; s >>= 1) {
        if(tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    // Первый поток блока добавляет сумму блока в глобальный результат
    if(tid == 0) atomicAdd(result, sdata[0]);
}
