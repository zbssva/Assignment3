%%bash
cat << 'EOF' > add_arrays_test.cu
#include <iostream>
#include <cuda_runtime.h>

// CUDA-ядро для сложения двух массивов
__global__ void add_arrays(float* a, float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Проверка границ массива
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

int main() {
    const int N = 1000000; // Размер массива
    const int size = N * sizeof(float);

    float* h_a; // Массивы в оперативной памяти (CPU)
    float* h_b;
    float* h_c;

    float* d_a; // Массивы в памяти GPU
    float* d_b;
    float* d_c;

    // Выделяем память на CPU
    h_a = (float*)malloc(size);
    h_b = (float*)malloc(size);
    h_c = (float*)malloc(size);

    // Заполняем массивы значениями
    for (int i = 0; i < N; i++) {
        h_a[i] = 1.0f;
        h_b[i] = 2.0f;
    }

    // Выделяем память на GPU
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);

    // Копируем данные с CPU на GPU
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

    // Создаём CUDA-таймеры
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    float time;

    // Запуск ядра с blockSize = 64
    int blockSize1 = 64;
    int gridSize1 = (N + blockSize1 - 1) / blockSize1;

    cudaEventRecord(start);
    add_arrays<<<gridSize1, blockSize1>>>(d_a, d_b, d_c, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    std::cout << "Array addition with blockSize = " << blockSize1 << ", time: " << time << " ms" << std::endl;

    // Запуск ядра с blockSize = 256
    int blockSize2 = 256;
    int gridSize2 = (N + blockSize2 - 1) / blockSize2;

    cudaEventRecord(start);
    add_arrays<<<gridSize2, blockSize2>>>(d_a, d_b, d_c, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    std::cout << "Array addition with blockSize = " << blockSize2 << ", time: " << time << " ms" << std::endl;

    // Освобождаем память
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    free(h_a);
    free(h_b);
    free(h_c);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
EOFX

nvcc add_arrays_test.cu -o add_arrays_test_cuda
./add_arrays_test_cuda
