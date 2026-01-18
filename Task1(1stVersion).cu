//ТОЛЬКО с глобальной памятью
#include <iostream>
#include <cuda_runtime.h>

using namespace std;

// CUDA-ядро (код, который выполняется на GPU)
__global__ void multiply_global(float* data, float k, int n) {

    // Вычисляем глобальный индекс элемента
    // blockIdx.x     — номер блока
    // blockDim.x     — сколько потоков в блоке
    // threadIdx.x    — номер потока внутри блока
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Проверяем, чтобы не выйти за границы массива
    if (idx < n) {
        // Берём элемент из глобальной памяти,
        // умножаем на число k
        // и записываем обратно в глобальную память
        data[idx] = data[idx] * k;
    }
}

int main() {
    const int N = 1'000'000;        // Размер массива
    const int size = N * sizeof(float);

    float* h_data;                  // Массив в оперативной памяти (CPU)
    float* d_data;                  // Массив в памяти GPU

    // Выделяем память на CPU
    h_data = (float*)malloc(size);

    // Заполняем массив значениями
    for (int i = 0; i < N; i++) {
        h_data[i] = 1.0f;
    }

    // Выделяем память на GPU
    cudaMalloc(&d_data, size);

    // Копируем данные с CPU на GPU
    cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice);

    // Настраиваем запуск ядра
    int blockSize = 256;                       // Потоков в одном блоке
    int gridSize = (N + blockSize - 1) / blockSize; // Количество блоков

    // Создаём CUDA-таймеры
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Запускаем таймер
    cudaEventRecord(start);

    // Запускаем CUDA-ядро
    multiply_global<<<gridSize, blockSize>>>(d_data, 2.0f, N);

    // Останавливаем таймер
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float time;
    cudaEventElapsedTime(&time, start, stop);

    cout << "Global memory time: " << time << " ms" << endl;

    // Копируем результат обратно на CPU
    cudaMemcpy(h_data, d_data, size, cudaMemcpyDeviceToHost);

    // Освобождаем память
    cudaFree(d_data);
    free(h_data);

    return 0;
}
