#pragma once

// Редукция суммы в глобальной памяти
__global__ void reduce_global(int *arr, int *result, int N);

// Редукция суммы с shared memory
__global__ void reduce_shared(int *arr, int *result, int N);

// Сортировка подмассивов (bubble sort) с использованием shared memory
__global__ void sort_subarrays(int *arr, int subsize, int N);

// Функция сортировки для потоков блока
__device__ void bubble_sort_local(int *arr, int N);
