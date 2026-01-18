#include "cuda_kernels.h"

// -------------------------
// 1. Bubble sort для подмассива в локальной/shared памяти
// -------------------------
__device__ void bubble_sort_local(int *arr, int N) {
    for(int i = 0; i < N-1; ++i) {
        for(int j = 0; j < N-i-1; ++j) {
            if(arr[j] > arr[j+1]) {
                int tmp = arr[j];
                arr[j] = arr[j+1];
                arr[j+1] = tmp;
            }
        }
    }
}

// -------------------------
// 2. Сортировка подмассивов на GPU
// subsize - размер подмассива для одного блока
// -------------------------
__global__ void sort_subarrays(int *arr, int subsize, int N) {
    extern __shared__ int s_arr[]; // shared memory для подмассива

    int start = blockIdx.x * subsize;
    int tid = threadIdx.x;
    if(start + tid < N)
        s_arr[tid] = arr[start + tid]; // копируем подмассив в shared memory
    __syncthreads();

    bubble_sort_local(s_arr, subsize); // сортируем локально
    __syncthreads();

    if(start + tid < N)
        arr[start + tid] = s_arr[tid]; // копируем обратно в глобальную память
}
