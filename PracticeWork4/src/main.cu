#include <iostream>
#include <vector>
#include <fstream>
#include <cuda_runtime.h>
#include "utils.h"
#include "cuda_kernels.h"

// -------------------------
// Замеры времени с использованием cudaEvent
// -------------------------
float measure_kernel_time(dim3 grid, dim3 block, size_t shared_mem, void (*kernel)(int*, int*, int), int *d_array, int *d_result, int N) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    kernel<<<grid, block, shared_mem>>>(d_array, d_result, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    return milliseconds;
}

int main() {
    // -------------------------
    // 1. Параметры
    // -------------------------
    std::vector<int> sizes = {10000, 100000, 1000000};
    int blockSize = 256;

    std::ofstream out("results/reduction_times.csv");
    out << "size,global_time,shared_time\n";

    for(auto N : sizes) {
        std::vector<int> h_array(N);
        generate_array(h_array);

        int *d_array, *d_result;
        cudaMalloc(&d_array, N * sizeof(int));
        cudaMemcpy(d_array, h_array.data(), N * sizeof(int), cudaMemcpyHostToDevice);
        cudaMalloc(&d_result, sizeof(int));

        dim3 grid((N + blockSize - 1) / blockSize);

        // -------------------------
        // 2. Редукция с глобальной памятью
        // -------------------------
        cudaMemset(d_result, 0, sizeof(int));
        reduce_global<<<grid, blockSize>>>(d_array, d_result, N);
        int h_result_global;
        cudaMemcpy(&h_result_global, d_result, sizeof(int), cudaMemcpyDeviceToHost);

        if(!check_sum(h_array, h_result_global))
            std::cout << "Global memory reduction failed for N = " << N << "\n";

        // -------------------------
        // 3. Редукция с shared памятью
        // -------------------------
        cudaMemset(d_result, 0, sizeof(int));
        reduce_shared<<<grid, blockSize, blockSize * sizeof(int)>>>(d_array, d_result, N);
        int h_result_shared;
        cudaMemcpy(&h_result_shared, d_result, sizeof(int), cudaMemcpyDeviceToHost);

        if(!check_sum(h_array, h_result_shared))
            std::cout << "Shared memory reduction failed for N = " << N << "\n";

        // -------------------------
        // 4. Замеры времени
        // -------------------------
        cudaMemset(d_result, 0, sizeof(int));
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cudaEventRecord(start);
        reduce_global<<<grid, blockSize>>>(d_array, d_result, N);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float global_time;
        cudaEventElapsedTime(&global_time, start, stop);

        cudaMemset(d_result, 0, sizeof(int));
        cudaEventRecord(start);
        reduce_shared<<<grid, blockSize, blockSize * sizeof(int)>>>(d_array, d_result, N);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float shared_time;
        cudaEventElapsedTime(&shared_time, start, stop);

        // -------------------------
        // 5. Сохраняем результаты
        // -------------------------
        out << N << "," << global_time << "," << shared_time << "\n";

        cudaFree(d_array);
        cudaFree(d_result);
    }

    out.close();
    std::cout << "Experiment completed. Results saved in results/reduction_times.csv\n";

    return 0;
}
