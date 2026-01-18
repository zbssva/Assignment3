__global__ void coalesced(float* data, int n) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        // Потоки читают соседние элементы памяти
        data[idx] *= 2.0f;
    }
}


__global__ void non_coalesced(float* data, int n) {

    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 2;

    if (idx < n) {
        // Потоки обращаются к памяти с шагом,
        // из-за чего GPU делает больше обращений к памяти
        data[idx] *= 2.0f;
    }
}
