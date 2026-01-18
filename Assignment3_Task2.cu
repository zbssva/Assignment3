__global__ void add_arrays(float* a, float* b, float* c, int n) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Проверка границ массива
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}
int blockSizes[] = {128, 256, 512};

for (int i = 0; i < 3; i++) {
    int blockSize = blockSizes[i];
    int gridSize = (N + blockSize - 1) / blockSize;

    add_arrays<<<gridSize, blockSize>>>(a, b, c, N);
}
