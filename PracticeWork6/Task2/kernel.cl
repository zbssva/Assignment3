// Ядро для умножения двух матриц
__kernel void mat_mul(
    __global const float* A, // Матрица A размером N x M
    __global const float* B, // Матрица B размером M x K
    __global float* C,       // Результирующая матрица C размером N x K
    const int N,             // Количество строк A
    const int M,             // Количество столбцов A / строк B
    const int K              // Количество столбцов B
) {
    int row = get_global_id(0); // Индекс строки в C
    int col = get_global_id(1); // Индекс столбца в C

    if(row < N && col < K) {
        float sum = 0.0f;
        for(int i = 0; i < M; i++)
            sum += A[row*M + i] * B[i*K + col]; // Сумма произведений строки и столбца
        C[row*K + col] = sum; // Записываем результат
    }
}
