#include <CL/cl.hpp>
#include <vector>
#include <iostream>
#include <fstream>
#include <chrono>

int main() {
    const int N = 512;  // Строки матрицы A
    const int M = 512;  // Столбцы A / строки B
    const int K = 512;  // Столбцы матрицы B

    // 1. Инициализация матриц
    std::vector<float> A(N*M, 1.0f); // Матрица A заполнена 1
    std::vector<float> B(M*K, 2.0f); // Матрица B заполнена 2
    std::vector<float> C(N*K, 0.0f); // Результат

    // 2. Получаем платформы OpenCL
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    cl::Platform platform = platforms.front();

    // 3. Получаем устройства (CPU/GPU)
    std::vector<cl::Device> devices;
    platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);
    cl::Device device = devices.front();

    // 4. Создаем контекст и очередь команд
    cl::Context context(device);
    cl::CommandQueue queue(context, device);

    // 5. Загружаем и компилируем ядро
    std::ifstream file("kernel.cl");
    std::string src(std::istreambuf_iterator<char>(file), {});
    cl::Program program(context, src);
    program.build({device});

    cl::Kernel kernel(program, "mat_mul");

    // 6. Создаем буферы
    cl::Buffer bufA(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float)*N*M, A.data());
    cl::Buffer bufB(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float)*M*K, B.data());
    cl::Buffer bufC(context, CL_MEM_WRITE_ONLY, sizeof(float)*N*K);

    // 7. Передаем аргументы ядра
    kernel.setArg(0, bufA);
    kernel.setArg(1, bufB);
    kernel.setArg(2, bufC);
    kernel.setArg(3, N);
    kernel.setArg(4, M);
    kernel.setArg(5, K);

    // 8. Запуск ядра и замер времени
    auto start = std::chrono::high_resolution_clock::now();
    queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(N,K), cl::NullRange);
    queue.finish();
    auto end = std::chrono::high_resolution_clock::now();

    // 9. Чтение результатов
    queue.enqueueReadBuffer(bufC, CL_TRUE, 0, sizeof(float)*N*K, C.data());

    // 10. Вывод результатов
    std::cout << "Время выполнения: "
              << std::chrono::duration<double,std::milli>(end-start).count() << " мс\n";
    std::cout << "C[0] = " << C[0] << ", C[N*K-1] = " << C[N*K-1] << std::endl;

    return 0;
}
