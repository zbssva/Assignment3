#include <CL/cl.hpp>       // Заголовок OpenCL C++ API
#include <vector>
#include <iostream>
#include <fstream>
#include <chrono>          // Для измерения времени выполнения

int main() {
    const int N = 1024*1024; // Размер массивов

    // 1. Подготовка данных
    std::vector<float> A(N, 1.0f); // Массив A, заполнен 1.0
    std::vector<float> B(N, 2.0f); // Массив B, заполнен 2.0
    std::vector<float> C(N, 0.0f); // Массив C для результата

    // 2. Получаем доступные платформы OpenCL
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    if(platforms.empty()) { 
        std::cout << "OpenCL платформа не найдена!\n"; 
        return 1; 
    }
    cl::Platform platform = platforms.front(); // Берем первую платформу

    // 3. Получаем доступные устройства (CPU/GPU)
    std::vector<cl::Device> devices;
    platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);
    if(devices.empty()) { 
        std::cout << "OpenCL устройства не найдены!\n"; 
        return 1; 
    }
    cl::Device device = devices.front(); // Выбираем первое устройство

    // 4. Создаем контекст для выбранного устройства
    cl::Context context(device);

    // 5. Создаем очередь команд для запуска ядра
    cl::CommandQueue queue(context, device);

    // 6. Загружаем код ядра из файла kernel.cl
    std::ifstream file("kernel.cl");
    std::string src(std::istreambuf_iterator<char>(file), {});
    cl::Program program(context, src);

    // 7. Компилируем ядро для выбранного устройства
    if(program.build({device}) != CL_SUCCESS) {
        std::cout << "Ошибка компиляции: " 
                  << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device) << "\n";
        return 1;
    }

    // 8. Создаем объект ядра
    cl::Kernel kernel(program, "vector_add");

    // 9. Создаем буферы в глобальной памяти устройства
    cl::Buffer bufA(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * N, A.data());
    cl::Buffer bufB(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * N, B.data());
    cl::Buffer bufC(context, CL_MEM_WRITE_ONLY, sizeof(float) * N);

    // 10. Передаем буферы в ядро
    kernel.setArg(0, bufA);
    kernel.setArg(1, bufB);
    kernel.setArg(2, bufC);

    // 11. Запуск ядра и измерение времени
    auto start = std::chrono::high_resolution_clock::now();
    queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(N), cl::NullRange);
    queue.finish(); // Ждем окончания выполнения всех команд
    auto end = std::chrono::high_resolution_clock::now();

    // 12. Читаем результат из буфера на хост
    queue.enqueueReadBuffer(bufC, CL_TRUE, 0, sizeof(float) * N, C.data());

    // 13. Вывод результатов и времени выполнения
    std::cout << "Время выполнения: " 
              << std::chrono::duration<double,std::milli>(end-start).count() << " мс\n";
    std::cout << "C[0] = " << C[0] << ", C[N-1] = " << C[N-1] << std::endl;

    return 0;
}
