#include <iostream>  // подключаем библиотеку для ввода/вывода на CPU
#include <cuda.h>    // подключаем библиотеку CUDA для работы с GPU

#define STACK_SIZE 1024       // максимальный размер стека
#define QUEUE_SIZE 1024       // максимальный размер очереди
#define THREADS 256           // количество потоков в блоке
#define BLOCKS 4              // количество блоков в сетке

// ======================= Параллельный стек =======================

// Структура параллельного стека
struct Stack {
    int *data;        // указатель на массив данных стека в GPU памяти
    int top;          // индекс вершины стека
    int capacity;     // максимальная емкость стека

    // Метод инициализации стека в GPU памяти
    __device__ void init(int *buffer, int size) {
        data = buffer;   // присваиваем массив данных
        top = -1;        // стек пуст, вершина = -1
        capacity = size; // сохраняем емкость
    }

    // Метод добавления элемента в стек (push)
    __device__ bool push(int value) {
        int pos = atomicAdd(&top, 1) + 1; // атомарно увеличиваем top и получаем позицию
        if (pos < capacity) {             // проверка на переполнение
            data[pos] = value;            // записываем значение
            return true;                  // успешно
        } else {
            atomicSub(&top, 1);           // откат, если переполнение
            return false;                 // push не выполнен
        }
    }

    // Метод извлечения элемента из стека (pop)
    __device__ bool pop(int *value) {
        int pos = atomicSub(&top, 1);     // атомарно уменьшаем top и получаем позицию
        if (pos >= 0) {                   // проверка на пустой стек
            *value = data[pos];           // возвращаем значение через указатель
            return true;                  // успешно
        } else {
            atomicAdd(&top, 1);           // откат, если стек пуст
            return false;                 // pop не выполнен
        }
    }
};

// CUDA-ядро для тестирования стека
__global__ void stackKernel(Stack stack, int *results) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x; // глобальный индекс потока
    stack.push(tid);         // каждый поток пытается добавить свой ID
    int value;
    if (stack.pop(&value)) { // каждый поток пытается извлечь значение
        results[tid] = value; // сохраняем результат для проверки
    }
}

// ======================= Параллельная очередь =======================

struct Queue {
    int *data;      // указатель на массив данных очереди в GPU памяти
    int head;       // индекс головы очереди
    int tail;       // индекс хвоста очереди
    int capacity;   // максимальная емкость очереди

    // Инициализация очереди
    __device__ void init(int *buffer, int size) {
        data = buffer; // массив данных
        head = 0;      // голова на 0
        tail = 0;      // хвост на 0
        capacity = size;
    }

    // Добавление элемента (enqueue)
    __device__ bool enqueue(int value) {
        int pos = atomicAdd(&tail, 1); // атомарно увеличиваем tail
        if (pos < capacity) {          // проверка переполнения
            data[pos] = value;         // записываем значение
            return true;               // успешно
        } else {
            atomicSub(&tail, 1);       // откат при переполнении
            return false;              // не удалось
        }
    }

    // Извлечение элемента (dequeue)
    __device__ bool dequeue(int *value) {
        int pos = atomicAdd(&head, 1); // атомарно увеличиваем head
        if (pos < tail) {              // проверка, что очередь не пуста
            *value = data[pos];        // возвращаем значение
            return true;               // успешно
        } else {
            atomicSub(&head, 1);       // откат, если очередь пуста
            return false;              // dequeue не выполнен
        }
    }
};

// CUDA-ядро для тестирования очереди
__global__ void queueKernel(Queue queue, int *results) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x; // глобальный индекс потока
    queue.enqueue(tid);       // каждый поток добавляет свой ID
    int value;
    if (queue.dequeue(&value)) { // каждый поток пытается извлечь значение
        results[tid] = value;    // сохраняем результат для проверки
    }
}

// ======================= Хост-код =======================

int main() {
    // ----- Работа со стеком -----
    Stack h_stack;                    // переменная для стека на хосте
    int *d_stackData;                 // указатель на данные стека в GPU памяти
    int *d_stackResults;              // массив для результатов
    cudaMalloc(&d_stackData, STACK_SIZE * sizeof(int));    // выделяем память под данные стека
    cudaMalloc(&d_stackResults, THREADS * BLOCKS * sizeof(int)); // выделяем память под результаты

    Stack *d_stack;                   // указатель на стек в GPU памяти
    cudaMalloc(&d_stack, sizeof(Stack));
    Stack tempStack;
    tempStack.init(d_stackData, STACK_SIZE);             // инициализация стека
    cudaMemcpy(d_stack, &tempStack, sizeof(Stack), cudaMemcpyHostToDevice); // копируем на GPU

    stackKernel<<<BLOCKS, THREADS>>>(*d_stack, d_stackResults); // запуск CUDA-ядра
    cudaDeviceSynchronize();                                      // синхронизация потоков

    int h_stackResults[THREADS * BLOCKS];                        
    cudaMemcpy(h_stackResults, d_stackResults, THREADS * BLOCKS * sizeof(int), cudaMemcpyDeviceToHost); // копируем результаты на CPU

    std::cout << "Stack results:\n";  // вывод результатов
    for(int i = 0; i < THREADS * BLOCKS; i++) std::cout << h_stackResults[i] << " ";
    std::cout << std::endl;

    // освобождение памяти стека
    cudaFree(d_stackData);
    cudaFree(d_stackResults);
    cudaFree(d_stack);

    // ----- Работа с очередью -----
    Queue h_queue;
    int *d_queueData;
    int *d_queueResults;
    cudaMalloc(&d_queueData, QUEUE_SIZE * sizeof(int));          // память для очереди
    cudaMalloc(&d_queueResults, THREADS * BLOCKS * sizeof(int)); // память для результатов очереди

    Queue *d_queue;
    cudaMalloc(&d_queue, sizeof(Queue));
    Queue tempQueue;
    tempQueue.init(d_queueData, QUEUE_SIZE);                    // инициализация очереди
    cudaMemcpy(d_queue, &tempQueue, sizeof(Queue), cudaMemcpyHostToDevice); // копирование на GPU

    queueKernel<<<BLOCKS, THREADS>>>(*d_queue, d_queueResults); // запуск ядра очереди
    cudaDeviceSynchronize();                                     // синхронизация

    int h_queueResults[THREADS * BLOCKS];
    cudaMemcpy(h_queueResults, d_queueResults, THREADS * BLOCKS * sizeof(int), cudaMemcpyDeviceToHost); // копируем результаты

    std::cout << "Queue results:\n";  // вывод результатов
    for(int i = 0; i < THREADS * BLOCKS; i++) std::cout << h_queueResults[i] << " ";
    std::cout << std::endl;

    // освобождение памяти очереди
    cudaFree(d_queueData);
    cudaFree(d_queueResults);
    cudaFree(d_queue);

    return 0;
}
