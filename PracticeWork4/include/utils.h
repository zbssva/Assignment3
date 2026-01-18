#pragma once
#include <vector>
#include <random>
#include <iostream>

// Генерация массива случайных чисел
inline void generate_array(std::vector<int> &arr, int lo = 0, int hi = 100) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dist(lo, hi);
    for(auto &x : arr) x = dist(gen);
}

// Проверка корректности редукции: сравнение с суммой на CPU
inline bool check_sum(const std::vector<int> &cpu, int gpu_sum) {
    int sum = 0;
    for(auto x : cpu) sum += x;
    if(sum != gpu_sum) {
        std::cout << "Error: CPU sum = " << sum << ", GPU sum = " << gpu_sum << "\n";
        return false;
    }
    return true;
}
