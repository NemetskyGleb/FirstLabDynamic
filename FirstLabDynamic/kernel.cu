
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>
#include <sstream>
#include <chrono>
#include <functional>
#include <algorithm>

constexpr int T = 1024; // max threads per block

void substractWithCuda(int* c, const int* a, const int* b, uint32_t size);

void subsractCPU(int* c, const int* a, const int* b, uint32_t arraySize)
{
    for (size_t i = 0; i < arraySize; i++)
    {
        c[i] = a[i] - b[i];
    }
}

__global__ void substractKernel(int* c, const int* a, const int* b, int size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size)
    { 
        c[i] = a[i] - b[i];
    }
}

void calculateFunctionTime(int* c, const int* a, const int* b, uint32_t size,
        std::function<void(int* c, const int*, const int*, uint32_t)> substract)
{
    using namespace std::chrono;

    auto start = high_resolution_clock::now();

    substract(c, a, b, size);

    auto stop = high_resolution_clock::now();

    auto duration = duration_cast<milliseconds>(stop - start);

    std::cout << "Time taken by function: "
              << duration.count() << " milliseconds" << std::endl;
}

void PrintResult(const int* a, const int* b, const int* c, uint32_t size)
{
    std::string result;
    
    result += "{" + std::to_string(a[0]) + ", " + std::to_string(a[1]) + ",..., " 
               + std::to_string(a[size - 1]) + ", " + std::to_string(a[size]) + "} - ";
    result += "{" + std::to_string(b[0]) + ", " + std::to_string(b[1]) + ",..., "
               + std::to_string(b[size - 1]) + ", " + std::to_string(b[size]) + "} = ";
    result += "{" + std::to_string(c[0]) + ", " + std::to_string(c[1]) + ",..., "
               + std::to_string(c[size - 1]) + ", " + std::to_string(c[size]) + "}";

    std::cout << result << std::endl;
}


int main()
{
    uint32_t arraySize;
    std::cout << "Enter size of array: ";
    std::cin >> arraySize;

    int* a = new int[arraySize];
    int* b = new int[arraySize];
    int* c1 = new int[arraySize];
    int* c2 = new int[arraySize];

    auto randNumber = []() -> int {
        return rand() % 100;
    };

    std::generate(a, a + arraySize, randNumber);
    std::generate(b, b + arraySize, randNumber);

    calculateFunctionTime(c1, a, b, arraySize, &subsractCPU);
    std::cout << "Result on CPU:" << std::endl;
    
    PrintResult(a, b, c1, arraySize);

    delete[] c1;

    substractWithCuda( c2, a, b, arraySize );

    PrintResult(a, b, c2, arraySize);
 
    delete[] c2, a, b;
    
    getchar();
    
    return 0;
}

 //Helper function for using CUDA to substract vectors in parallel.
void substractWithCuda(int* c, const int* a, const int* b, uint32_t size)
{
    int* dev_a = 0;
    int* dev_b = 0;
    int* dev_c = 0;
    cudaError_t cudaStatus;

    auto checkError = [&](cudaError_t status)
    {
        if (status != cudaSuccess)
        {
            std::cerr << "Error! ";
            std::cerr << cudaGetErrorString(status) << std::endl;
            cudaFree(dev_c);
            cudaFree(dev_a);
            cudaFree(dev_b);
            return;
        }
    };

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    checkError(cudaStatus);

    // Allocate GPU buffers for three vectors (two input, one output).
    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
    checkError(cudaStatus);
    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
    checkError(cudaStatus);
    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
    checkError(cudaStatus);

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    checkError(cudaStatus);
    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
    checkError(cudaStatus);

    // инициализируем события
    cudaEvent_t start, stop;
    float elapsedTime;
    // создаем события
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    // запись события
    cudaEventRecord(start, 0);

    // Launch a kernel on the GPU with one thread for each element.
    substractKernel<<<(int)ceil((float)size / T ), T>>>(dev_c, dev_a, dev_b, size);
    
    cudaStatus = cudaEventRecord(stop, 0);
    checkError(cudaStatus);
    // ожидание завершения работы ядра
    cudaStatus = cudaEventSynchronize(stop);
    checkError(cudaStatus);
    cudaStatus = cudaEventElapsedTime(&elapsedTime, start, stop);
    checkError(cudaStatus);
    // вывод информации
    printf("Time spent executing by the GPU: %.2f milliseconds\n", elapsedTime);
    // уничтожение события
    cudaStatus = cudaEventDestroy(start);
    checkError(cudaStatus);
    cudaStatus = cudaEventDestroy(stop);
    checkError(cudaStatus);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    checkError(cudaStatus);

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    checkError(cudaStatus);

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
    checkError(cudaStatus);
    
    cudaStatus = cudaDeviceReset();
    checkError(cudaStatus);

    // Free resources.
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
}
