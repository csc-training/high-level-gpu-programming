#ifdef __NVCC__
#include <cuda_runtime.h>
#define gpuFree                  cudaFree
#define gpuMalloc                cudaMalloc
#define gpuMemcpy                cudaMemcpy
#define gpuMemcpyHostToDevice    cudaMemcpyHostToDevice
#define gpuMemcpyDeviceToHost    cudaMemcpyDeviceToHost
#define gpuDeviceSynchronize     cudaDeviceSynchronize
#else
#include <hip/hip_runtime.h>
#define gpuFree                  hipFree
#define gpuMalloc                hipMalloc
#define gpuMemcpy                hipMemcpy
#define gpuMemcpyHostToDevice    hipMemcpyHostToDevice
#define gpuMemcpyDeviceToHost    hipMemcpyDeviceToHost
#define gpuDeviceSynchronize     hipDeviceSynchronize
#endif

#include <chrono>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>


__global__ void init_(const size_t n, double *x, double *y)
{
    size_t i = threadIdx.x + blockIdx.x * blockDim.x;
    size_t stride = gridDim.x * blockDim.x;

    for (; i < n; i += stride) {
        x[i] = sin(i) * 2.3;
        y[i] = cos(i) * 1.1;
    }
}

void init(const size_t n, double *x, double *y)
{
    dim3 blocks(n / 1024);
    dim3 threads(1024);
    init_<<<blocks, threads>>>(n, x, y);
}

__global__ void daxpy_(const size_t n, const double a, const double *x, double *y)
{
    size_t i = threadIdx.x + blockIdx.x * blockDim.x;
    size_t stride = gridDim.x * blockDim.x;

    for (; i < n; i += stride) {
        y[i] += a * x[i];
    }
}

void daxpy(const size_t n, const double a, const double *x, double *y)
{
    dim3 blocks(n / 1024);
    dim3 threads(1024);
    daxpy_<<<blocks, threads>>>(n, a, x, y);
#ifdef SYNC
    gpuDeviceSynchronize();
#endif
}

bool check(const std::vector<double> &y);

int main(int argc, char *argv[])
{
    const size_t n = argc > 1 ? (size_t)std::stoll(argv[1]) : 1000000;
    const size_t nit = argc > 2 ? (size_t)std::stoll(argv[2]) : 4;
    const size_t nbytes = sizeof(double) * n;
    double a = 3.4;

#ifndef BENCHMARK
    printf("Vector size %zu (%.2f MiB)\n", n, (double)nbytes / (1024*1024));
    fflush(stdout);
#endif

    // Allocate memory
    double *x, *y;
    gpuMalloc(&x, nbytes);
    gpuMalloc(&y, nbytes);

    // Initialize data on GPU
    init(n, x, y);

    // Calculate on GPU
    daxpy(n, a, x, y);

#ifndef BENCHMARK
    // Copy result to CPU and check correctness
    std::vector<double> h_y(n);
    gpuMemcpy(h_y.data(), y, nbytes, gpuMemcpyDeviceToHost);
    if (check(h_y)) {
        printf("Correctness OK!\n");
        fflush(stdout);
    } else {
        printf("Correctness ERROR!\n");
        return 1;
    }
#endif

    // Measure performance
    gpuDeviceSynchronize();
    using clock = std::chrono::high_resolution_clock;
    auto t0 = clock::now();
    for (size_t i = 0; i < nit; i++) {
        daxpy(n, a, x, y);
    }
    gpuDeviceSynchronize();
    auto t1 = clock::now();
    double time = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count() / 1e9;
    double gflops = 2.0 * n * nit / time / 1e9;
    double gbytess = 3.0 * nbytes * nit / time / 1e9;

#ifndef BENCHMARK
    printf("Time: %.4f s\n", time);
    printf("Performance: %.3f GFLOPS, %.3f GB/s\n", gflops, gbytess);
#else
    printf("%16zu %16.4f %16.3f %16.3f\n", n, (double)nbytes / pow(1024, 2), gflops, gbytess);
#endif

    gpuFree(x);
    gpuFree(y);

    return 0;
}

bool check(const std::vector<double> &y)
{
    double tolerance = 1e-6;
    for (size_t i = 0; i < y.size(); i++) {
        double y_ref = 3.4 * sin(i) * 2.3 + cos(i) * 1.1;

        if (abs(y_ref - y[i]) > tolerance) {
            return false;
        }
    }
    return true;
}
