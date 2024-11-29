#ifdef __NVCC__
#include <cuda_runtime.h>
#define gpuFree                  cudaFree
#define gpuMalloc                cudaMalloc
#define gpuMemcpy                cudaMemcpy
#define gpuMemcpyHostToDevice    cudaMemcpyHostToDevice
#define gpuMemcpyDeviceToHost    cudaMemcpyDeviceToHost
#define gpuMemset                cudaMemset
#define gpuDeviceSynchronize     cudaDeviceSynchronize
#define gpuGetDeviceCount        cudaGetDeviceCount
#define gpuGetDevice             cudaGetDevice
#define gpuSetDevice             cudaSetDevice
#else
#include <hip/hip_runtime.h>
#define gpuFree                  hipFree
#define gpuMalloc                hipMalloc
#define gpuMemcpy                hipMemcpy
#define gpuMemcpyHostToDevice    hipMemcpyHostToDevice
#define gpuMemcpyDeviceToHost    hipMemcpyDeviceToHost
#define gpuMemset                hipMemset
#define gpuDeviceSynchronize     hipDeviceSynchronize
#define gpuGetDeviceCount        hipGetDeviceCount
#define gpuGetDevice             hipGetDevice
#define gpuSetDevice             hipSetDevice
#endif

#if (__cplusplus >= 202002L)
#include <format>
#else
#include <cstdio>
#endif
#include <iostream>
#include <sstream>
#include <vector>

#include <mpi.h>


__global__ void init_(const size_t n, double *x)
{
    size_t i = threadIdx.x + blockIdx.x * blockDim.x;
    size_t stride = gridDim.x * blockDim.x;

    for (; i < n; i += stride) {
        x[i] = 1.1 * i;
    }
}

void init(const size_t n, double *x)
{
    dim3 blocks(n / 256 + n % 256 ? 1 : 0);
    dim3 threads(256);
    init_<<<blocks, threads>>>(n, x);
}

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);

    const size_t n = argc > 1 ? (size_t)std::stoll(argv[1]) : 1024;
    const size_t nbytes = sizeof(double) * n;

    int size, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int count, device;
    gpuGetDeviceCount(&count);
    gpuSetDevice(rank % count);
    gpuGetDevice(&device);

#if (__cplusplus >= 202002L)
    std::cout << std::format("Hello from MPI rank {}/{} with GPU {}/{}\n", rank, size, device, count);
#else
    printf("Hello from MPI rank %d/%d with GPU %d/%d\n", rank, size, device, count);
#endif

    // Device data
    double *x;
    gpuMalloc(&x, nbytes);
    gpuMemset(x, 0, nbytes);
    gpuDeviceSynchronize();

    if (rank == 0) {
        // Initialize data on rank 0
        init(n, x);
        gpuDeviceSynchronize();

        // Send with rank 0
        MPI_Send(x, n, MPI_DOUBLE, 1, 123, MPI_COMM_WORLD);
#if (__cplusplus >= 202002L)
        std::cout << std::format("Rank {} sent\n", rank);
#else
        printf("Rank %d sent\n", rank);
#endif

    } else if (rank == 1) {
        // Receive with rank 1
        MPI_Recv(x, n, MPI_DOUBLE, 0, 123, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
#if (__cplusplus >= 202002L)
        std::cout << std::format("Rank {} received\n", rank);
#else
        printf("Rank %d received\n", rank);
#endif
    }

    // Copy result to CPU and print
    std::vector<double> h_x(n);
    gpuMemcpy(h_x.data(), x, nbytes, gpuMemcpyDeviceToHost);
    std::stringstream ss;
    ss << "Rank " << rank << " has";
    for (int i = 0; i < std::min(8ul, n); ++i) ss << " " << h_x[i];
    if (n > 8) ss << " ...";
    ss << "\n";
    std::cout << ss.str();

    gpuFree(x);

    MPI_Finalize();
}

