

#include <iostream>
#include <vector>
#include <chrono>
#include <cstdlib>
#include <unistd.h> 
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define CHECK_ERROR(FUNC) checkCudaErrorMsg(FUNC, #FUNC)

void inline checkCudaErrorMsg(cublasStatus_t status, const char *msg) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        std::cout << "ERROR CUBLAS: " << msg << " - " << status << std::endl;
        exit(EXIT_FAILURE);
    }
}

int main(int argc, char *argv[]) {
    
    size_t N = 1024;
    size_t M = 32;
    int VERIFY = 0;
    int PRINT_OUTPUT_MATRIX = 0;
    
    int arg;
    while ((arg = getopt (argc, argv, "n:m:vp:h")) != -1) {
        switch (arg) {
            case 'n':
                N = std::atoi(optarg);
                break;
            case 'm':
                M = std::atoi(optarg);
                break;
            case 'v':
                VERIFY = 1;
                break;
            case 'p':
                PRINT_OUTPUT_MATRIX = 1;
                break;
            case 'h':
                std::cout << std::endl;
                std::cout << "Usage   : ./a.out -n <MATRIX_SIZE> -m <WORK_GROUP_SIZE> -v -p\n\n";
                std::cout << "          [-n] size for matrix, eg: 1024\n";
                std::cout << "          [-m] size of work_group, eg: 8/16\n";
                std::cout << "          [-v] verify output with linear computation on cpu\n";
                std::cout << "          [-p] print output matrix\n";
                std::cout << "Example : ./a.out -n 1024 -m 16 -v -p\n\n";
                std::exit(0);
        }
    }

    // Define vectors for matrices
    std::vector<float> matrix_a(N*N);
    std::vector<float> matrix_b(N*N);
    std::vector<float> matrix_c(N*N);
    std::vector<float> matrix_d(N*N);
    
    // Initialize matrices with values
    float v1 = 2.f;
    float v2 = 3.f;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            matrix_a[i*N+j] = v1++;
            matrix_b[i*N+j] = v2++;
            matrix_c[i*N+j] = 0.f;
            matrix_d[i*N+j] = 0.f;
        }
    }

    // CUDA initialization
    cudaSetDevice(0);
    cublasHandle_t handle;
    CHECK_ERROR(cublasCreate(&handle));

    // Start timing
    auto start = std::chrono::high_resolution_clock::now();

    // Copy matrices to device memory
    float *dev_a, *dev_b, *dev_c;
    cudaMalloc(&dev_a, N*N*sizeof(float));
    cudaMalloc(&dev_b, N*N*sizeof(float));
    cudaMalloc(&dev_c, N*N*sizeof(float));
    cudaMemcpy(dev_a, matrix_a.data(), N*N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, matrix_b.data(), N*N*sizeof(float), cudaMemcpyHostToDevice);
    
    // Scalar multipliers
    float alpha = 1.f, beta = 1.f;

    // Perform matrix multiplication
    CHECK_ERROR(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, dev_a, N, dev_b, N, &beta, dev_c, N));

    // End timing
    auto duration = std::chrono::high_resolution_clock::now() - start;
    double elapsed_seconds = std::chrono::duration_cast<std::chrono::duration<double>>(duration).count();
    std::cout << "\n Execute in " << elapsed_seconds << " s\n";

    // Copy result back to host
    cudaMemcpy(matrix_c.data(), dev_c, N*N*sizeof(float), cudaMemcpyDeviceToHost);

    // Print Output
    if (PRINT_OUTPUT_MATRIX) {
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                std::cout << matrix_c[i*N+j] << " ";
            }
            std::cout << "\n";
        }
    } else {
        std::cout << "[0][0] = " << matrix_c[0] << "\n";
    }

    // Compute local and compare with offload computation
    if (VERIFY) {
        int fail = 0;
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                for (int k = 0; k < N; k++) {
                    matrix_d[i*N+j] += matrix_a[i*N+k] * matrix_b[k*N+j];
                }
                if (matrix_c[i*N+j] != matrix_d[i*N+j]) {
                    fail = 1;
                    break;
                }
            }
            if (fail) break;
        }
        if (fail == 1) {
            std::cout << "FAIL\n";
        } else {
            std::cout << "PASS\n";
        }
    }

    // Clean up
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);
    cublasDestroy(handle);

    return 0;
}

