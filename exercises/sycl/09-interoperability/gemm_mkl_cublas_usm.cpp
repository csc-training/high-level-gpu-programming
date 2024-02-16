//==============================================================
// Matrix Multiplication: MKL, CUBLAS
//==============================================================
// This code built using snippets of code from Intel trainings
//
// SPDX-License-Identifier: MIT
// =============================================================

// icpx -std=c++17 -O3 -fsycl -fsycl-targets=spir64_x86_64 -I$MKLROOT/include  -L$MKLROOT/lib/intel64/  -lmkl_sycl -lmkl_core  -lmkl_sequential -lmkl_intel_ilp64  gemm_mkl_usm.cpp
// syclcc -O3 --hipsycl-targets="cuda:sm_80"  
// Add -DMKL_LIB if oneMKL is used

// Add -DCUBLAS if cublas library is use. It needs also -isystem $CUDA_HOME/include/ -L$CUDA_HOME/lib64/ -lcublas -lcudart -lcuda
// Add the appropriate targets: -fsycl-targets=nvptx64-nvidia-cuda,spir64_x86_64 -Xsycl-target-backend=nvptx64-nvidia-cuda --cuda-gpu-arch=sm_80

// Add -DACPP if cublas and Adaptive Cpp is used for compiling
// Add -DDPCPP -DCUDA_NO_HALF if oneapi is used for compiling

// USe one of the option below
// ************************************************************************************************************************************************************

// icpx -std=c++17 -fuse-ld=lld -isystem $CUDA_HOME/include/  -DCUBLAS -DDPCPP -std=c++17 -O3 -fsycl -fsycl-targets=nvptx64-nvidia-cuda -Xsycl-target-backend=nvptx64-nvidia-cuda --cuda-gpu-arch=sm_80  -L$CUDA_HOME/lib64/ -lcublas -lcudart -lcuda

// /projappl/project_2008874/AdaptiveCpp/bin/acpp -fuse-ld=lld -O3 -DCUBLAS -DACPP -I$CUDA_HOME/include/ -L$CUDA_HOME/lib64/ -lcublas -lcudart -lcuda

// icpx -std=c++17 -fuse-ld=lld -O3 -fsycl -fsycl-targets=spir64_x86_64 -I$MKLROOT/include  -L$MKLROOT/lib/intel64/  -lmkl_sycl -lmkl_core  -lmkl_sequential -lmkl_intel_ilp64   -DMKL_LIB gemm_mkl_cublas_usm.cpp 

// ************************************************************************************************************************************************************
//
// If uyou are lucky you make this https://github.com/oneapi-src/oneMKL work
//

#include <sycl/sycl.hpp>
#include <ctime>
#include <chrono>
#include <getopt.h>
#if MKL_LIB
#include "oneapi/mkl.hpp"  //# oneMKL DPC++ interface
#endif

#if CUBLAS 
// cuda interface
#include <cublas_v2.h>
#include <cuda.h>


#define CHECK_ERROR(FUNC) checkCudaErrorMsg(FUNC, " " #FUNC)

void inline checkCudaErrorMsg(cublasStatus_t status, const char *msg) {
  if (status != CUBLAS_STATUS_SUCCESS) {
    std::cout << "ERROR CUBLAS:" << msg << " - " << status << std::endl;
    exit(EXIT_FAILURE);
  }
}
#endif

using namespace sycl;
#if MKL_LIB
using namespace oneapi::mkl;
#endif

int main(int argc, char *argv[]) {
    
    size_t N = 1024;
    size_t M = 32;
    int VERIFY = 0;
    int PRINT_OUTPUT_MATRIX = 0;
    
    int arg;
    while ((arg = getopt (argc, argv, "n:m:vp:h")) != -1)
        switch (arg){
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

    //# Define vectors for matricies
    std::vector<float> matrix_a(N*N);
    std::vector<float> matrix_b(N*N);
    std::vector<float> matrix_c(N*N);
    std::vector<float> matrix_d(N*N);
    
    //# Initialize matricies with values
    float v1 = 2.f;
    float v2 = 3.f;
    for (int i=0; i<N; i++)
        for (int j=0; j<N; j++){
            matrix_a[i*N+j] = v1++;
            matrix_b[i*N+j] = v2++;
            matrix_c[i*N+j] = 0.f;
            matrix_d[i*N+j] = 0.f;
    }
    //# Define queue with default device for offloading computation
    queue q{property::queue::in_order{}};

    // First we warm-up the device
    std::cout << "Warm-up first" << "\n"; 

    {
        //# Create buffers for matrices

        buffer<float, 1> a(matrix_a.data(), range<1>(N*N));
        buffer<float, 1> b(matrix_b.data(), range<1>(N*N));
        buffer<float, 1> c(matrix_c.data(), range<1>(N*N));
       
         //# Submit command groups to execute on device
         q.submit([&](handler &h){
            //# Create accessors to copy buffers to the device
            auto A = a.get_access<access::mode::read>(h);
            auto B = b.get_access<access::mode::read>(h);
            auto C = c.get_access<access::mode::write>(h);

            //# Define size for ND-Range and work-group size
            range<2> global_size(N,N);
            range<2> work_group_size(M,M);

            //# Parallel Compute Matrix Multiplication
            h.parallel_for(nd_range<2>{global_size, work_group_size}, [=](nd_item<2> item){
                const int i = item.get_global_id(0);
                const int j = item.get_global_id(1);
                //# Use private mem to store intermediate result
                float temp=0.f;
                for (int k = 0; k < N; k++) {
                   temp += A[i*N+k] * B[k*N+j];
               }
               C[i*N+j]  = temp;
            });
        });
    } // warm-up done

    //# Initialize matrices with values
    v1 = 2.f;
    v2 = 3.f;
    for (int i=0; i<N; i++)
        for (int j=0; j<N; j++){
            matrix_a[i*N+j] = v1++;
            matrix_b[i*N+j] = v2++;
            matrix_c[i*N+j] = 0.f;
            matrix_d[i*N+j] = 0.f;
    }
    
    auto start = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    
    std::cout << "Offload Device        : " << q.get_device().get_info<info::device::name>() << "\n";
    std::cout << "max_work_group_size   : " << q.get_device().get_info<info::device::max_work_group_size>() << "\n";
    std::cout << "Configuration         : MATRIX_SIZE= " << N << "x" << N << "\n";
    float* dev_a = sycl::malloc_device<float>(N*N, q);
    float* dev_b = sycl::malloc_device<float>(N*N, q);
    float* dev_c = sycl::malloc_device<float>(N*N, q); 
    q.memcpy(dev_a, matrix_a.data(), N*N*sizeof(float)).wait();
    q.memcpy(dev_b, matrix_b.data(), N*N*sizeof(float)).wait();
    
    //# scalar multipliers
    float alpha = 1.f, beta = 1.f;

#if MKL_LIB // uses mkl blas    
        
    //# transpose status of matrices for oneMKL
    oneapi::mkl::transpose transA = oneapi::mkl::transpose::nontrans;
    oneapi::mkl::transpose transB = oneapi::mkl::transpose::nontrans;
  
        
    //# Submit MKL library call to execute on device
    blas::gemm(q, transA, transB, N, N, N, alpha, dev_b, N, dev_a, N, beta, dev_c, N); 

    q.wait(); 
#endif  

#if CUBLAS

// Create cublas handle
  cublasHandle_t handle;
  CHECK_ERROR(cublasCreate(&handle));

#if ACPP
std::cout << "\n"<< "Running with ACPP interoperability. \n";
q.submit([&](handler &cgh) {
     cgh.hipSYCL_enqueue_custom_operation([=](sycl::interop_handle &ih) {
       // Set the correct  stream
       auto cuStream = ih.get_native_queue<sycl::backend::cuda>();
       cublasSetStream(handle, cuStream);
       // Call generalised matrix-matrix multiply
       // Call generalised matrix-matrix multiply
       CHECK_ERROR(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N,N,
                               N, &alpha, dev_b, N, dev_a, N, &beta,
                               dev_c, N));
     });
   }).wait();
#endif

#if DPCPP  
  std::cout << "\n"<< "Warning!!! " << " \n" << " The DPC++ & CUDA \n relies on the number of assumptions:\n in-order queues,\n no event- or buffer-based dependencies, \n no frequent switching between multiple devices \n stars aligning properly.\n\n"; 
  q.submit([&](handler &h) {
     h.host_task([=](sycl::interop_handle ih) {
       // Set the correct cuda context & stream
       cuCtxSetCurrent(ih.get_native_context<backend::ext_oneapi_cuda>());
       auto cuStream = ih.get_native_queue<backend::ext_oneapi_cuda>();
       cublasSetStream(handle, cuStream);

       // Call generalised matrix-matrix multiply
       CHECK_ERROR(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N,N,
                               N, &alpha, dev_b, N, dev_a, N, &beta,
                               dev_c, N));
       cuStreamSynchronize(cuStream);
     });
   }).wait();
#endif

#endif
  
    //auto duration = std::chrono::high_resolution_clock::now() - start;
    //double elapsed_seconds = std::chrono::duration_cast<std::chrono::duration<double>>(duration).count();
    //std::cout << "\n Execute in " << elapsed_seconds << " s\n";

    q.memcpy(matrix_c.data(), dev_c, N*N*sizeof(float)).wait();
    //# Print Output
    if (PRINT_OUTPUT_MATRIX){
        for (int i=0; i<N; i++){
            for (int j=0; j<N; j++){
                std::cout << matrix_c[i*N+j] << " ";
            }
            std::cout << "\n";
        }
    } else {
        std::cout << " [0][0] = " << matrix_c[0] << "\n";
    }
    
    //# Compute local and compare with offload computation
    if (VERIFY){
        int fail = 0;
        for(int i=0; i<N; i++){
            for (int j = 0; j < N; j++) {
                for(int k=0; k<N; k++){
                    matrix_d[i*N+j] += matrix_a[i*N+k] * matrix_b[k*N+j];
                }
                if(matrix_c[i*N+j] != matrix_d[i*N+j]) fail = 1;
            }
        }
        if(fail == 1){
            std::cout << "FAIL\n";
        } else {
            std::cout << "PASS\n";
        }
    }
    free(dev_a, q);
    free(dev_b, q);
    free(dev_c, q);
}





