# Using 3rd party libraries. 
Third-party libraries are essential for GPU programming, offering optimized functions and algorithms tailored for specific tasks while abstracting low-level hardware details. This abstraction enables developers to focus on algorithm design and application development, enhancing productivity and efficiency.

For Intel hardware, the [Intel® oneAPI Math Kernel Library](https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl.html) (oneMKL) provides a comprehensive set of low-level routines for math operations. It is highly optimized and extensively parallelized, supporting Intel CPUs and GPUs, with a SYCL interface for integration into SYCL codes.

Nvidia hardware users can leverage the [CUDA toolkit](https://developer.nvidia.com/cuda-toolkit), offering a suite of highly optimized libraries such as cuBLAS and cuFFT. These libraries enhance performance for various computational tasks on Nvidia GPUs.

Similarly, for AMD hardware, the [ROCm/HIP](https://rocm.docs.amd.com/projects/HIP/en/latest/) libraries provide optimized routines for GPU-accelerated computing, enabling efficient utilization of AMD GPUs.

At the moment there is no unified interface to select automatically the needed library based on the device with which each queue is associated. Furthermore some libraries such as cu/hipBlas or cu/hipFFT require a gpu stream submit the work in an asynchronous way and integrate the call with previous ans subsenquent kernels. However in general a SYCL queue is not necesseraly associated with given stream. The SYCL standard defines the queue behaviour, but it is up to the `SYCL implementation` how that is mapped to the `cuda/hip stream`. 

## Matrix-Matrix multiplication example
Take the example of dense matrix-matrix multimplication. We saw the [memory optimization exercise](../04-matrix-matrix-mul/) how we can improve the performance. This direct programming is acceptable if the operation is only a very small percentage of the total computating time or if only a few cases (system sizes) are needed and optimization is done for those specific cases. 

For more general cases an optimized library is needed and it depends on the target hardware with wich the queue is associated.

### Intel Hardware with oneMKL and oneAPI
The oneMKL libraries have SYCL interface. They take as argument the SYCL queues and they support both buffers and USM pointers. In some way this is the easiestway. To multiply two matrices just use this:
```
    //# transpose status of matrices for oneMKL
    oneapi::mkl::transpose transA = oneapi::mkl::transpose::nontrans, transB = oneapi::mkl::transpose::nontrans;
    oneapi::mkl::blas::gemm(q, transA, transB, N, N, N, alpha, dev_b, N, dev_a, N, beta, dev_c, N);
```
Where `dev_a` and `dev_b` contain the input data and the results is saved in `dev_c`. These pointers are allocated via USM calls. This call is asynchronous and if the data is needed imediatly the `q.wait()`  call is needed. The SYCL queue which is given as argument needs to be associated with a CPU or an Intel GPU (spir targets). 
In addition to this the proper header needs to be included:
```
#include "oneapi/mkl.hpp"
```

Compiling the code requires some extra flags for linking to the MKL library:
```
-I$MKLROOT/include  -L$MKLROOT/lib/intel64/  -lmkl_sycl -lmkl_core  -lmkl_sequential -lmkl_intel_ilp64
```
Where `MKLROOT` is an environment variable which is set during the initial set up of the oneAPI.

### Nvidia and AMD hardware
