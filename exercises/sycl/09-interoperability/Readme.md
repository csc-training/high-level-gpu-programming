# Using 3rd party libraries. 
Third-party libraries are essential for GPU programming, offering optimized functions and algorithms tailored for specific tasks while abstracting low-level hardware details. This abstraction enables developers to focus on algorithm design and application development, enhancing productivity and efficiency.

For Intel hardware, the [Intel® oneAPI Math Kernel Library](https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl.html) (oneMKL) provides a comprehensive set of low-level routines for math operations. It is highly optimized and extensively parallelized, supporting Intel CPUs and GPUs, with a SYCL interface for integration into SYCL codes.

Nvidia hardware users can leverage the [CUDA toolkit](https://developer.nvidia.com/cuda-toolkit), offering a suite of highly optimized libraries such as cuBLAS and cuFFT. These libraries enhance performance for various computational tasks on Nvidia GPUs.

Similarly, for AMD hardware, the [ROCm/HIP](https://rocm.docs.amd.com/projects/HIP/en/latest/) libraries provide optimized routines for GPU-accelerated computing, enabling efficient utilization of AMD GPUs.

At the moment there is no unified interface to select automatically the needed library based on the device with which each queue is associated. Furthermore some libraries such as cu/hipBlas or cu/hipFFT require a gpu stream submit the work in an asynchronous way and integrate the call with previous ans subsenquent kernels. However in general a SYCL queue is not necesseraly associated with given stream. The SYCL standard defines the queue behaviour, but it is up to the `SYCL implementation` how that is mapped to the `cuda/hip stream`. 

## Matrix-Matrix multiplication example
