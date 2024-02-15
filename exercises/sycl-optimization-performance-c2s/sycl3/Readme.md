# SYCLomatic Exercise

We use one NVIDIA CUDA example, called  `Odd-Even MergeSort`, and follow the steps from here:\
https://github.com/oneapi-src/oneAPI-samples/tree/development/DirectProgramming/C%2B%2BSYCL/DenseLinearAlgebra/guided_odd_even_merge_sort_SYCLMigration

Because this example uses a Makefile and contains multiple CUDA source files, we're using the `intercept-build` tool.

![](pics/sycl.png)

## 1. Get the NVIDIA CUDA example

```bash
git clone https://github.com/NVIDIA/cuda-samples.git
```

## 2. Intercept CUDA code and convert


```bash
cd cuda-samples/Samples/2_Concepts_and_Techniques/sortingNetworks/
intercept-build make
c2s -p compile_commands.json --in-root ../../.. --gen-helper-function
```

## 3. How well did it go?

Check the conversion messages (purple warnings) and look at the respective files.\
-> This will still require some work to run.
<br>
Your are welcome to do this, but we will have a look at the solution
## 4. Get the Solution

```bash
git clone https://github.com/oneapi-src/oneAPI-samples.git
```

```bash
cp -fr oneAPI-samples/DirectProgramming/C++SYCL/DenseLinearAlgebra/guided_odd_even_merge_sort_SYCLMigration/* cuda-samples/Samples/2_Concepts_and_Techniques/sortingNetworks/
```

Feel free to inspect the modifications applied in the migrated files in the directory `cuda-samples/Samples/2_Concepts_and_Techniques/sortingNetworks/02_sycl_migrated/`. The originally converted files are at `cuda-samples/Samples/2_Concepts_and_Techniques/sortingNetworks/dpct_output/`.

## 5. Build and run for NVIDIA GPU

```bash
cd cuda-samples/Samples/2_Concepts_and_Techniques/sortingNetworks/
rm -r build
mkdir build && cd build
cmake -D NVIDIA_GPU=1 .. && make VERBOSE=1
```

## 6. Build and run for Intel/AMD CPU

```bash
cd cuda-samples/Samples/2_Concepts_and_Techniques/sortingNetworks/
rm -r build
mkdir build && cd build
cmake .. && make VERBOSE=1
```

# SIMPLE VECTORADD
```bash
cd cuda-samples/Samples/0_Introduction/vectorAdd
c2s vectorAdd.cu 
cd ~/cuda-samples/Samples/0_Introduction/vectorAdd/dpct_output
```
In there is the sycl file vectorAdd.dp.cpp

```C++
/**
 * Vector addition: C = A + B.
 *
 * This sample is a very basic sample that implements element by element
 * vector addition. It is the same as the sample illustrating Chapter 2
 * of the programming guide with some additions like error checking.
 */

#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <stdio.h>

// For the CUDA runtime routines (prefixed with "cuda_")

#include <helper_cuda.h>
#include <cmath>

/**
 * CUDA Kernel Device code
 *
 * Computes the vector addition of A and B into C. The 3 vectors have the same
 * number of elements numElements.
 */
void vectorAdd(const float *A, const float *B, float *C,
                          int numElements, const sycl::nd_item<3> &item_ct1) {
  int i = item_ct1.get_local_range(2) * item_ct1.get_group(2) +
          item_ct1.get_local_id(2);

  if (i < numElements) {
    C[i] = A[i] + B[i] + 0.0f;
  }
}

/**
 * Host main routine
 */
int main(void) try {
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.in_order_queue();
  // Error code to check return values for CUDA calls
  dpct::err0 err = 0;

  // Print the vector length to be used, and compute its size
  int numElements = 50000;
  size_t size = numElements * sizeof(float);
  printf("[Vector addition of %d elements]\n", numElements);

  // Allocate the host input vector A
  float *h_A = (float *)malloc(size);

  // Allocate the host input vector B
  float *h_B = (float *)malloc(size);

  // Allocate the host output vector C
  float *h_C = (float *)malloc(size);

  // Verify that allocations succeeded
  if (h_A == NULL || h_B == NULL || h_C == NULL) {
    fprintf(stderr, "Failed to allocate host vectors!\n");
    exit(EXIT_FAILURE);
  }

  // Initialize the host input vectors
  for (int i = 0; i < numElements; ++i) {
    h_A[i] = rand() / (float)RAND_MAX;
    h_B[i] = rand() / (float)RAND_MAX;
  }

  // Allocate the device input vector A
  float *d_A = NULL;
  err = DPCT_CHECK_ERROR(d_A = (float *)sycl::malloc_device(size, q_ct1));

  // Allocate the device input vector B
  float *d_B = NULL;
  err = DPCT_CHECK_ERROR(d_B = (float *)sycl::malloc_device(size, q_ct1));

  // Allocate the device output vector C
  float *d_C = NULL;
  err = DPCT_CHECK_ERROR(d_C = (float *)sycl::malloc_device(size, q_ct1));

  // Copy the host input vectors A and B in host memory to the device input
  // vectors in
  // device memory
  printf("Copy input data from the host memory to the CUDA device\n");
  err = DPCT_CHECK_ERROR(q_ct1.memcpy(d_A, h_A, size).wait());

  err = DPCT_CHECK_ERROR(q_ct1.memcpy(d_B, h_B, size).wait());

  // Launch the Vector Add CUDA Kernel
  int threadsPerBlock = 256;
  int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;
  printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid,
         threadsPerBlock);
  /*
  DPCT1049:0: The work-group size passed to the SYCL kernel may exceed the
  limit. To get the device limit, query info::device::max_work_group_size.
  Adjust the work-group size if needed.
  */
  q_ct1.parallel_for(
      sycl::nd_range<3>(sycl::range<3>(1, 1, blocksPerGrid) *
                            sycl::range<3>(1, 1, threadsPerBlock),
                        sycl::range<3>(1, 1, threadsPerBlock)),
      [=](sycl::nd_item<3> item_ct1) {
        vectorAdd(d_A, d_B, d_C, numElements, item_ct1);
      });
  /*
  DPCT1010:6: SYCL uses exceptions to report errors and does not use the error
  codes. The call was replaced with 0. You need to rewrite this code.
  */
  err = 0;

  // Copy the device result vector in device memory to the host result vector
  // in host memory.
  printf("Copy output data from the CUDA device to the host memory\n");
  err = DPCT_CHECK_ERROR(q_ct1.memcpy(h_C, d_C, size).wait());

  // Verify that the result vector is correct
  for (int i = 0; i < numElements; ++i) {
    if (fabs(h_A[i] + h_B[i] - h_C[i]) > 1e-5) {
      fprintf(stderr, "Result verification failed at element %d!\n", i);
      exit(EXIT_FAILURE);
    }
  }

  printf("Test PASSED\n");

  // Free device global memory
  err = DPCT_CHECK_ERROR(sycl::free(d_A, q_ct1));

  err = DPCT_CHECK_ERROR(sycl::free(d_B, q_ct1));

  err = DPCT_CHECK_ERROR(sycl::free(d_C, q_ct1));

  // Free host memory
  free(h_A);
  free(h_B);
  free(h_C);

  printf("Done\n");
  return 0;
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}
```

## We can compile this sycl code just as we did with all the other examples!
```bash
filename=$(basename  $1)
outname=${filename%.*}.x

echo running icpx -fuse-ld=lld  -fsycl -fsycl-targets=amdgcn-amd-amdhsa,spir64_x86_64 -Xsycl-target-backend=amdgcn-amd-amdhsa --offload-arch=gfx90a -I../../../../Common/  -o $outname $filename $2 $3 $4 $5

icpx -fuse-ld=lld -fsycl -fsycl-targets=nvptx64-nvidia-cuda,spir64_x86_64 -Xsycl-target-backend=nvptx64-nvidia-cuda --cuda-gpu-arch=sm_80 -I../../../../Common/ -o  $outname $filename $2 $3 $4 $5
```

can be invoked like:
```bash
./compile.sh vectorAdd.dp.cpp
```
which gives you vectorAdd.dp.x
```bash
#!/bin/bash
#SBATCH --job-name=usm
#SBATCH --account=project_2008874
#SBATCH --partition=gpusmall
#SBATCH --reservation=hlgp-gpu-f2024-thu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=00:01:00
#SBATCH --gres=gpu:a100:1

srun vectorAdd.dp.x
```

