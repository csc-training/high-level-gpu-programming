## Introduction to GPU programming model
- execution model
- hardware to software mapping (CU --> block, ...)
- memory hierarchy
- streams
  
## cpp concepts
- classes, structures
- templates
- lambdas
- functors
- smart pointers
- namespace
- C++ stdpar
## SYCL ( https://enccs.github.io/sycl-workshop/ )
- queues:
    - device association
    - device info
    - behaivior
- expressing parallelism:
    - lambdas
    - range
    - nd_range
- memory:
    - buffers
    - USM
    - local share memory
- dependencies:
  
- internal profiling and debugging:
    
- cuda to sycl: CUDA heat equation  as an exercise
- installation on Mahti  and LUMI (hipsycl/AdaptiveCpp & (intel llvm))
## SYCL and outside libraries
- SYCL & MPI
- SYCL & CUDA/HIP libraries
## Kokkos Core
- Short intro to inline building and programming

## Possible exercises:
- selecting device in a node with several GPUs
- SYCL + mpi for computing *pi* (https://www.intel.com/content/www/us/en/developer/articles/technical/compile-and-run-mpi-programs-using-dpcpp-language.html)
- syclomatic for the heat equation
- calling other libraries (not necessarily portable)
- check examples from here https://github.com/oneapi-src/oneAPI-samples/tree/master/DirectProgramming/C%2B%2BSYCL
## Sources
- https://sycl.tech/
- https://link.springer.com/book/10.1007/978-1-4842-9691-2
- https://tinyurl.com/book-SYCL
- https://github.com/Apress/data-parallel-CPP/tree/main
- `git clone --recursive --branch sc23 https://github.com/codeplaysoftware/syclacademy.git` 
- https://www.youtube.com/watch?v=IeOnlBXTdn4&list=PL46sP9LM8GsyHAxj1k7MbWrv5f5SlMpIF
- https://enccs.github.io/sycl-workshop/
- https://enccs.github.io/gpu-programming/
- https://www.dropbox.com/scl/fo/bmgeetfr31i5vcz7j5sjw/h?rlkey=qg0tdg14tx65n0kxpxbz50bzo&dl=0
- https://www.intel.com/content/www/us/en/developer/tools/oneapi/training/migrate-from-cuda-to-cpp-with-sycl.html#gs.0zbdmk
- https://documentation.sigma2.no/code_development/guides_gpu.html#dev-guides-gpu
- https://fs.hlrs.de/projects/par/events/2023/intel-oneapi/
- https://fs.hlrs.de/projects/par/events/2023/GPU-AMD/ (some Kokkos)
- https://github.com/oneapi-src/oneAPI-samples/tree/master/DirectProgramming/C%2B%2BSYCL/Jupyter/oneapi-essentials-training
- https://github.com/oneapi-src/oneAPI-samples/tree/master/DirectProgramming/C%2B%2BSYCL/Jupyter/gpu-optimization-sycl-training
- https://github.com/oneapi-src/oneAPI-samples/tree/master/DirectProgramming/C%2B%2BSYCL/Jupyter/sycl-performance-portability-training
- https://github.com/oneapi-src/oneAPI-samples/tree/master/DirectProgramming/C%2B%2BSYCL/Jupyter/cuda-to-sycl-migration-training
- **heat equation** in cuda: https://github.com/cschpc/heat-equation/tree/main/cuda
- **heat equation** in sycl: https://github.com/ENCCS/gpu-programming/tree/main/content/examples/stencil/sycl
- **heat equation** at intel: https://www.intel.com/content/www/us/en/developer/articles/technical/solve-a-2d-heat-equation-using-data-parallel-c.html
- cuda training repository at CSC: https://github.com/csc-training/CUDA
- https://www.intel.com/content/www/us/en/developer/articles/technical/solve-a-2d-heat-equation-using-data-parallel-c.html
- https://doku.lrz.de/files/17826165/16942343/2/1686045904023/syclomatic-lrz-workshop+1.pdf
- hipstdpsar: https://gpuopen.com/learn/amd-lab-notes/amd-lab-notes-hipstdpar-readme/
