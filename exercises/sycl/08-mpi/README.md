# MPI send and receive with GPUs

Here are example codes with MPI + {CUDA,HIP,SYCL,stdpar} for two MPI tasks.
Task 0 fills an array with values and sends the array to task 1 that receives the values.
Finally both tasks print their arrays.

These codes are minimal examples for associating MPI tasks with available GPUs and performing MPI communication using GPU-aware MPI.

## Mahti

### CUDA

```bash
ml cuda/11.5.0 openmpi/4.1.2-cuda

nvcc -std=c++17 -O3 -x cu -arch sm_80 --expt-relaxed-constexpr -Xcompiler "`mpicxx --showme:compile`" -lmpi mpi_cuda_hip.cpp -o cuda_hip.x
srun -p gputest --nodes=1 --ntasks-per-node=2 --gres=gpu:a100:2 -t 00:05:00 ./cuda_hip.x
```

### OneAPI

```bash
source /projappl/project_2012125/intel/oneapi/setvars.sh --include-intel-llvm
ml cuda/11.5.0 openmpi/4.1.2-cuda

icpx -fuse-ld=lld -std=c++17 -O3 -fsycl -fsycl-targets=nvptx64-nvidia-cuda -Xsycl-target-backend=nvptx64-nvidia-cuda --cuda-gpu-arch=sm_80 `mpicxx --showme:compile` `mpixx --showme:link` mpi_sycl_usm.cpp -o sycl.x
srun -p gputest --nodes=1 --ntasks-per-node=2 --gres=gpu:a100:2 -t 00:05:00 ./sycl_usm.x

icpx -fuse-ld=lld -std=c++17 -O3 -fsycl -fsycl-targets=nvptx64-nvidia-cuda -Xsycl-target-backend=nvptx64-nvidia-cuda --cuda-gpu-arch=sm_80 `mpicxx --showme:compile` `mpicxx --showme:link` mpi_sycl_buf.cpp -o sycl_buf.x
srun -p gputest --nodes=1 --ntasks-per-node=2 --gres=gpu:a100:2 -t 00:05:00 ./sycl_buf.x
```

See also [MPI guide](https://developer.codeplay.com/products/oneapi/nvidia/2025.0.0/guides/MPI-guide).

## LUMI

### HIP

```bash
ml craype-x86-trento craype-accel-amd-gfx90a rocm/6.0.3
export MPICH_GPU_SUPPORT_ENABLED=1

CC -std=c++20 -x hip -Wno-unused-result mpi_cuda_hip.cpp -o cuda_hip.x
srun -p dev-g --nodes=1 --ntasks-per-node=2 --gpus-per-node=2 -t 00:05:00 ./cuda_hip.x
```

### OneAPI

```bash
source /projappl/project_462000752/intel/oneapi/setvars.sh --include-intel-llvm
ml craype-x86-trento craype-accel-amd-gfx90a rocm/6.0.3
export MPICH_GPU_SUPPORT_ENABLED=1

icpx -fuse-ld=lld -std=c++20 -O3 -fsycl -fsycl-targets=amdgcn-amd-amdhsa -Xsycl-target-backend=amdgcn-amd-amdhsa --offload-arch=gfx90a `CC --cray-print-opts=all` mpi_sycl_usm.cpp -o sycl_usm.x
srun -p dev-g --nodes=1 --ntasks-per-node=2 --gpus-per-node=2 -t 00:05:00 ./sycl_usm.x

icpx -fuse-ld=lld -std=c++20 -O3 -fsycl -fsycl-targets=amdgcn-amd-amdhsa -Xsycl-target-backend=amdgcn-amd-amdhsa --offload-arch=gfx90a `CC --cray-print-opts=all` mpi_sycl_buf.cpp -o sycl_buf.x
srun -p dev-g --nodes=1 --ntasks-per-node=2 --gpus-per-node=2 -t 00:05:00 ./sycl_buf.x
```

See also [MPI guide](https://developer.codeplay.com/products/oneapi/amd/2025.0.0/guides/MPI-guide).
