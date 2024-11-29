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

### C++ stdpar

```bash
ml purge
ml use /appl/opt/nvhpc/modulefiles
ml nvhpc-hpcx-cuda12/24.3
ml gcc/11.2.0
export PATH=/appl/spack/v017/install-tree/gcc-8.5.0/binutils-2.37-ed6z3n/bin:$PATH

mpicxx -O4 -std=c++20 -stdpar=gpu -gpu=cc80 --gcc-toolchain=$(dirname $(which g++)) mpi_stdpar.cpp -o stdpar.x
srun -p gputest --nodes=1 --ntasks-per-node=2 --gres=gpu:a100:2 -t 00:05:00 ./stdpar.x
```

### OneAPI

```bash
source /projappl/project_2012125/intel/oneapi/setvars.sh --include-intel-llvm
ml cuda/11.5.0 openmpi/4.1.2-cuda

icpx -fuse-ld=lld -std=c++17 -O3 -fsycl -fsycl-targets=nvptx64-nvidia-cuda -Xsycl-target-backend=nvptx64-nvidia-cuda --cuda-gpu-arch=sm_80 `mpicxx --showme:compile` `mpixx --showme:link` mpi_sycl_usm.cpp -o sycl.x
srun -p gputest --nodes=1 --ntasks-per-node=2 --gres=gpu:a100:2 -t 00:05:00 ./sycl_usm.x

icpx -DSYCL_BACKEND_CUDA -fuse-ld=lld -std=c++17 -O3 -fsycl -fsycl-targets=nvptx64-nvidia-cuda -Xsycl-target-backend=nvptx64-nvidia-cuda --cuda-gpu-arch=sm_80 `mpicxx --showme:compile` `mpicxx --showme:link` mpi_sycl_buf.cpp -o sycl_buf.x
srun -p gputest --nodes=1 --ntasks-per-node=2 --gres=gpu:a100:2 -t 00:05:00 ./sycl_buf.x
```

See also [MPI guide](https://developer.codeplay.com/products/oneapi/nvidia/2025.0.0/guides/MPI-guide).

### Kokkos

```bash
ml cuda/11.5.0 openmpi/4.1.2-cuda

export KOKKOS_PATH=$SCRATCH/$USER/kokkos
make -f Makefile.kokkos TARGET=mahtig -j16
srun -p gputest --nodes=1 --ntasks-per-node=2 --gres=gpu:a100:2 -t 00:05:00 ./kokkos.x

# Examples using Kokkos settings
srun -p gputest --nodes=1 --ntasks-per-node=2 --gres=gpu:a100:2 -t 00:05:00 ./kokkos.x --kokkos-map-device-id-by=mpi_rank
srun -p gputest --nodes=1 --ntasks-per-node=2 --gres=gpu:a100:2 -t 00:05:00 ./kokkos.x --kokkos-device-id=0   # you can use e.g. $SLURM_PROCID in a sbatch script here
```

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

icpx -DSYCL_BACKEND_HIP -fuse-ld=lld -std=c++20 -O3 -fsycl -fsycl-targets=amdgcn-amd-amdhsa -Xsycl-target-backend=amdgcn-amd-amdhsa --offload-arch=gfx90a `CC --cray-print-opts=all` mpi_sycl_buf.cpp -o sycl_buf.x
srun -p dev-g --nodes=1 --ntasks-per-node=2 --gpus-per-node=2 -t 00:05:00 ./sycl_buf.x
```

See also [MPI guide](https://developer.codeplay.com/products/oneapi/amd/2025.0.0/guides/MPI-guide).


### AdaptiveCpp

```bash
ml LUMI/24.03
ml partition/G
ml use /appl/local/csc/modulefiles
ml rocm/6.0.3
ml acpp/24.06.0
export MPICH_GPU_SUPPORT_ENABLED=1

acpp -std=c++20 -O3 --acpp-targets=hip:gfx90a `CC --cray-print-opts=all` mpi_sycl_usm.cpp -o sycl_usm.x
srun -p dev-g --nodes=1 --ntasks-per-node=2 --gpus-per-node=2 -t 00:05:00 ./sycl_usm.x

```

### Kokkos

```bash
ml craype-x86-trento craype-accel-amd-gfx90a rocm/6.0.3
export MPICH_GPU_SUPPORT_ENABLED=1

export KOKKOS_PATH=$SCRATCH/$USER/kokkos
make -f Makefile.kokkos TARGET=lumig -j16
srun -p dev-g --nodes=1 --ntasks-per-node=2 --gpus-per-node=2 -t 00:05:00 ./kokkos.x

# Examples using Kokkos settings
srun -p dev-g --nodes=1 --ntasks-per-node=2 --gpus-per-node=2 -t 00:05:00 ./kokkos.x --kokkos-map-device-id-by=mpi_rank
srun -p dev-g --nodes=1 --ntasks-per-node=2 --gpus-per-node=2 -t 00:05:00 ./kokkos.x --kokkos-device-id=0   # you can use e.g. $SLURM_PROCID in a sbatch script here
```
