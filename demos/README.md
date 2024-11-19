# DAXPY demo

## Mahti

```bash
ml purge
ml use /appl/opt/nvhpc/modulefiles
ml nvhpc/24.3
ml gcc/11.2.0
export PATH=/appl/spack/v017/install-tree/gcc-8.5.0/binutils-2.37-ed6z3n/bin:$PATH

nvc++ -O4 -std=c++20 -stdpar=gpu -gpu=cc80 --gcc-toolchain=$(dirname $(which g++)) daxpy_stdpar.cpp -o stdpar.x
nvcc  -O3 -x cu -arch sm_80 --expt-relaxed-constexpr daxpy_cuda_hip.cpp -o cuda_hip.x
nvcc  -O3 -x cu -arch sm_80 --expt-relaxed-constexpr daxpy_blas.cpp -lcublas -o blas.x

srun -p gpusmall --ntasks-per-node=1 --gres=gpu:a100_1g.5gb:1 -t 0:05:00 ./stdpar.x
srun -p gpusmall --ntasks-per-node=1 --gres=gpu:a100_1g.5gb:1 -t 0:05:00 ./cuda_hip.x
srun -p gpusmall --ntasks-per-node=1 --gres=gpu:a100_1g.5gb:1 -t 0:05:00 ./blas.x

nvc++ -DBENCHMARK -O4 -std=c++20 -stdpar=gpu -gpu=cc80 --gcc-toolchain=$(dirname $(which g++)) daxpy_stdpar.cpp -o stdpar.x
nvcc  -DBENCHMARK -O3 -x cu -arch sm_80 --expt-relaxed-constexpr daxpy_cuda_hip.cpp -o cuda_hip.x
nvcc  -DBENCHMARK -O3 -x cu -arch sm_80 --expt-relaxed-constexpr daxpy_blas.cpp -lcublas -o blas.x

sbatch -p gputest --ntasks-per-node=1 --gres=gpu:a100:1 -t 0:05:00 submit.sh


# sycl

source /projappl/project_2012125/intel/oneapi/setvars.sh --include-intel-llvm
ml cuda/11.5.0 openmpi/4.1.2-cuda

icpx  -DBENCHMARK -fuse-ld=lld -std=c++20 -O3 -fsycl -fsycl-targets=nvptx64-nvidia-cuda,spir64_x86_64 -Xsycl-target-backend=nvptx64-nvidia-cuda --cuda-gpu-arch=sm_80 daxpy_sycl.cpp -o sycl.x

```
