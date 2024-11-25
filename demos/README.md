# DAXPY demo

## Mahti

CUDA and cublas:

```bash
ml cuda/11.5.0 openmpi/4.1.2-cuda

bash compile_mahti.sh -DBENCHMARK
sbatch -p gputest --gres=gpu:a100:1 daxpy_benchmark.sh mahti cuda_hip cuda_hip_sync blas blas_sync

srun -p gpusmall --ntasks-per-node=1 --gres=gpu:a100_1g.5gb:1 -t 0:05:00 ./cuda_hip.x
```

C++ stdpar:

```bash
ml purge
ml use /appl/opt/nvhpc/modulefiles
ml nvhpc/24.3
ml gcc/11.2.0
export PATH=/appl/spack/v017/install-tree/gcc-8.5.0/binutils-2.37-ed6z3n/bin:$PATH

bash compile_mahti_stdpar.sh -DBENCHMARK

sbatch -p gputest --gres=gpu:a100:1 daxpy_benchmark.sh mahti stdpar
```

SYCL with oneAPI:

```bash
source /projappl/project_2012125/intel/oneapi/setvars.sh --include-intel-llvm
ml cuda/11.5.0 openmpi/4.1.2-cuda

bash compile_mahti_sycl.sh -DBENCHMARK
sbatch -p gputest --gres=gpu:a100:1 daxpy_benchmark.sh mahti sycl sycl_sync
```
