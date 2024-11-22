# DAXPY demo

## Mahti

```bash
ml purge
ml use /appl/opt/nvhpc/modulefiles
ml nvhpc/24.3
ml gcc/11.2.0
export PATH=/appl/spack/v017/install-tree/gcc-8.5.0/binutils-2.37-ed6z3n/bin:$PATH

# test
bash compile_mahti.sh

srun -p gpusmall --ntasks-per-node=1 --gres=gpu:a100_1g.5gb:1 -t 0:05:00 ./stdpar.x
srun -p gpusmall --ntasks-per-node=1 --gres=gpu:a100_1g.5gb:1 -t 0:05:00 ./cuda_hip.x
srun -p gpusmall --ntasks-per-node=1 --gres=gpu:a100_1g.5gb:1 -t 0:05:00 ./blas.x

# benchmark
bash compile_mahti.sh -DBENCHMARK

sbatch daxpy_benchmark.sh mahti blas blas_sync cuda_hip cuda_hip_sync stdpar


# sycl

source /projappl/project_2012125/intel/oneapi/setvars.sh --include-intel-llvm
ml cuda/11.5.0 openmpi/4.1.2-cuda

bash compile_mahti_sycl.sh -DBENCHMARK
sbatch daxpy_benchmark.sh mahti sycl sycl_sync
```
