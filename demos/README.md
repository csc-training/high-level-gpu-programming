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

## LUMI

HIP and hipblas:

```bash
ml rocm/6.0.3

bash compile_lumi.sh -DBENCHMARK
sbatch -p dev-g --gpus-per-node=1 daxpy_benchmark.sh lumi-rocm603 cuda_hip cuda_hip_sync blas blas_sync
```

C++ stdpar:

```bash
export CONTAINER_EXEC="singularity exec /projappl/project_462000752/rocm_6.2.4_stdpar.sif"
export HIPSTDPAR_PATH="/opt/rocm-6.2.4/include/thrust/system/hip/hipstdpar"
export SINGULARITY_BIND="/pfs,/scratch,/projappl,/project,/flash,/appl"
export SINGULARITYENV_LC_ALL=C
export HSA_XNACK=1

bash compile_lumi_stdpar.sh -DBENCHMARK

sbatch -p dev-g --gpus-per-node=1 daxpy_benchmark.sh lumi-rocm624 stdpar
```

SYCL with oneAPI:

```bash
source /projappl/project_462000752/intel/oneapi/setvars.sh --include-intel-llvm
ml rocm/6.0.3

bash compile_lumi_sycl_oneapi.sh -DBENCHMARK
sbatch -p dev-g --gpus-per-node=1 daxpy_benchmark.sh lumi-rocm603 sycl_oneapi sycl_sync_oneapi
```

SYCL with AdaptiveCPP:

```bash
ml LUMI/24.03
ml partition/G
ml use /appl/local/csc/modulefiles
ml rocm/6.0.3
ml acpp/24.06.0

bash compile_lumi_sycl_acpp.sh -DBENCHMARK

# This crashes with large `nit` values in the benchmark script. Value nit=20 works.
sbatch -p dev-g --gpus-per-node=1 daxpy_benchmark.sh lumi-rocm603 sycl_acpp sycl_sync_acpp
```
