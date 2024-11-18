# DAXPY demo

## Mahti

```bash
ml purge
ml use /appl/opt/nvhpc/modulefiles
ml nvhpc/24.3
ml gcc/11.2.0

nvc++ -O4 -std=c++20 -stdpar=gpu -gpu=cc80 daxpy_stdpar.cpp --gcc-toolchain=$(dirname $(which g++)) -o stdpar.x
nvcc  -O3 -x cu -arch sm_80 --expt-relaxed-constexpr daxpy_cuda_hip.cpp -o cuda_hip.x
nvcc  -O3 -x cu -arch sm_80 --expt-relaxed-constexpr daxpy_blas.cpp -lcublas -o blas.x

srun -p gpusmall --ntasks-per-node=1 --gres=gpu:a100_1g.5gb:1 -t 0:05:00 ./stdpar.x
srun -p gpusmall --ntasks-per-node=1 --gres=gpu:a100_1g.5gb:1 -t 0:05:00 ./cuda_hip.x
srun -p gpusmall --ntasks-per-node=1 --gres=gpu:a100_1g.5gb:1 -t 0:05:00 ./blas.x


nvc++ -DBENCHMARK -O4 -std=c++20 -stdpar=gpu -gpu=cc80 daxpy_stdpar.cpp --gcc-toolchain=$(dirname $(which g++)) -o stdpar.x
nvcc  -DBENCHMARK -O3 -x cu -arch sm_80 --expt-relaxed-constexpr daxpy_cuda_hip.cpp -o cuda_hip.x
nvcc  -DBENCHMARK -O3 -x cu -arch sm_80 --expt-relaxed-constexpr daxpy_blas.cpp -lcublas -o blas.x
sbatch -p gputest --ntasks-per-node=1 --gres=gpu:a100:1 -t 0:05:00 submit.sh
```
