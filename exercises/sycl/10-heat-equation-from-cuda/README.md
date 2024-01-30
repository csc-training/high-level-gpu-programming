# Heat equation from CUDA to SYCL

Using the Intel syclomatic tool to convert [the heat equation from CUDA](https://github.com/cschpc/heat-equation/tree/main/cuda) to SYCL.


Please ensure that the environment is correct:

```bash
. /scratch/project_2008874/cristian/intel/oneapi/setvars.sh --include-intel-llvm
ml cuda openmpi/4.1.2-cuda
```

## Test CUDA code

Let's make a test run of the CUDA code on Mahti

```bash
cd cuda
make clean
make

srun -p gputest --nodes=1 --ntasks-per-node=1 --cpus-per-task=1 --gres=gpu:a100:2 -1 00:15:00 ./heat_cuda
```

## Run syclomatic

```bash
cd cuda
make clean
intercept-build make  # creates compile_commands.json
dpct -p compile_commands.json --gen-build-script --out-root=../dpct_output
```

This creates `dpct_output` directory of syclomatic-converted `cuda` directory.

A reference code is given in `dpct_sycl` with an added Makefile.
Note that default file names have been updated.


## Test generated SYCL code

```bash
cd dpct_sycl
make clean
make

# Run on GPU
srun -p gputest --nodes=1 --ntasks-per-node=1 --cpus-per-task=1 --gres=gpu:a100:1 -t 00:15:00 ./heat_dpct

# Run on CPU
srun -p test --nodes=1 --ntasks-per-node=1 --cpus-per-task=64 -t 00:15:00 ./heat_dpct
```


