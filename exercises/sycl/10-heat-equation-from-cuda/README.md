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

## Test with MPI parallelization

Let's run the code in parallel:

    srun -p gputest --nodes=1 --ntasks-per-node=2 --cpus-per-task=1 --gres=gpu:a100:2 -t 00:15:00 ./heat_dpct

This fails!

Note that in setup.cpp syclomatic produced a line `dpct::select_device(nodeRank)` with a warning
"DPCT1093:0: The "nodeRank" device may be not the one intended for use.".
See [the documentation of the warning code](https://oneapi-src.github.io/SYCLomatic/dev_guide/diagnostic_ref/dpct1093.html).

This line selects the device from *all* SYCL devices. Let's check the output of `sycl-ls`:

    srun -p gputest --nodes=1 --ntasks-per-node=1 --cpus-per-task=1 --gres=gpu:a100:2 -t 00:15:00 sycl-ls

    [opencl:acc:0] Intel(R) FPGA Emulation Platform for OpenCL(TM), Intel(R) FPGA Emulation Device OpenCL 1.2  [2023.16.12.0.12_195853.xmain-hotfix]
    [opencl:cpu:1] Intel(R) OpenCL, AMD EPYC 7H12 64-Core Processor                 OpenCL 3.0 (Build 0) [2023.16.12.0.12_195853.xmain-hotfix]
    [ext_oneapi_cuda:gpu:0] NVIDIA CUDA BACKEND, NVIDIA A100-SXM4-40GB 8.0 [CUDA 12.0]
    [ext_oneapi_cuda:gpu:1] NVIDIA CUDA BACKEND, NVIDIA A100-SXM4-40GB 8.0 [CUDA 12.0]

So, the device selection is not going to work correctly.

As a workaround, we can restrict the SYCL devices visible to oneAPI by:

    export ONEAPI_DEVICE_SELECTOR=*:gpu

Resulting in:

    srun -p gputest --nodes=1 --ntasks-per-node=1 --cpus-per-task=1 --gres=gpu:a100:2 -t 00:15:00 sycl-ls

    [ext_oneapi_cuda:gpu:0] NVIDIA CUDA BACKEND, NVIDIA A100-SXM4-40GB 8.0 [CUDA 12.0]
    [ext_oneapi_cuda:gpu:1] NVIDIA CUDA BACKEND, NVIDIA A100-SXM4-40GB 8.0 [CUDA 12.0]

With `ONEAPI_DEVICE_SELECTOR` environment variable active, the MPI-parallelized code works expectedly:

    srun -p gputest --nodes=1 --ntasks-per-node=2 --cpus-per-task=1 --gres=gpu:a100:2 -t 00:15:00 ./heat_dpct


## Convert the generated code to standard SYCL

The code generated with syclomatic is specific to oneAPI as it uses functions from oneAPI's `dpct` namespace.
We also found above that device selector logic needs to be updated to avoid
the need of `ONEAPI_DEVICE_SELECTOR` workaround.
(furthermore, MPI-parallelized CPU-execution doesn't work with `ONEAPI_DEVICE_SELECTOR`).

See `sycl/` for reference.


