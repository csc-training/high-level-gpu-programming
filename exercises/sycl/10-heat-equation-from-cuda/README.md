# [Heat equation](https://enccs.github.io/openmp-gpu/miniapp/) from CUDA to SYCL

Using the Intel syclomatic tool to convert [the heat equation from CUDA](https://github.com/cschpc/heat-equation/tree/main/cuda) to SYCL.

Before starting, please ensure that the environment is correct (Mahti):

    . /projappl/project_2008874/intel/oneapi/setvars.sh --include-intel-llvm
    ml cuda/11.5.0 openmpi/4.1.2-cuda


## Test the CUDA code

Let's make a test run of the CUDA code on Mahti:

    cd cuda
    make

    srun --account=project_2008874 -p gputest --nodes=1 --ntasks-per-node=1 --cpus-per-task=1 --gres=gpu:a100:1 -t 00:05:00 ./heat


## Run syclomatic

Let's execute `dpct` to convert CUDA to SYCL:

    cd cuda
    make clean
    intercept-build make  # creates compile_commands.json
    dpct -p compile_commands.json --gen-build-script --out-root=../dpct_output

This creates syclomatic-converted `dpct_output` directory of the `cuda` directory.

A reference output is given in `dpct_sycl` with an added Makefile and some files renamed.


## Test the generated SYCL code

Let's make a test run of the generated SYCL code on Mahti:

    cd dpct_sycl
    make

    # Run on GPU
    srun -p gputest --nodes=1 --ntasks-per-node=1 --cpus-per-task=1 --gres=gpu:a100:1 -t 00:05:00 ./heat

    # Run on CPU
    srun -p test --nodes=1 --ntasks-per-node=1 --cpus-per-task=128 -t 00:05:00 ./heat


### Test with MPI parallelization

Let's run the code in parallel:

    srun -p gputest --nodes=1 --ntasks-per-node=2 --cpus-per-task=1 --gres=gpu:a100:2 -t 00:05:00 ./heat

This fails!

Note that in `setup.cpp` dpct tool produced a line `dpct::select_device(nodeRank)` with a warning
"DPCT1093:0: The "nodeRank" device may be not the one intended for use.".
See [the documentation of the warning code](https://oneapi-src.github.io/SYCLomatic/dev_guide/diagnostic_ref/dpct1093.html).

This line selects the device from *all* SYCL devices. Let's check the output of `sycl-ls`:

    srun -p gputest --nodes=1 --ntasks-per-node=1 --cpus-per-task=1 --gres=gpu:a100:2 -t 00:05:00 sycl-ls

    [opencl:acc:0] Intel(R) FPGA Emulation Platform for OpenCL(TM), Intel(R) FPGA Emulation Device OpenCL 1.2  [2023.16.12.0.12_195853.xmain-hotfix]
    [opencl:cpu:1] Intel(R) OpenCL, AMD EPYC 7H12 64-Core Processor                 OpenCL 3.0 (Build 0) [2023.16.12.0.12_195853.xmain-hotfix]
    [ext_oneapi_cuda:gpu:0] NVIDIA CUDA BACKEND, NVIDIA A100-SXM4-40GB 8.0 [CUDA 12.0]
    [ext_oneapi_cuda:gpu:1] NVIDIA CUDA BACKEND, NVIDIA A100-SXM4-40GB 8.0 [CUDA 12.0]

So, the device selection is not going to work correctly.

As a workaround, we can restrict the SYCL devices in oneAPI to GPUs by:

    export ONEAPI_DEVICE_SELECTOR=*:gpu  # alternatively: export SYCL_DEVICE_FILTER=*:gpu

Resulting in:

    srun -p gputest --nodes=1 --ntasks-per-node=1 --cpus-per-task=1 --gres=gpu:a100:2 -t 00:05:00 sycl-ls

    [ext_oneapi_cuda:gpu:0] NVIDIA CUDA BACKEND, NVIDIA A100-SXM4-40GB 8.0 [CUDA 12.0]
    [ext_oneapi_cuda:gpu:1] NVIDIA CUDA BACKEND, NVIDIA A100-SXM4-40GB 8.0 [CUDA 12.0]

With this `ONEAPI_DEVICE_SELECTOR` environment variable active, the MPI-parallelized code works expectedly:

    srun -p gputest --nodes=1 --ntasks-per-node=2 --cpus-per-task=1 --gres=gpu:a100:2 -t 00:05:00 ./heat

Note though that MPI-parallelization with CPU devices (similar to hybrid MPI+OpenMP approach) doesn't work with such a trick.


## Convert the generated code to standard SYCL

The code generated with syclomatic is specific to oneAPI as it uses functions from oneAPI's `dpct` namespace.
We also found above that generated device selector logic is not generally suitable.

Please have a look at `sycl/` for a reference SYCL code with `dpct` calls converted to standard SYCL.
Compare the changes to code in `dpct_sycl/`.

Let's make a test run:

    cd sycl
    make

    # Run with MPI on 2 GPUs
    srun -p gputest --nodes=1 --ntasks-per-node=2 --cpus-per-task=1 --gres=gpu:a100:2 -t 00:05:00 ./heat

    # Run with MPI on 2 nodes using CPUs only
    srun -p test --nodes=2 --ntasks-per-node=1 --cpus-per-task=128 -t 00:05:00 ./heat

This SYCL code can be compiled and run with AdaptiveCpp too.


## Performance

Task: Compare the performance of CUDA and SYCL implementations.
For more reliable timing, use larger and longer calculations, e.g., `./heat 8000 8000 4000`.

