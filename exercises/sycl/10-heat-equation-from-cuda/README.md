# Heat equation from CUDA to SYCL

This exercise is using the Intel DPC++ Compatibility Tool in oneAPI on Mahti to convert [the heat equation from CUDA](https://github.com/cschpc/heat-equation/tree/main/cuda) to SYCL.

## Test the CUDA code

Let's first start by testing the CUDA code:

    cd cuda
    make
    srun -p gputest --nodes=1 --ntasks-per-node=1 --cpus-per-task=1 --gres=gpu:a100:1 -t 00:05:00 ./heat.x

## Run the conversion tool

Let's execute the tool to convert CUDA to SYCL:

    cd cuda
    make clean
    intercept-build make  # runs make and creates compile_commands.json
    dpct -p compile_commands.json --gen-build-script --change-cuda-files-extension-only --sycl-file-extension=cpp --out-root=../dpct_output

This generates `dpct_output` from the `cuda` directory.

## Test the generated SYCL code

Let's try compiling the generated SYCL code using the generated makefile:

    cd dpct_output
    make -f Makefile.dpct

This fails. A fixed makefile is given in `dpct_sycl`.


## Test the generated SYCL code

Let's make a test run of the generated SYCL code with a fixed makefile:

    cd dpct_sycl
    make

    # Run on GPU
    srun -p gputest --nodes=1 --ntasks-per-node=1 --cpus-per-task=1 --gres=gpu:a100:1 -t 00:05:00 ./heat.x

    # Run on CPU
    srun -p test --nodes=1 --ntasks-per-node=1 --cpus-per-task=128 -t 00:05:00 ./heat.x


### Test with MPI parallelization

Let's run the code in parallel:

    srun -p gputest --nodes=1 --ntasks-per-node=2 --cpus-per-task=1 --gres=gpu:a100:2 -t 00:05:00 ./heat.x

This is really slow compared to the single-GPU run or crashes!

Note that in `setup.cpp` dpct tool produced a line `dpct::select_device(nodeRank)` with a warning
"DPCT1093:0: The "nodeRank" device may be not the one intended for use.".
See [the documentation of the warning code](https://www.intel.com/content/www/us/en/docs/dpcpp-compatibility-tool/developer-guide-reference/2025-0/dpct1093.html).

This line selects the device from *all* SYCL devices. Let's check the output of `sycl-ls`:

    srun -p gputest --nodes=1 --ntasks-per-node=1 --cpus-per-task=1 --gres=gpu:a100:2 -t 00:05:00 sycl-ls

    [opencl:cpu][opencl:0] Intel(R) OpenCL, AMD EPYC 7H12 64-Core Processor                 OpenCL 3.0 (Build 0) [2024.18.10.0.08_160000]
    [cuda:gpu][cuda:0] NVIDIA CUDA BACKEND, NVIDIA A100-SXM4-40GB 8.0 [CUDA 12.2]
    [cuda:gpu][cuda:1] NVIDIA CUDA BACKEND, NVIDIA A100-SXM4-40GB 8.0 [CUDA 12.2]

So, the device selection is not going to work correctly as sycl detects a mix of CPU and GPU devices.

As a workaround, we can restrict the SYCL devices in oneAPI to GPUs by:

    export ONEAPI_DEVICE_SELECTOR=*:gpu  # alternatively: export SYCL_DEVICE_FILTER=*:gpu

Resulting in:

    srun -p gputest --nodes=1 --ntasks-per-node=1 --cpus-per-task=1 --gres=gpu:a100:2 -t 00:05:00 sycl-ls

    [cuda:gpu] NVIDIA CUDA BACKEND, NVIDIA A100-SXM4-40GB 8.0 [CUDA 12.2]
    [cuda:gpu] NVIDIA CUDA BACKEND, NVIDIA A100-SXM4-40GB 8.0 [CUDA 12.2]

With this `ONEAPI_DEVICE_SELECTOR` environment variable active, the MPI-parallelized code works expectedly:

    srun -p gputest --nodes=1 --ntasks-per-node=2 --cpus-per-task=1 --gres=gpu:a100:2 -t 00:05:00 ./heat.x

Note though that MPI-parallelization with CPU devices (similar to hybrid MPI+OpenMP approach) doesn't work with such a trick.


## Convert the generated code to standard SYCL

In `sycl/` there is a reference code with `dpct` calls converted to standard SYCL.
Compare the changes to code in `dpct_sycl/`, but note alternative choices could be done too,
e.g., not using global queue.

Let's make a test run:

    cd sycl
    make

    # Run with MPI on 2 GPUs
    srun -p gputest --nodes=1 --ntasks-per-node=2 --cpus-per-task=1 --gres=gpu:a100:2 -t 00:05:00 ./heat.x

    # Run with MPI on 2 nodes using CPUs only
    srun -p test --nodes=2 --ntasks-per-node=1 --cpus-per-task=128 -t 00:05:00 ./heat.x

This SYCL code can be compiled and run with AdaptiveCpp too.


## Performance

Task: Compare the performance of CUDA and SYCL implementations.
For more reliable timing, use larger and longer calculations, e.g., `./heat.x 8000 8000 4000`.

