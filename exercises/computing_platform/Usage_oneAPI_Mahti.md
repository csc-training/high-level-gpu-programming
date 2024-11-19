# Using oneAPI on Mahti

Set the environments paths:

    source /projappl/project_2012125/intel/oneapi/setvars.sh --include-intel-llvm
    ml cuda/11.5.0 openmpi/4.1.2-cuda

Compile for nvidia and cpu targets:

    icpx -fuse-ld=lld -std=c++17 -O3 -fsycl -fsycl-targets=nvptx64-nvidia-cuda,spir64_x86_64 -Xsycl-target-backend=nvptx64-nvidia-cuda --cuda-gpu-arch=sm_80 sycl_code.cpp

Run as an usual gpu program:

    srun --partition=gputest --account=project_2012125 --nodes=1 --ntasks-per-node=1 --cpus-per-task=1 --gres=gpu:a100:1 --time=00:15:00 ./a.out


## The Intel® DPC++ Compatibility Tool

The Intel® DPC++ Compatibility Tool (syclomatic) is included in the oneAPI basekit. For migrating cuda to sycl use (for example):

    dpct --in-root=./ src/vector_add.cu

See [the heat equation exercise](sycl/10-heat-equation-from-cuda/) for a complete example.
