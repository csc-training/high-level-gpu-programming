#!/bin/bash

args="$@"

icpx  $args -fuse-ld=lld -std=c++20 -O3 -fsycl -fsycl-targets=nvptx64-nvidia-cuda,spir64_x86_64 -Xsycl-target-backend=nvptx64-nvidia-cuda --cuda-gpu-arch=sm_80 daxpy_sycl.cpp -o sycl.x
icpx  $args -fuse-ld=lld -std=c++20 -O3 -fsycl -fsycl-targets=nvptx64-nvidia-cuda,spir64_x86_64 -Xsycl-target-backend=nvptx64-nvidia-cuda --cuda-gpu-arch=sm_80 -DSYNC daxpy_sycl.cpp -o sycl_sync.x
