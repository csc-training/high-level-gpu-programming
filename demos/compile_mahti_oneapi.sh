#!/bin/bash

args="$@"

icpx $args -fuse-ld=lld -std=c++20 -O3 -fsycl -fsycl-pstl-offload=gpu -fsycl-targets=nvptx64-nvidia-cuda -Xsycl-target-backend=nvptx64-nvidia-cuda --cuda-gpu-arch=sm_80 daxpy_stdpar.cpp -o stdpar_oneapi.x
icpx $args -fuse-ld=lld -std=c++20 -O3 -fsycl                         -fsycl-targets=nvptx64-nvidia-cuda -Xsycl-target-backend=nvptx64-nvidia-cuda --cuda-gpu-arch=sm_80 daxpy_sycl.cpp -o sycl_oneapi.x
icpx $args -fuse-ld=lld -std=c++20 -O3 -fsycl                         -fsycl-targets=nvptx64-nvidia-cuda -Xsycl-target-backend=nvptx64-nvidia-cuda --cuda-gpu-arch=sm_80 -DSYNC daxpy_sycl.cpp -o sycl_sync_oneapi.x
