#!/bin/bash

args="$@"

icpx  $args -fuse-ld=lld -std=c++20 -O3 -fsycl -fsycl-pstl-offload=gpu -fsycl-targets=amdgcn-amd-amdhsa -Xsycl-target-backend=amdgcn-amd-amdhsa --offload-arch=gfx90a:xnack+ daxpy_stdpar.cpp -o stdpar_oneapi.x
#icpx  $args -fuse-ld=lld -std=c++20 -O3 -fsycl -fsycl-targets=amdgcn-amd-amdhsa,spir64_x86_64 -Xsycl-target-backend=amdgcn-amd-amdhsa --offload-arch=gfx90a daxpy_sycl.cpp -o sycl_oneapi.x
#icpx  $args -fuse-ld=lld -std=c++20 -O3 -fsycl -fsycl-targets=amdgcn-amd-amdhsa,spir64_x86_64 -Xsycl-target-backend=amdgcn-amd-amdhsa --offload-arch=gfx90a -DSYNC daxpy_sycl.cpp -o sycl_sync_oneapi.x
