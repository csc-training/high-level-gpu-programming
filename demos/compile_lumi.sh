#!/bin/bash

args="$@"

hipcc $args -O3 --offload-arch=gfx90a -Wno-unused-result daxpy_cuda_hip.cpp -o cuda_hip.x
hipcc $args -O3 --offload-arch=gfx90a -Wno-unused-result -DSYNC daxpy_cuda_hip.cpp -o cuda_hip_sync.x
hipcc $args -O3 --offload-arch=gfx90a -Wno-unused-result daxpy_blas.cpp -lhipblas -o blas.x
hipcc $args -O3 --offload-arch=gfx90a -Wno-unused-result -DSYNC daxpy_blas.cpp -lhipblas -o blas_sync.x
