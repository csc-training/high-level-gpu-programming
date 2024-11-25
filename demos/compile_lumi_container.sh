#!/bin/bash

args="$@"
$CONTAINER_EXEC hipcc $args -O3 -std=c++20 --hipstdpar --hipstdpar-path=$HIPSTDPAR_PATH --offload-arch=gfx90a:xnack+ daxpy_stdpar.cpp -o stdpar.x
$CONTAINER_EXEC hipcc $args -O3 --offload-arch=gfx90a:xnack+ -Wno-unused-result daxpy_cuda_hip.cpp -o cuda_hip.x
$CONTAINER_EXEC hipcc $args -O3 --offload-arch=gfx90a:xnack+ -Wno-unused-result -DSYNC daxpy_cuda_hip.cpp -o cuda_hip_sync.x
$CONTAINER_EXEC hipcc $args -O3 --offload-arch=gfx90a:xnack+ -Wno-unused-result daxpy_blas.cpp -lhipblas -o blas.x
$CONTAINER_EXEC hipcc $args -O3 --offload-arch=gfx90a:xnack+ -Wno-unused-result -DSYNC daxpy_blas.cpp -lhipblas -o blas_sync.x

