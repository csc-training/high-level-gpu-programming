#!/bin/bash

[[ -z "$CONTAINER_EXEC" ]] || echo "CONTAINER_EXEC='$CONTAINER_EXEC'"

args="$@"

$CONTAINER_EXEC hipcc $args -O3 --offload-arch=gfx90a -Wno-unused-result daxpy_cuda_hip.cpp -o cuda_hip.x
$CONTAINER_EXEC hipcc $args -O3 --offload-arch=gfx90a -Wno-unused-result -DSYNC daxpy_cuda_hip.cpp -o cuda_hip_sync.x
$CONTAINER_EXEC hipcc $args -O3 --offload-arch=gfx90a -Wno-unused-result daxpy_blas.cpp -lhipblas -o blas.x
$CONTAINER_EXEC hipcc $args -O3 --offload-arch=gfx90a -Wno-unused-result -DSYNC daxpy_blas.cpp -lhipblas -o blas_sync.x
