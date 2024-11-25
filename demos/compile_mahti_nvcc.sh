#!/bin/bash

args="$@"

nvcc $args -O3 -x cu -arch sm_80 --expt-relaxed-constexpr daxpy_cuda_hip.cpp -o cuda_hip.x
nvcc $args -O3 -x cu -arch sm_80 --expt-relaxed-constexpr -DSYNC daxpy_cuda_hip.cpp -o cuda_hip_sync.x
nvcc $args -O3 -x cu -arch sm_80 --expt-relaxed-constexpr daxpy_blas.cpp -lcublas -o blas.x
nvcc $args -O3 -x cu -arch sm_80 --expt-relaxed-constexpr -DSYNC daxpy_blas.cpp -lcublas -o blas_sync.x
