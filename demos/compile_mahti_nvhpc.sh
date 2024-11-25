#!/bin/bash

args="$@"

nvc++ $args -O4 -std=c++20 -stdpar=gpu -gpu=cc80 --gcc-toolchain=$(dirname $(which g++)) daxpy_stdpar.cpp -o stdpar.x
nvcc  $args -O3 -x cu -arch sm_80 --expt-relaxed-constexpr daxpy_cuda_hip.cpp -o cuda_hip.x
nvcc  $args -O3 -x cu -arch sm_80 --expt-relaxed-constexpr -DSYNC daxpy_cuda_hip.cpp -o cuda_hip_sync.x
nvcc  $args -O3 -x cu -arch sm_80 --expt-relaxed-constexpr daxpy_blas.cpp -lcublas -o blas.x
nvcc  $args -O3 -x cu -arch sm_80 --expt-relaxed-constexpr -DSYNC daxpy_blas.cpp -lcublas -o blas_sync.x
