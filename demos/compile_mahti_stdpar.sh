#!/bin/bash

args="$@"

nvc++ $args -O4 -std=c++20 -stdpar=gpu -gpu=cc80 --gcc-toolchain=$(dirname $(which g++)) daxpy_stdpar.cpp -o stdpar.x
