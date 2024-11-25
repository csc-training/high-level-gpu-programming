#!/bin/bash

args="$@"

acpp $args -std=c++20 -O3 --acpp-targets=hip:gfx90a daxpy_sycl.cpp -o sycl_acpp.x
acpp $args -std=c++20 -O3 --acpp-targets=hip:gfx90a -DSYNC daxpy_sycl.cpp -o sycl_sync_acpp.x
