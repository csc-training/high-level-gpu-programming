#!/bin/bash

[[ -z "$CONTAINER_EXEC" ]] || echo "CONTAINER_EXEC='$CONTAINER_EXEC'"

args="$@"

$CONTAINER_EXEC acpp $args -std=c++20 -O3 --acpp-stdpar --acpp-targets=hip:gfx90a -ltbb daxpy_stdpar.cpp -o stdpar_acpp.x
$CONTAINER_EXEC acpp $args -std=c++20 -O3 --acpp-targets=hip:gfx90a daxpy_sycl.cpp -o sycl_acpp.x
$CONTAINER_EXEC acpp $args -std=c++20 -O3 --acpp-targets=hip:gfx90a -DSYNC daxpy_sycl.cpp -o sycl_sync_acpp.x
