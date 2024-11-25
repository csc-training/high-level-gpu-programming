#!/bin/bash

[[ -z "$CONTAINER_EXEC" ]] || echo "CONTAINER_EXEC='$CONTAINER_EXEC'"

args="$@"

$CONTAINER_EXEC hipcc $args -O3 -std=c++20 --hipstdpar --hipstdpar-path=$HIPSTDPAR_PATH --offload-arch=gfx90a:xnack+ daxpy_stdpar.cpp -o stdpar.x
