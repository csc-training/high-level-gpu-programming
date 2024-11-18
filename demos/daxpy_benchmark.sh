#!/bin/bash
#SBATCH -J test

nit=200
for version in blas cuda_hip stdpar; do
    echo $version
    f=daxpy_$version.dat
    rm $f
    for n in $(python3 -c "print(' '.join([str(2**n) for n in range(10, 31)]))"); do
        ./$version.x $n $nit | tee -a $f
    done
done
