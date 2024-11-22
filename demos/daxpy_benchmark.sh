#!/bin/bash
#SBATCH -J test
#SBATCH -p gputest
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:a100:1
#SBATCH -t 0:05:00

outdpath="$1"

echo "Output directory: $outdpath"
mkdir -p "$outdpath"

for version in "$@"; do
    echo $version
    f="$outdpath/daxpy_$version.dat"
    rm $f
    for n in $(python3 -c "print(' '.join([str(2**n) for n in range(10, 31)]))"); do
        ./$version.x $n $nit | tee -a $f
    done
done
