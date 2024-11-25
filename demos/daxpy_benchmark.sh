#!/bin/bash
#SBATCH -J test
#SBATCH -p dev-g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-node=1
#SBATCH -t 0:05:00

outdpath="$1"
shift

echo "Output directory: $outdpath"
mkdir -p "$outdpath"
[[ -z "$CONTAINER_EXEC" ]] || echo "CONTAINER_EXEC='$CONTAINER_EXEC'"

nit=200

for version in "$@"; do
    echo $version
    f="$outdpath/daxpy_$version.dat"
    rm -f $f
    for n in $(python3 -c "print(' '.join([str(2**n) for n in range(10, 31)]))"); do
        $CONTAINER_EXEC ./$version.x $n $nit | tee -a $f
    done
done
