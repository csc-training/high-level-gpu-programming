## Usage

```
module load LUMI/22.08
module load partition/G
module load rocm/5.3.3
module load cce/16.0.1
. /scratch/project_462000456//intel/oneapi/setvars.sh --include-intel-llvm
export MPICH_GPU_SUPPORT_ENABLED=1
```

```
icpx -fsycl -fsycl-targets=amdgcn-amd-amdhsa,spir64_x86_64 -Xsycl-target-backend=amdgcn-amd-amdhsa  --offload-arch=gfx90a  <sycl_code>.cpp
```

## Running
CPU
```
srun -p debug --exclusive  -n 1 --cpus-per-task=128  --time=00:05:00 --account=project_462000456 ./a.out
```
```
#SBATCH 
```
