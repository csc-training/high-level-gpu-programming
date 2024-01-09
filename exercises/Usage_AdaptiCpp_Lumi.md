## Usage

Load the modules needed:
```
module load LUMI/22.08
module load partition/G
module load Boost/1.79.0-cpeCray-22.08
module load rocm/5.3.3
module load cce/16.0.1
```
Compile `sycl` codes:
```
/scratch/project_462000456/AdaptiveCpp/bin/syclcc -O2 --hipsycl-targets="omp.accelerated;hip:gfx90a" <sycl_code>.cpp
```
