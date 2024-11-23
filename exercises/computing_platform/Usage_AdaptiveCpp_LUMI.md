## Usage

Load the modules needed:
```
module load LUMI/24.03
module load partition/G
module load rocm/6.0.3
/appl/lumi/SW/LUMI-24.03/G/EB/Boost/1.83.0-cpeGNU-24.03/lib64/
```
Compile `sycl` codes:
```
/projappl/project_462000752/ACPP/bin/acpp -O2 --hipsycl-targets="omp.accelerated;hip:gfx90a" <sycl_code>.cpp
```
