## Install from source:

Clone and switch to the appropriate version:
```
git clone https://github.com/AdaptiveCpp/AdaptiveCpp.git
git switch --detach  v0.9.4
```
Load the modules needed:
```
module load LUMI/22.08
module load partition/G
module load Boost/1.79.0-cpeCray-22.08
module load rocm/5.3.3
module load cce/16.0.1
```
Compile with both cpu and gpu (mi250x) acceleration:
```
cd /scratch/project_462000456/AdaptiveCpp
mkdir build
cd build
cmake -DCMAKE_INSTALL_PREFIX=/scratch/project_462000456/AdaptiveCpp/  -DROCM_PATH=$ROCM_PATH -DWITH_CPU_BACKEND=ON -DWITH_CUDA_BACKEND=OFF  -DWITH_ROCM_BACKEND=ON  -DDEFAULT_GPU_ARCH=gfx90a -DWITH_ACCELERATED_CPU=ON -DWITH_SSCP_COMPILER=OFF  -DWITH_OPENCL_BACKEND=OFF -DWITH_LEVEL_ZERO_BACKEND=OFF  ..
make -j 64
make install 
```

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
<install_path>/syclcc -O2 --hipsycl-targets="omp.accelerated;hip:gfx90a" <sycl_code>.cpp
```
