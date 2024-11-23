## Install from source:

Clone and switch to the appropriate version:
```
git clone https://github.com/AdaptiveCpp/AdaptiveCpp.git
# git switch --detach  v0.9.4 # use this only if there are problems with the latest version
```
Load the modules needed:
```
module load LUMI/24.03
module load partition/G
module load rocm/6.0.3
```
Compile with both cpu and gpu (mi250x) acceleration:
```
git clone https://github.com/AdaptiveCpp/AdaptiveCpp.git
cd AdaptiveCpp
mkdir build
cd build
cmake -DCMAKE_INSTALL_PREFIX=/projappl/project_462000752/ACPP/  -DROCM_PATH=$ROCM_PATH -DWITH_CPU_BACKEND=ON -DWITH_CUDA_BACKEND=OFF  -DWITH_ROCM_BACKEND=ON -DACPP_TARGETS="gfx90a"  -DWITH_ACCELERATED_CPU=ON -DWITH_SSCP_COMPILER=OFF  -DWITH_OPENCL_BACKEND=OFF -DWITH_LEVEL_ZERO_BACKEND=OFF -DBOOST_ROOT=/appl/lumi/SW/LUMI-24.03/G/EB/Boost/1.83.0-cpeGNU-24.03/ ..
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
