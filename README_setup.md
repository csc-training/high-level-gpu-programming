# Usage

## OneAPI on Mahti

Set up the environment:

    source /projappl/project_2012125/intel/oneapi/setvars.sh --include-intel-llvm
    ml cuda/11.5.0 openmpi/4.1.2-cuda

Compile for nvidia and cpu targets:

    icpx -fuse-ld=lld -std=c++20 -O3 -fsycl -fsycl-targets=nvptx64-nvidia-cuda,spir64_x86_64 -Xsycl-target-backend=nvptx64-nvidia-cuda --cuda-gpu-arch=sm_80 sycl_code.cpp

Run as an usual gpu program:

    srun -A project_2012125 -p gputest --nodes=1 --ntasks-per-node=1 --cpus-per-task=1 --gres=gpu:a100:1 --time=00:15:00 ./a.out

## OneAPI on LUMI

Set up the environment:

    source /projappl/project_462000752/intel/oneapi/setvars.sh --include-intel-llvm
    ml rocm/6.0.3
    export MPICH_GPU_SUPPORT_ENABLED=1

Compile for amd and cpu targets:

    icpx -fuse-ld=lld -std=c++20 -O3 -fsycl -fsycl-targets=amdgcn-amd-amdhsa,spir64_x86_64 -Xsycl-target-backend=amdgcn-amd-amdhsa --offload-arch=gfx90a sycl_code.cpp

Run as an usual gpu program:

    srun -A project_462000752 -p dev-g --nodes=1 --ntasks-per-node=1 --cpus-per-task=1 --gpus-per-node=1 --time=00:15:00 ./a.out

## AdaptiveCpp on LUMI

Set up the environment:

    module load LUMI/24.03
    module load partition/G
    module load rocm/6.0.3
    export PATH=/projappl/project_462000752/ACPP/bin/:$PATH
    export LD_LIBRARY_PATH=/appl/lumi/SW/LUMI-24.03/G/EB/Boost/1.83.0-cpeGNU-24.03/lib64/:$LD_LIBRARY_PATH
    export LD_PRELOAD=/opt/rocm-6.0.3/llvm/lib/libomp.so
    
Compile for amd and cpu targets:

    acpp -O2 --acpp-targets="omp.accelerated;hip:gfx90a" <sycl_code>.cpp
    
Run as an usual gpu program:

    srun -A project_462000752 -p dev-g --nodes=1 --ntasks-per-node=1 --cpus-per-task=1 --gpus-per-node=1 --time=00:15:00 ./a.out
    
# Installations

*Here are instructions how the modules used above were installed.*

## OneAPI on Mahti

Download [Intel oneAPI base toolkit](https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit-download.html?packages=oneapi-toolkit&oneapi-toolkit-os=linux&oneapi-lin=offline):

    wget https://registrationcenter-download.intel.com/akdlm/IRC_NAS/96aa5993-5b22-4a9b-91ab-da679f422594/intel-oneapi-base-toolkit-2025.0.0.885_offline.sh

Install:

    sh ./intel-oneapi-base-toolkit-2025.0.0.885_offline.sh -a --silent --cli --eula accept --download-cache $SCRATCH/$USER/oneapi_tmp --install-dir $PROJAPPL/intel/oneapi

Get [Codeplay oneAPI for NVIDIA GPUs](https://developer.codeplay.com/products/oneapi/nvidia/2025.0.0/guides/get-started-guide-nvidia#installation):

    curl -LOJ "https://developer.codeplay.com/api/v1/products/download?product=oneapi&variant=nvidia&version=2025.0.0&filters[]=12.0&filters[]=linux"

Install:

    sh ./oneapi-for-nvidia-gpus-2025.0.0-cuda-12.0-linux.sh -y --extract-folder $SCRATCH/$USER/oneapi_tmp --install-dir $PROJAPPL/intel/oneapi

## OneAPI on LUMI

Download [Intel oneAPI base toolkit](https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit-download.html?packages=oneapi-toolkit&oneapi-toolkit-os=linux&oneapi-lin=offline):

    wget https://registrationcenter-download.intel.com/akdlm/IRC_NAS/96aa5993-5b22-4a9b-91ab-da679f422594/intel-oneapi-base-toolkit-2025.0.0.885_offline.sh

Install:

    sh ./intel-oneapi-base-toolkit-2025.0.0.885_offline.sh -a --silent --cli --eula accept --download-cache $SCRATCH/$USER/oneapi_tmp --install-dir $PROJAPPL/intel/oneapi

Get [Codeplay oneAPI for AMD GPUs](https://developer.codeplay.com/products/oneapi/amd/2025.0.0/guides/get-started-guide-amd#installation):

    curl -LOJ "https://developer.codeplay.com/api/v1/products/download?product=oneapi&variant=amd&version=2025.0.0&filters[]=6.0.2&filters[]=linux"

Install:

    sh ./oneapi-for-amd-gpus-2025.0.0-rocm-6.0.2-linux.sh -y --extract-folder $SCRATCH/$USER/oneapi_tmp --install-dir $PROJAPPL/intel/oneapi

## AdaptiveCpp on LUMI


Load the modules needed:
```
module load LUMI/24.03
module load partition/G
module load rocm/6.0.3
```
Clone repository and Compile with both cpu and rocm support:
```
git clone https://github.com/AdaptiveCpp/AdaptiveCpp.git
cd AdaptiveCpp
mkdir build
cd build
cmake -DCMAKE_INSTALL_PREFIX=/projappl/project_462000752/ACPP/  -DROCM_PATH=$ROCM_PATH -DWITH_CPU_BACKEND=ON -DWITH_CUDA_BACKEND=OFF  -DWITH_ROCM_BACKEND=ON -DACPP_TARGETS="gfx90a"  -DWITH_ACCELERATED_CPU=ON -DWITH_SSCP_COMPILER=OFF  -DWITH_OPENCL_BACKEND=OFF -DWITH_LEVEL_ZERO_BACKEND=OFF -DBOOST_ROOT=/appl/lumi/SW/LUMI-24.03/G/EB/Boost/1.83.0-cpeGNU-24.03/ ..
make -j 64 
make install 
```
