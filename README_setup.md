# Usage

## C++ stdpar on Mahti

Set up the environment:

    ml purge
    ml use /appl/opt/nvhpc/modulefiles
    ml nvhpc/24.3
    ml gcc/11.2.0
    export PATH=/appl/spack/v017/install-tree/gcc-8.5.0/binutils-2.37-ed6z3n/bin:$PATH

Compile:

    nvc++ -O4 -std=c++20 -stdpar=gpu -gpu=cc80 --gcc-toolchain=$(dirname $(which g++)) code.cpp

Run on one GPU:

    srun -A project_2012125 -p gputest --nodes=1 --ntasks-per-node=1 --cpus-per-task=1 --gres=gpu:a100:1 --time=00:15:00 ./a.out

## C++ stdpar on LUMI

Set up the environment with container:

    export CONTAINER_EXEC="singularity exec /projappl/project_462000752/rocm_6.2.4_stdpar.sif"
    export HIPSTDPAR_PATH="/opt/rocm-6.2.4/include/thrust/system/hip/hipstdpar"
    export SINGULARITY_BIND="/pfs,/scratch,/projappl,/project,/flash,/appl"
    export SINGULARITYENV_LC_ALL=C
    export HSA_XNACK=1

Compile:

   $CONTAINER_EXEC hipcc --hipstdpar --hipstdpar-path=$HIPSTDPAR_PATH --offload-arch=gfx90a:xnack+ code.cpp

Run on one GPU through container:

    srun -A project_462000752 -p dev-g --nodes=1 --ntasks-per-node=1 --cpus-per-task=1 --gpus-per-node=1 --time=00:15:00 $CONTAINER_EXEC ./a.out

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

## AdaptiveCpp on Mahti

Load the modules needed:
```
module purge
module use /scratch/project_2012125/cristian/spack/share/spack/modules/linux-rhel8-x86_64_v3/
module load hipsycl/24.06.0-gcc-10.4.0-4nny2ja 
```
Compile for cpu anf nvidia targets:

```
acpp -fuse-ld=lld -O3 -L/appl/spack/v020/install-tree/gcc-8.5.0/gcc-10.4.0-2oazqj/lib64/ --acpp-targets="omp.accelerated;cuda:sm_80" vector_add_buffer.cpp  vector_add_buffer.cpp
```
## AdaptiveCpp on LUMI

Set up the environment:

    module load LUMI/24.03
    module load partition/G
    module load rocm/6.0.3
    export PATH=/projappl/project_462000752/ACPP/bin/:$PATH
    export LD_LIBRARY_PATH=/appl/lumi/SW/LUMI-24.03/G/EB/Boost/1.83.0-cpeGNU-24.03/lib64/:$LD_LIBRARY_PATH
    export LD_PRELOAD=/opt/rocm-6.0.3/llvm/lib/libomp.so
    
Compile for amd and cpu targets:

    acpp -O3 --acpp-targets="omp.accelerated;hip:gfx90a" <sycl_code>.cpp
    
Run as an usual gpu program:

    srun -A project_462000752 -p dev-g --nodes=1 --ntasks-per-node=1 --cpus-per-task=1 --gpus-per-node=1 --time=00:15:00 ./a.out
    
# Installations

*Here are instructions how the modules used above were installed.*

## C++ stdpar on LUMI

Fetch rocm container (new enough ubuntu for system GCC to support C++20):

    export SINGULARITY_CACHEDIR=$PWD/singularity_cache
    singularity build --sandbox sandbox/ docker://docker.io/rocm/dev-ubuntu-24.04:6.2.4-complete

Add tbb to the container:

    # Download and extract
    mkdir tbb
    cd tbb
    for deb in \
        o/onetbb/libtbbbind-2-5_2021.11.0-2ubuntu2_amd64.deb \
        o/onetbb/libtbbmalloc2_2021.11.0-2ubuntu2_amd64.deb \
        o/onetbb/libtbb12_2021.11.0-2ubuntu2_amd64.deb \
        o/onetbb/libtbb-dev_2021.11.0-2ubuntu2_amd64.deb \
        h/hwloc/libhwloc15_2.10.0-1build1_amd64.deb \
    ; do
        wget http://mirrors.kernel.org/ubuntu/pool/universe/$deb
        deb=${deb##*/}
        ar x $deb data.tar.zst
        tar xvf data.tar.zst
        rm data.tar.zst
    done
    cd ..

    # Copy to sandbox
    chown -R $USER:$USER tbb/usr/
    chmod -R o+rX tbb/usr/
    cp -vpr tbb/usr sandbox/
    rm -r tbb/

Build the container:

    singularity build rocm_6.2.4_stdpar.sif sandbox/
    rm -r sandbox/

As an end result, the container image contains tbb as if it was installed with apt except that `/etc/ld.so.cache` doesn't contain libtbb (would require running `ldconfig` that `apt install` does).

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

## AdaptiveCpp on Mahti

```
module purge
git clone -c feature.manyFiles=true https://github.com/spack/spack.git
cd spack
 . share/spack/setup-env.sh
spack bootstrap now
spack compiler find
spack config add "modules:default:enable:[tcl]"
spack install lmod
$(spack location -i lmod)/lmod/lmod/init/bash
. share/spack/setup-env.sh
module load gcc/10.4.0
spack compiler find
```
Edit the recipe var/spack/repos/builtin/packages/hipsycl/package.py. Add "-DWITH_ACCELERATED_CPU:Bool=TRUE", after line 84. Edit the recipe var/spack/repos/builtin/packages/llvmpackage.py, remove all the versions higher than 18.
```
spack install hipsycl@24.06  %gcc@10.4.0 + cuda
```


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
