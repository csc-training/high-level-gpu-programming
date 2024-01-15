Load cuda 

```

module load cuda
```

Download and intall the oneapi basekit:

```
wget https://registrationcenter-download.intel.com/akdlm/IRC_NAS/163da6e4-56eb-4948-aba3-debcec61c064/l_BaseKit_p_2024.0.1.46_offline.sh 
chmod +x l_
./l_BaseKit_p_2024.0.1.46_offline.sh  -a -s --eula accept --download-cache /scratch/project_2008874/cristian/tttt/ --install-dir /scratch/project_2008874/cristian/intel/oneapi 
```

Now get the cuda plugin (the link below might be changed in the future):

Get and install the plug-in:
```
curl -LOJ "https://developer.codeplay.com/api/v1/products/download?product=oneapi&variant=nvidia&version=2024.0.1&filters[]=12.0&filters[]=linux"
 ./oneapi-for-nvidia-gpus-2024.0.1-cuda-12.0-linux.sh -y --extract-folder /scratch/project_2008874/cristian/tttt/ --install-dir /scratch/project_2008874/cristian/intel/oneapi
```
## Usage

Set the environments paths:

```
. /scratch/project_2008874/cristian/intel/oneapi/setvars.sh --include-intel-llvm
```
Compile for nvidia and cpu targets:

```
module load cuda
 clang++ -std=c++17 -O3 -fsycl -fsycl-targets=nvptx64-nvidia-cuda,spir64_x86_64 -Xsycl-target-backend=nvptx64-nvidia-cuda --cuda-gpu-arch=sm_80  <sycl_code>.cpp
```

Run as an usual gpu program:

```
srun  --time=00:15:00 --partition=gputest --account=project_2008874 --nodes=1 --ntasks-per-node=1  --cpus-per-task=1 --gres=gpu:a100:1  ./a.out
```

## he  Intel® DPC++ Compatibility Tool

The  Intel® DPC++ Compatibility Tool (syclomatic) is included in the oneAPI basekit. For migrating cuda to sycl use:
```
. /scratch/project_2008874/cristian/intel/oneapi/setvars.sh --include-intel-llvm
module load cuda
```

For example:
```
dpct --cuda-include-path=/scratch/project_2008874/spack/opt/spack/linux-rhel8-x86_64_v3/gcc-11.2.0/cuda-11.5.0-gmjjetscy7qwcwrrmuoqsujhdqkkyjss/include --in-root=./ src/vector_add.cu
```