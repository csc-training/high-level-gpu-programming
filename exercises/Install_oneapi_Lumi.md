


Download and intall the oneapi basekit:

```
wget https://registrationcenter-download.intel.com/akdlm/IRC_NAS/163da6e4-56eb-4948-aba3-debcec61c064/l_BaseKit_p_2024.0.1.46_offline.sh
 
module load LUMI/22.08
module load partition/G
module load rocm/5.3.3
module load cce/16.0.1
chmod +x l_/l_BaseKit_p_2024.0.1.46_offline.sh
./l_BaseKit_p_2024.0.1.46_offline.sh  -a -s --eula accept --download-cache /scratch/project_462000456/cristian/tttt/ --install-dir /scratch/project_462000456/intel/oneapi 
```

Now get the hip plugin (the link below might be changed in the future):

Get and install the plug-in:
```
curl -LOJ "https://developer.codeplay.com/api/v1/products/download?product=oneapi&variant=amd&version=2024.0.1&filters[]=5.4.3&filters[]=linux"
sh oneapi-for-amd-gpus-2024.0.1-rocm-5.4.3-linux.sh -y --extract-folder /scratch/project_462000456/tttt/ --install-dir /scratch/project_462000456/intel/oneapi
```

## Usage

```
. /scratch/project_462000456/intel/oneapi/setvars.sh --include-intel-llvm
module load LUMI/22.08
module load partition/G
module load rocm/5.3.3
module load cce/16.0.1
icpx -fsycl -fsycl-targets=amdgcn-amd-amdhsa,spir64_x86_64 -Xsycl-target-backend=amdgcn-amd-amdhsa  --offload-arch=gfx90a  <sycl_code>.cpp
```
