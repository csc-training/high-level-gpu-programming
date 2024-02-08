# Installing oneAPI on Mahti

Load cuda:

    module load cuda

Download the oneAPI basekit:

    wget https://registrationcenter-download.intel.com/akdlm/IRC_NAS/163da6e4-56eb-4948-aba3-debcec61c064/l_BaseKit_p_2024.0.1.46_offline.sh
    chmod +x l_BaseKit_p_2024.0.1.46_offline.sh

Install:

    ./l_BaseKit_p_2024.0.1.46_offline.sh -a -s --eula accept --download-cache /scratch/project_2008874/$USER/oneapi_tmp/ --install-dir /scratch/project_2008874/$USER/intel/oneapi

Get the cuda plugin (the link below might be changed in the future):

    curl -LOJ "https://developer.codeplay.com/api/v1/products/download?product=oneapi&variant=nvidia&version=2024.0.1&filters[]=12.0&filters[]=linux"

Install:

    ./oneapi-for-nvidia-gpus-2024.0.1-cuda-12.0-linux.sh -y --extract-folder /scratch/project_2008874/$USER/oneapi_tmp/ --install-dir /scratch/project_2008874/$USER/intel/oneapi

## References

* [Intel oneAPI installation guide](https://www.intel.com/content/www/us/en/docs/oneapi/installation-guide-linux/2024-0/install-with-command-line.html)
* [Intel oneAPI download page](https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit-download.html?operatingsystem=linux&distributions=offline)
* [Codeplay oneAPI for NVIDIA GPUs](https://developer.codeplay.com/products/oneapi/nvidia/2024.0.1/guides/get-started-guide-nvidia)

