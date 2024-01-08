```
module purge
module use /scratch/project_2008874/spack/share/spack/modules/linux-rhel8-x86_64_v3/
module load gcc/11.2.0-gcc-8.5.0-gpvckmb
module load cmake/3.27.7-gcc-11.2.0-t3n4tps
module load cuda/11.5.0-gcc-11.2.0-gmjjets
```

```
export SYCLOMATIC_HOME=/scratch/project_2008874/workspace
export PATH_TO_C2S_INSTALL_FOLDER=/scratch/project_2008874/workspace/c2s_install
export PATH=$PATH_TO_C2S_INSTALL_FOLDER/bin:$PATH
export CPATH=$PATH_TO_C2S_INSTALL_FOLDER/include:$CPATH
```
Convert a simple cuda code to sycl:

```
dpct --cuda-include-path=/scratch/project_2008874/spack/opt/spack/linux-rhel8-x86_64_v3/gcc-11.2.0/cuda-11.5.0-gmjjetscy7qwcwrrmuoqsujhdqkkyjss/include --in-root=./ src/vector_add.cu
```
Now we can try to compil using [intel llvm](Instructions_intel_llvm_Mahti.md).
