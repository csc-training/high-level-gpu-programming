## Compile from source on Mahti [github instructions](https://github.com/oneapi-src/SYCLomatic#build-from-source-code) 

```
export SYCLOMATIC_HOME=/scratch/project_2008874/workspace
export PATH_TO_C2S_INSTALL_FOLDER=/scratch/project_2008874/workspace/c2s_install
cd $SYCLOMATIC_HOME/
module purge
module use /scratch/project_2008874/spack/share/spack/modules/linux-rhel8-x86_64_v3/
module load gcc/11.2.0-gcc-8.5.0-gpvckmb
module load cmake
module load ninja
```

```
cd $SYCLOMATIC_HOME/
mkdir build
cd build
cmake -G Ninja -DCMAKE_INSTALL_PREFIX=$PATH_TO_C2S_INSTALL_FOLDER  -DCMAKE_BUILD_TYPE=Release  -DLLVM_ENABLE_PROJECTS="clang"  -DLLVM_TARGETS_TO_BUILD="X86;NVPTX" ../SYCLomatic/llvm
ninja install-c2s
```

```
export PATH=$PATH_TO_C2S_INSTALL_FOLDER/bin:$PATH
export CPATH=$PATH_TO_C2S_INSTALL_FOLDER/include:$CPATH
```
