AdaptiveCPP was previously known as hipSYCL.

# Installation with spack `v0.21`
Purge the modules. 
``` 
git clone -c feature.manyFiles=true https://github.com/spack/spack.git
cd spack
git checkout releases/v0.21
module purge
```
Set the `lmod`

```
. share/spack/setup-env.sh
spack config add "modules:default:enable:[tcl]"
spack install lmod
$(spack location -i lmod)/lmod/lmod/init/bash
. share/spack/setup-env.sh
``` 

Install:
It needs needs proper C++17 support to be built. If not available, first intall `gcc`:

 ```
spack install gcc@11.2.0
```
Add the compiler to spack:

```
spack load gcc@11.2.0
spack compiler add
```
Finally the `AdaptiveCpp` installation:
```
spack install hipsycl@0.9.4 %gcc@11.2.0 +cuda ^cuda@11.5.0 %gcc@11.2.0
```

After installation check the package with `syclcc --hipsycl-version` or `syclcc --hipsycl-info`.

### (Optional, Not working at the moment on Mahti) OpenMP offloading to Nvidia GPUs

```
spack load gcc@11.2.0
spack load cuda@11.5.0 
spack install gcc@12.2.0 %gcc@11.2.0+ nvptx ^cuda@11.5.0  %gcc@11.2.0
```

Version `11.2.0`
```
spack load gcc@11.2.0
spack load cuda@11.5.0 
spack install gcc@11.2.0 %gcc@11.2.0+ nvptx ^cuda@11.5.0  %gcc@11.2.0
```

### (Optional) Install `hip`

```
spack install hip
```


### Usage
After installation:
If the initial set-up was succesful there should be modules files available. 

```
module use /scratch/project_2008874/spack/share/spack/modules/linux-rhel8-x86_64_v3/
module load gcc/11.2.0-gcc-8.5.0-gpvckmb 
module load hipsycl
export LD_LIBRARY_PATH=/scratch/project_2008874/spack/opt/spack/linux-rhel8-x86_64_v3/gcc-8.5.0/gcc-11.2.0-gpvckmbfwjw5vhti5pznbiwpgzap2qld/lib64/:$LD_LIBRARY_PATH
```
Alternaetively if there are no modules files, one can use the `spack load <package>`. 
Compile with `syclcc` or `sycl-clang`. 
```
syclcc -O2 --hipsycl-targets="omp;cuda:sm_80" hello.cpp
```

