# Installation with spack `v0.21`
Purge the modules. 
``` 
git clone -c feature.manyFiles=true https://github.com/spack/spack.git
```
Set the `lmod`

```
. spack/share/spack/setup-env.sh
spack config add "modules:default:enable:[tcl]"
spack install lmod
$(spack location -i lmod)/lmod/lmod/init/bash
. spack/share/spack/setup-env.sh
``` 

Install:
 ```
spack install hipsycl@0.9.4 +cuda
```


After installation check the package with `syclcc --hipsycl-version` or `syclcc --hipsycl-info`.


### Usage
After installation:

```
module purge
cd spack
. share/spack/setup-env.sh # note the "." at the beginning of the command
spack load gcc
spack load cuda
spack load llvm
spack load hipsycl
```
Compile with `syclcc` or `sycl-clang`. 
```
syclcc -O2 --hipsycl-targets="omp;cuda:sm_80" hello.cpp
```

