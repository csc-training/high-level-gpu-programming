# Building and running a Kokkos program

Here, the purpose of this exercise is to just understand the simplistic Kokkos hello world program `hello.cpp`, and the `Makefile` implementing Kokkos inline build strategy.

## Compiling and running

1. If not already done, clone the Kokkos repository  in a location of your choice by `git clone https://github.com/kokkos/kokkos.git`. Set environment variable `KOKKOS_PATH` to point into that directory, e.g.
```
export KOKKOS_PATH=/projappl/proj_xxxx/$USER/kokkos
``` 
You can also add `KOKKOS_PATH=/projappl/proj_xxxx/$USER/kokkos` to the beginning of the [Makefile](Makefile)

2. Edit the [Makefile](Makefile) and pick up `CXX`, `KOKKOS_DEVICES`, and `KOKKOS_ARCH` appropriate to your system. We recommend that you try this exercise both in LUMI and Mahti, and if your
laptop has software development tools (e.g. GCC) also there. Build the code by typing `make` to compile. If you encounter compilation errors, make sure the backend compiler for the desired architecture is available, ie, `nvcc` for Mahti (use `module load cuda`) or `hipcc` for Lumi (use `module load rocm`). 

3. Run on Lumi or Mahti by `srun ./executable`  (add required flags according to the underlying system and user, eg, --account=XXX, --partition=YYY, etc.)

## Refs
The exercise code is based on [ENCCS material](https://enccs.github.io/gpu-programming/10-portable-kernel-models/) (CC-BY-4.0 license).
