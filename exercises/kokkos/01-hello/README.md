# Kokkos hello world example

Here, the purpose of this exercise is to just understand the simplistic Kokkos hello world program `hello.cpp`, and the `Makefile` implementing Kokkos inline build strategy.

## Compiling and running
1. If not already done, enter directory `/path/higher-level-gpu-programming/exercises/kokkos/` and clone kokkos by `git clone https://github.com/kokkos/kokkos.git`. Now the kokkos repo should be located in `/path/higher-level-gpu-programming/exercises/kokkos/kokkos/` (you can use different location but this location is hardcoded in the solution Makefiles).

2. Return to `/path/higher-level-gpu-programming/exercises/kokkos/01-hello/` and just type `make` to compile. If you encounter compilation errors, make sure the backend compiler for the desired architecture is available, ie, `nvcc` for Mahti (use `module load cuda`) or `hipcc` for Lumi (use `module load rocm`). 

3. Run on Lumi or Mahti by `srun ./executable`  (add required flags according to the underlying system and user, eg, --account=XXX, --partition=YYY, etc.)

## Refs
The exercise code is based on [ENCCS material](https://enccs.github.io/gpu-programming/10-portable-kernel-models/) (CC-BY-4.0 license).