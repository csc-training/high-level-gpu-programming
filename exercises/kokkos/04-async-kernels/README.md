# Bonus exercise: Asynchronous kernels

Here, the purpose is to write a simple program using Kokkos that splits a loop of size `nx` into `n` asynchronously running GPU kernels. It is not important what the loop actually does, the goal is to understand asynchronous kernel execution in Kokkos. You can, for example, store the loop index and the kernel number and print them at the end of the program to verify everything functions as desired. In this exercise, in addition to the Kokkos functions introduced in the earlier exercises, you will need [Kokkos::Experimental::partition_space](https://kokkos.org/kokkos-core-wiki/API/core/spaces/partition_space.html). For memory management, you can decide whether to use `Kokkos::kokkos_malloc` or `Kokkos::View`.

## Compiling and running
1. If not already done, enter directory `/path/higher-level-gpu-programming/exercises/kokkos/` and clone kokkos by `git clone https://github.com/kokkos/kokkos.git`. Now the kokkos repo should be located in `/path/higher-level-gpu-programming/exercises/kokkos/kokkos/` (you can use different location but this location is hardcoded in the solution Makefiles).

2. Then just create a source file and Makefile and type `make`. If you encounter compilation errors, make sure the backend compiler for the desired architecture is available, ie, `nvcc` for Mahti (use `module load cuda`) or `hipcc` for Lumi (use `module load rocm`). Hint! You can use the Makefile from the solution folder as a reference and just change the Kokkos path and the file name.

3. Run on Lumi or Mahti by `srun ./executable` (add required flags according to the underlying system and user, eg, --account=XXX, --partition=YYY, etc.)

## Example solution
An example Kokkos implementation (.cpp) based on [ENCCS material](https://enccs.github.io/gpu-programming/10-portable-kernel-models/) (CC-BY-4.0 license) is given in the solution folder. However, the intention is to try solving the exercise first without looking into this.
