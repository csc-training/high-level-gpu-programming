# Reduction loop exercise

Here, the purpose is to write a simple program using Kokkos that evaluates a reduction loop from `i = 0` to `ì = 9`. The loop calculates the sum between each `i` on a GPU and assigns that sum to a `sum` variable. After this, the program should print out `sum` to see the results are correct (45, if `i` ranges from `0` to `9`). In addition to `Kokkos::initialize()` and `Kokkos::finalize`, you will need `Kokkos::parallel_reduce` and `Kokkos::fence` in this exercise. In the exercise, it is not necessary to allocate any Unified or device memory, or explicitly program any memory transfers.

## Compiling and running
1. If not already done, enter directory `/path/higher-level-gpu-programming/exercises/kokkos/` and clone kokkos by `git clone https://github.com/kokkos/kokkos.git`. Now the kokkos repo should be located in `/path/higher-level-gpu-programming/exercises/kokkos/kokkos/` (you can use different location but this location is hardcoded in the solution Makefiles).

2. Then just create a source file and Makefile and type `make`. If you encounter compilation errors, make sure the backend compiler for the desired architecture is available, ie, `nvcc` for Mahti (use `module load cuda`) or `hipcc` for Lumi (use `module load rocm`). Hint! You can use the Makefile from the solution folder as a reference and just change the Kokkos path and the file name.

3. Run on Lumi or Mahti by `srun ./executable` (add required flags according to the underlying system and user, eg, --account=XXX, --partition=YYY, etc.)

## Example solution
An example Kokkos implementation (.cpp) based on [ENCCS material](https://enccs.github.io/gpu-programming/10-portable-kernel-models/) (CC-BY-4.0 license) is given in the solution folder. However, the intention is to try solving the exercise first without looking into this.
