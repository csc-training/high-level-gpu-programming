# For loop exercise

Here, the purpose is to write a simple program using Kokkos that evaluates a loop from `i = 0` to `ì = 5`. The loop calculates `c[i] = a[i] * b[i]` for all `i` on a GPU. Initialize the arrays `a` and `b`, eg, as follows:
```
for (unsigned i = 0; i < n; i++){
  a[i] = i;
  b[i] = 1;
}
```
At the end, the program should evaluate a verification loop (without Kokkos) to print out each `c[i]` to see the results are correct. 

In addition to `Kokkos::initialize()` and `Kokkos::finalize`, you will need `Kokkos::parallel_for` and `Kokkos::fence` in this exercise. Furthermore, two different memory management strategies are investigated here:

### Case 1: Raw pointers with Unified Memory
`Kokkos::kokkos_malloc` and `Kokkos::kokkos_free` are needed here.

### Case 2: Views with explicit memory management
`Kokkos::View` and `Kokkos::deep_copy` are needed here.


## Compiling and running
1. If not already done, enter directory `/path/higher-level-gpu-programming/exercises/kokkos/` and clone kokkos by `git clone https://github.com/kokkos/kokkos.git`. Now the kokkos repo should be located in `/path/higher-level-gpu-programming/exercises/kokkos/kokkos/` (you can use different location but this location is hardcoded in the solution Makefiles).

2. Then just create a source file and Makefile and type `make`. If you encounter compilation errors, make sure the backend compiler for the desired architecture is available, ie, `nvcc` for Mahti (use `module load cuda`) or `hipcc` for Lumi (use `module load rocm`). Hint! You can use the Makefile from the solution folder as a reference and just change the Kokkos path and the file name.

3. Run on Lumi or Mahti by `srun ./executable` (add required flags according to the underlying system and user, eg, --account=XXX, --partition=YYY, etc.)

## Example solution
Example Kokkos implementations (.cpp) based on [ENCCS material](https://enccs.github.io/gpu-programming/10-portable-kernel-models/) (CC-BY-4.0 license) are given in the solution folders. However, the intention is to try solving the exercise first without looking into these.
