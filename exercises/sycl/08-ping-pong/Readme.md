# Ping-Pong test

This is first SYCL + MPI code with 2 tasks. Task 0 initializes an array with 1s and sends it to Task 1, which receives the array, adds 1 to it via GPU or CPU, and then sends it back to Task 0. 
This code, while trivial, serves as an example of associating MPI tasks with available GPUs on the node using the SYCL queues. Additionally, it functions as a simple test to determine if the current MPI setup is GPU-aware, demonstrating the time difference between direct GPU-to-GPU data transfer and the alternative of transferring data to CPU first before sending or receiving and then copying to the GPU.

# Compilation
In order for this code to function correctly one needs to use GPU aware MPI. 
On Mahti one needs to load the specific cuda aware openmpi module (if using Intel onAPI):
```
module load openmpi/4.1.2-cuda
```
or manually link it (if using Adaptive Cpp).
On LUMI the GPU aware MPI can be used by setting an environment variable:
```
export MPICH_GPU_SUPPORT_ENABLED=1
```
Otherwise the compilation and execution is done as indicated in the [instructions](../../../Exercises_Instructions.md)
## Task
For this exercise one can start from the [cuda code](CUDA/src/) and just  replace all the cuda calls with equivalent SYCL calls. One has to define the SYCL queues associated with the right GPU in the node, allocate the GPU pointers, and replace the cuda kernel launch with equivalent SYCL lambdas or function operators. If you get stuck take a peek a the [solution](solution/pp_with_usm.cpp).

