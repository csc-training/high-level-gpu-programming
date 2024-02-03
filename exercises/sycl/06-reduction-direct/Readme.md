# Memory Optimization III

We saw in the [previous exercise](04-matrix-matrix-mul/) how local share memory can be used to improve the performance by reducing the amount of extra memory operations and by improving the access pattern. This codes show another way to use the local share memory for reductions.
This material is based on [ENCCS lecture](https://enccs.github.io/gpu-programming/9-non-portable-kernel-models/#reductions) 

## Efficient Reduction Operations on GPUs

`Reductions` refer to operations in which the elements of an array are aggregated in a single value through operations such as summing, finding the maximum or minimum, or performing logical operations. 

In the serial approach, the reduction is performed sequentially by iterating through the collection of values and accumulating the result step by step. This will be enough for small sizes, but for big problems this results in significant time spent in this part of an application. On a GPU, this approach is not feasible. Using just one thread to do this operation means the rest of the GPU is wasted. Doing reduction in parallel is a little tricky. In order for a thread to do work, it needs to have some partial result to use. If we launch, for example, a kernel performing a simple vector summation, ``sum[0]+=a[tid]``, with `N` threads we notice that this would result in undefined behaviour. GPUs have mechanisms to access the memory and lock the access for other threads while 1 thread is doing some operations to a given data via **atomics**, however this means that the memory access gets again to be serialized. There is not much gain. 
We note that when doing reductions the order of the iterations is not important (barring the typical non-associative behavior of floating-point operations). Also we can we might have to divide our problem in several subsets and do the reduction operation for each subset separately. On the GPUs, since the GPU threads are grouped in blocks, the size of the subset based on that. Inside the block, threads can cooperate with each other, they can share data via the shared memory and can be synchronized as well. All threads read the data to be reduced, but now we have significantly less partial results to deal with. In general, the size of the block ranges from 256 to 1024 threads. In case of very large problems, after this procedure if we are left too many partial results this step can be repeated.



At the block level we still have to perform a reduction in an efficient way. Doing it serially means that we are not using all GPU cores (roughly 97% of the computing capacity is wasted). Doing it naively parallel using **atomics**, but on the shared memory is also not a good option. Going back back to the fact the reduction operations are commutative and associative we can set each thread to "reduce" two elements of the local part of the array. Shared memory can be used to store the partial "reductions" as shown below in the code:

```
auto reductionKernel(sycl::handler &cgh, double *x, double *sum, int N) {
            sycl::local_accessor<double, 1> shtmp{{2*tpb}, cgh};
            return [=](sycl::nd_item<1> item) {
               int ibl = item.get_group(0);
               int ind = item.get_global_id(0);
               int tid = item.get_local_id(0);
               shtmp[tid] = 0;
               shtmp[tid + tpb] = 0;
               if (ind < N / 2) {
                  shtmp[tid] = x[ind];
               }
               if (ind + N / 2 < N) {
                  shtmp[tid + tpb] = x[ind + N / 2];
               }
               item.barrier();
               for (int s = tpb; s > 0; s >>= 1) {
                  if (tid < s) {
                      shtmp[tid] += shtmp[tid + s];
                  }
                  item.barrier();
               }
               if (tid == 0) {
                  sum[ibl] = shtmp[0]; // each block saves its partial result to an array
                  /*
                    sycl::atomic_ref<double, sycl::memory_order::relaxed,
                                   sycl::memory_scope::device,
                                   sycl::access::address_space::global_space>
                       ref(sum[0]);
                    ref.fetch_add(shtmp[0]);
                  */
                  // Alternatively, we could aggregate everything together at index 0.
                  // Only useful when there not many partial sums left and when the device supports
                  // atomic operations on FP64/double operands.
               }
            };
         }
```
