# Memory Optimization II

In the previouss [exercise](..07-jacobi/) we saw that having bad data management can have huge impact on the performance. But once we optimized the data movement between CPU and GPU we also need to make sure that it is efficiently access when executing the kernels.

In GPU memory loads from the memory is done in blocks. When gpu thread (or work item) need data from the GPU memory it triggeres a data movement of 64 B or 128B. However threads are physically locked in sub-groups (called warps in CUDA  or waves in Rocm). In order to maximize performance we need to make sure that all threads in a sub-group execute the same instruction and that the data corresponding to work items are close in the memory. This way instead having  64 B or 128B loaded for each work item, we have only on memory operation for all sub-group. This is called `coalesced` access. The GPUs in general have some caching capabilities, however very limited and it is the job of the programmer to write kernels ensuring coalesced accesses. 

A simple access pattern `c[id]=b[id]+a[id]` is coalesced if `id` is the index of the work item. In cases with complicated access patterns we can try to use `local share memory` to improve the memory operations. In addition improving  access the pattern the local share memory can be used as a programmable cache so that data that is needed by several work items in the same work group can use it without additional memory oprations.

Dense matrix operations are example of problems were optimizing  memory accesses is critical for good performance. Let's consider the matrix-matrix multiplication `C=AxB`, with all dimensions equal to `N`. In a naive implementation we would launch a a grid of threads `(N,N)` with each thread processing one element `(i,j)` of C:

```
//# Define size for ND-Range and work-group size
            range<2> global_size(N,N);
            range<2> work_group_size(M,M);

            //# Parallel Compute Matrix Multiplication
            h.parallel_for(nd_range<2>{global_size, work_group_size}, [=](nd_item<2> item){
                const int i = item.get_global_id(0);
                const int j = item.get_global_id(1);
                for (int k = 0; k < N; k++) {
                    matrix_c[i*N+j] +=matrix_a[i*N+k] * matrix_b[k*N+j];
                }
            });
```


