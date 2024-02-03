# Memory optimization I

GPUs are a specialized parallel hardware for floating point operations. The high performance of GPUs comes masive parallism by using many "small" gpu cores to perform the same operation on different of the data. It is critical to keep these gpu cores occupied at all time and provide them enough data to be processed. 

For small problem sizes a GPU  application  can run only on the GPU. The CPU would initialized the data, offload  all work to the GPU, and then at the end collect the data for further analysis. In this case there is data transfer between CPU and GPU at the beginning and at the end.  However is most applications today can not fit into the memory of a GPU, and many cases not even in a node.  It is extremely important to be aware of how the data is moved between CPU and GPU and minimize these transfers. 

In order to exemplify the importance of this, we consider the simple [Jacobi iterations](j_simple_with_buffer.cpp) in which we try to solve a problem by getting a series of solutions in many steps until the convergence criteria is reached. In this first implementation we focused on elegance and productivity. The application initializes the data on the CPU and then uses buffers to offload the operations to the GPUs. In the application we measure the total time to perform a specific number iterations via C++ chronos library and also the effective time spent in actual computation on the GPU using `sycl::event`.

## Timings and basic performance analysis
In order to test the code we executed the code on a nvidia GPU we ran the with quite large system size:

```
./j_simple_buffer -n 16000
```
The application reported a quite small time spent in executing kernels about 0.369 s. However the total time spent to execute all iterations was around 62 s.
```
Offload Device        : NVIDIA A100-SXM4-40GB
max_work_group_size   : 1024
Configuration         : MATRIX_SIZE= 16000x16000
 [0][0] = 8000
Warm up the device  
Kernel Execution Time : 0.368924 seconds
Compute Duration      : 62.1323 seconds
```

In this case we suspect that having the buffers created and destroyed every time step results in data being transfered between CPU and GPU. More information can be obtained using a performance analys tool. Since this is a code running on nvidia GPUs, using cuda as backend we can use the cuda toolkit performance anaylis tools included. We can get a lots of info by using [`nsys`](https://docs.csc.fi/computing/nsys/).

```
nsys profile -t nvtx,cuda -o results --stats=true --force-overwrite true ./j_simple_buffer -n 16000
```
From the output we only selected only some statistics related to the memory movements:

```
CUDA Memory Operation Statistics (by time):

 Time(%)  Total Time (ns)  Count  Average (ns)   Minimum (ns)  Maximum (ns)  StdDev (ns)       Operation     
 -------  ---------------  -----  -------------  ------------  ------------  ------------  ------------------
    64.3   39,958,457,360    404   98,907,072.7    80,076,865   141,251,921  19,554,201.0  [CUDA memcpy HtoD]
    35.7   22,211,682,360    202  109,958,823.6   107,057,217   123,094,875   2,242,793.1  [CUDA memcpy DtoH]



CUDA Memory Operation Statistics (by size):

 Total (MB)   Count  Average (MB)  Minimum (MB)  Maximum (MB)  StdDev (MB)      Operation     
 -----------  -----  ------------  ------------  ------------  -----------  ------------------
 413,696.000    404     1,024.000     1,024.000     1,024.000        0.000  [CUDA memcpy HtoD]
 206,848.000    202     1,024.000     1,024.000     1,024.000        0.000  [CUDA memcpy DtoH]
```
We note in the upper table that a lot of time (more than 62 s) executing cuda memory copy operations, while the lower panel we note that we had 606 memory operations and further more more than 600 GB of data moved around. The size of the problem is `2x976`MB. We can conclude that whole data was moved every iteration. 

# The task
The exercise is to reduce this data movement. The [solution](solution/) shows one way to solve this problem using unified shared memory. But you can try to experiment with moving the buffer declaration outside of the loop over iterations. Use the application timings and also the profilers to get the needed information. You can use Mahti or LUMI for this. 

On LUMI `rocm` is used as a backend. We can use `rocprof` to obtained similar information to the one given by `nsys` using:
```
rocprof --stats --hip-trace --hsa-trace ./j_simple_buffer -n 16000
```
