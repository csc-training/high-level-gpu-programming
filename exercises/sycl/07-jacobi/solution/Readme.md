# Jacobi Iterations with Unified Shared Memory

Using buffer is elegant and simple. Buffers lock the data so that only one kernel at the time can access it and manges using SYCL where data is at specific moment (CPU or GPU).  We saw that in the case of Jacobi iterations the buffers are created and destroyed every time step which results in very low performance. 

In this specific problem there is no need to transfer the data to CPU every time step. Tipically one needs to check for convergence after doing many iterations. What we need is to initailize the data on CPU transfer it to GPU perform a given number of steps and then check the state. 
One way to achive is by controlling the pointers allocations in a manner similar to CUDA or HIP. We can allocate data on GPU using the `malloc_device()` device method or `malloc_shared()`. In former the data to which pointer points  is only accesable from GPU, while in the latter the data resides on CPU or GPU depending on the performed operations. The cuda or the rocm  backends will take care of the data movement. 

The solution uses `malloc_shared` to allocate the arrays. The initialization is done on the CPU and the data resides on the CPU until the offloading starts. The data is automatically transfered to the GPU for processing. Because the CPU does not need the data there is no reason to have transfers until all iterations are performed. When running this code we note that we have very similar ecxecution time for the kernel, but now the total compute time is much smaller (from 62 s to 0.365). 

```
./j_simple_usm_shared -n 16000
Offload Device        : NVIDIA A100-SXM4-40GB
max_work_group_size   : 1024
Configuration         : MATRIX_SIZE= 16000x16000
 [0][0] = 8000
Warm up the device  
Kernel Execution Time : 0.359838 seconds
Compute Duration      : 0.365444 seconds
```
Doing the same performance with `nsys` or `rocprof` we now see that the time spent in trrasnfering data decreased to 0.058 s and the total size of transfered data is 593 MB, once at the beginning. 

```
CUDA Memory Operation Statistics (by time):

 Time(%)  Total Time (ns)  Count   Average (ns)  Minimum (ns)  Maximum (ns)  StdDev (ns)              Operation            
 -------  ---------------  ------  ------------  ------------  ------------  -----------  ---------------------------------
   100.0       58,052,881  13,040       4,451.9         2,622        56,128      3,820.7  [CUDA Unified Memory memcpy HtoD]
     0.0            6,655       2       3,327.5         2,207         4,448      1,584.6  [CUDA Unified Memory memcpy DtoH]



CUDA Memory Operation Statistics (by size):

 Total (MB)  Count   Average (MB)  Minimum (MB)  Maximum (MB)  StdDev (MB)              Operation            
 ----------  ------  ------------  ------------  ------------  -----------  ---------------------------------
    593.043  13,040         0.045         0.004         1.044        0.100  [CUDA Unified Memory memcpy HtoD]
      0.066       2         0.033         0.004         0.061        0.041  [CUDA Unified Memory memcpy DtoH]
``` 
**Note** It seems that the results was not transfered back to cpu. This can triggered by accessing the data from CPU.
