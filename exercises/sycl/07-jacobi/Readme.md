
### Timings

#### Using buffers 
```
./j_simple_buffer -n 16000
Offload Device        : NVIDIA A100-SXM4-40GB
max_work_group_size   : 1024
Configuration         : MATRIX_SIZE= 16000x16000
 [0][0] = 8000
Warm up the device  
Kernel Execution Time : 0.368924 seconds
Compute Duration      : 62.1323 seconds
```

#### Using USM, `malloc_shared`
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
### Memory Statistics (from nsys)
#### Using buffers
```

```

## Using USM, `malloc_shared`

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
## The task
1) Exercise test if moving the buffers outside of the for loop helps.
2) Test the codes on LUMI using rocprof for analysis
