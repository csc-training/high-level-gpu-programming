
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
CUDA Memory Operation Statistics (by time):

 Time(%)  Total Time (ns)  Count  Average (ns)  Minimum (ns)  Maximum (ns)  StdDev (ns)      Operation     
 -------  ---------------  -----  ------------  ------------  ------------  -----------  ------------------
    61.9      105,269,192    404     260,567.3       210,687       376,319     37,628.5  [CUDA memcpy HtoD]
    38.1       64,744,373    202     320,516.7       307,359       389,279     13,024.3  [CUDA memcpy DtoH]



CUDA Memory Operation Statistics (by size):

 Total (MB)  Count  Average (MB)  Minimum (MB)  Maximum (MB)  StdDev (MB)      Operation     
 ----------  -----  ------------  ------------  ------------  -----------  ------------------
  1,694.499    404         4.194         4.194         4.194        0.000  [CUDA memcpy HtoD]
    847.249    202         4.194         4.194         4.194        0.000  [CUDA memcpy DtoH]
```

## Using USM, `malloc_shared`

```

CUDA Memory Operation Statistics (by time):

 Time(%)  Total Time (ns)  Count  Average (ns)  Minimum (ns)  Maximum (ns)  StdDev (ns)              Operation            
 -------  ---------------  -----  ------------  ------------  ------------  -----------  ---------------------------------
    98.4          445,148     74       6,015.5         3,231        47,199      7,319.0  [CUDA Unified Memory memcpy HtoD]
     1.6            7,167      2       3,583.5         2,367         4,800      1,720.4  [CUDA Unified Memory memcpy DtoH]



CUDA Memory Operation Statistics (by size):

 Total (MB)  Count  Average (MB)  Minimum (MB)  Maximum (MB)  StdDev (MB)              Operation            
 ----------  -----  ------------  ------------  ------------  -----------  ---------------------------------
      4.194     74         0.057         0.004         0.852        0.141  [CUDA Unified Memory memcpy HtoD]
      0.066      2         0.033         0.004         0.061        0.041  [CUDA Unified Memory memcpy DtoH]
```
## The task
1) Exercise test if moving the buffers outside of the for loop helps.
2) Test the codes on LUMI using rocprof for analysis
