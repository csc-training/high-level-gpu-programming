
## Using buffers
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

```
CUDA Kernel Statistics:

 Time(%)  Total Time (ns)  Instances  Average (ns)  Minimum (ns)  Maximum (ns)  StdDev (ns)                                                  Name                                                
 -------  ---------------  ---------  ------------  ------------  ------------  -----------  ----------------------------------------------------------------------------------------------------
    49.5        1,074,621        100      10,746.2         9,760        11,296        218.7  Typeinfo name for main::[lambda(sycl::_V1::handler &) (instance 3)]::operator ()(sycl::_V1::handler…
    49.5        1,074,429        100      10,744.3         9,760        11,136        198.0  Typeinfo name for main::[lambda(sycl::_V1::handler &) (instance 4)]::operator ()(sycl::_V1::handler…
     0.5           11,712          1      11,712.0        11,712        11,712          0.0  Typeinfo name for main::[lambda(sycl::_V1::handler &) (instance 1)]::operator ()(sycl::_V1::handler…
     0.5           11,328          1      11,328.0        11,328        11,328          0.0  Typeinfo name for main::[lambda(sycl::_V1::handler &) (instance 2)]::operator ()(sycl::_V1::handler…



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
./j_simple_usm_shared -n 16000
Offload Device        : NVIDIA A100-SXM4-40GB
max_work_group_size   : 1024
Configuration         : MATRIX_SIZE= 16000x16000
 [0][0] = 8000
Warm up the device  
Kernel Execution Time : 0.359838 seconds
Compute Duration      : 0.365444 seconds
```

```
CUDA API Statistics:

 Time(%)  Total Time (ns)  Num Calls  Average (ns)  Minimum (ns)  Maximum (ns)  StdDev (ns)          Name        
 -------  ---------------  ---------  ------------  ------------  ------------  ------------  -------------------
    66.1       20,374,097          2  10,187,048.5        26,139    20,347,958  14,369,696.0  cuMemAllocManaged  
    17.2        5,299,712      1,206       4,394.5           580       161,159       7,427.4  cuEventSynchronize 
     8.0        2,461,856          3     820,618.7         2,440     2,456,516   1,416,728.7  cuStreamSynchronize
     3.9        1,190,985        606       1,965.3         1,660        13,990         698.0  cuEventRecord      
     2.7          831,966        201       4,139.1         3,660        40,080       2,650.4  cuLaunchKernel     
     0.9          278,029        606         458.8           340         4,320         193.6  cuEventCreate      
     0.6          174,468        603         289.3           240           810          53.1  cuEventDestroy_v2  
     0.5          167,939          1     167,939.0       167,939       167,939           0.0  cuModuleLoadDataEx 
     0.1           41,270          1      41,270.0        41,270        41,270           0.0  cuModuleUnload     
     0.1           16,289          1      16,289.0        16,289        16,289           0.0  cuStreamCreate     
     0.0            6,550          1       6,550.0         6,550         6,550           0.0  cuStreamDestroy_v2 



CUDA Kernel Statistics:

 Time(%)  Total Time (ns)  Instances  Average (ns)  Minimum (ns)  Maximum (ns)  StdDev (ns)                                                  Name                                                
 -------  ---------------  ---------  ------------  ------------  ------------  -----------  ----------------------------------------------------------------------------------------------------
    52.0        2,469,337          1   2,469,337.0     2,469,337     2,469,337          0.0  Typeinfo name for main::[lambda(sycl::_V1::handler &) (instance 1)]::operator ()(sycl::_V1::handler…
    24.0        1,140,348        100      11,403.5        11,295        11,840         87.3  Typeinfo name for main::[lambda(sycl::_V1::handler &) (instance 2)]::operator ()(sycl::_V1::handler…
    24.0        1,137,500        100      11,375.0        11,264        11,968         80.4  Typeinfo name for main::[lambda(sycl::_V1::handler &) (instance 3)]::operator ()(sycl::_V1::handler…



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
