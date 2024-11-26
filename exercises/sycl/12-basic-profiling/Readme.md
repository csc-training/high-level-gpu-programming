# Basic Profiling

Start from the [skeleton](naive.cpp).  Look for the lines `//TODO`

In this exercise you need to implement a basic measurement of the execution time for a kernel performing matrix-matrix multiplication.  
First modify the **queue** definition and enable profiling
```cpp
queue q{property::queue::enable_profiling{}};
```
Next set-up `sycl::event` object. the same way is done in the [previous exercise](/exercises/sycl/03-axpy/). Compute the execution time of the kernel by taking the difference between the end of the execution of the kernel and the start of the execution.
```
e.get_profiling_info<info::event_profiling::command_end>() - e.get_profiling_info<info::event_profiling::command_start>();
```
Remember to first synchronize the host and the device (`e.wait()`)!
