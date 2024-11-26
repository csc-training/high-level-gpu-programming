# Basic Profiling

In this exercise you need to implement a basic measurement of the execution time for a kernel performing matrix-matrix multiplication.  Compute the executiomn time of the kernel by taking the difference between the end of the execution of the kernel and the start of the execution.
```
e.get_profiling_info<info::event_profiling::command_end>() - e.get_profiling_info<info::event_profiling::command_start>();
```
Rember to first synchronize the host and the device!
