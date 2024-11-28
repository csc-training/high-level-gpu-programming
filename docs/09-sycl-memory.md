---
title:  SYCL Memory Management
subtitle: High-Level GPU Programming 
author:   CSC Training
date:     2024-11
lang:     en
---

# SYCL Memory Management{.section}

# Accelerator Model Today

- GPUs have their own memory separate from the CPU memory
- GPUs are connected to CPUs via PCIe
- Data must be copied from CPU to GPU over the PCIe bus

![](img/gpu-bws.png){width=100%}

# SYCL Memory Models

 - three memory-management abstractions in the SYCL standard:

      - **buffer and accessor API**: a buffer encapsulate the data and accessors describe how you access that data
      - **unified shared memory**: pointer-based approach to C/C++/CUDA/HIP
      - **images**: similar API to buffer types, but with extra functionality tailored for image processing (will not be discussed here)

# Buffers and Accesors I
 -  a **buffer** provides a high level abstract view of memory 
 - support 1-, 2-, or 3-dimensional data
 - dependencies between multiple kernels are implicitly handled
 - does not own the memory, it’s only a *constrained view* into it
 - **accessor** objects are used to access the data
 - various access modes, *read_write*, *read_only*, or *write_only*
 - can target local memory, **target::local**
 - can have also host accessors

# Buffers and Accesors II
 
```cpp
  std::vector<int> y(N, 1);
 {
    // Create buffers for data 
    buffer<int, 1> a_buf(y.data(), range<1>(N));
    q.submit([&](handler& cgh) {
      accessor y_acc{a_buf, cgh, read_write};
      cgh.parallel_for(range<1>(N), [=](id<1> id) {
        y_acc[id] +=1;
      });
    });
    host_accessor result{a_buf}; // host can access data also directly after buffer destruction
    for (int i = 0; i < N; i++) {
      assert(result[i] == 2);
    }
 }
``` 

# Unified Shared Memory (USM) I

- pointer-based approach to C/C++/CUDA/HIP
- explicit allocation and  freeing of memory
- explicit dependencies
- explicit host-device transfers (unless using managaged)
- explicit host-device synchronization 

# Unified Shared Memory II

<small>
```cpp
  std::vector<int> y(N, 1);

  // Allocate device memory
  int* d_y = malloc_device<int>(N, q); 
  // Copy data from host to device
  q.memcpy(d_y, y.data(), N * sizeof(int)).wait(); 

  q.submit([&](handler& cgh) {
    cgh.parallel_for(range<1>(N), [=](sid<1> id) {
      d_y[id] += 1;
    });
  }).wait();
  // Copy results back to host
  q.memcpy(y.data(), d_y, N * sizeof(int)).wait();

  // Free the device memory
  sycl::free(d_y, q);
  
  // Verify the results
  for (int i = 0; i < N; i++) {
    assert(y[i] == 2);
  }
```
</small>

# Unifed Shared Memory III

| Function        | Location	         | Device Accessible
------------------+--------------------+--------------------
| malloc_device	  | Device 	           | Yes                 
| malloc_shared	  | Dynamic migration  | Yes                 
| malloc_host	    | Host  	           | Device can read     

# Summary

 - buffers and accesors API
    - buffers are containers for data
    - accesors define how the data is accessed
 - unified shared memory 
    - pointer like memory management
    - device, shared, host
