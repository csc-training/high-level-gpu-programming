---
title:  Introduction to SYCL
subtitle: High-Level GPU Programming 
author:   CSC Training
date:     2024-11
lang:     en
---

# Introduction to SYCL{.section}

# What is SYCL?

 - C++ abstraction layer that can target various heterogeneous platforms in a single application
 - single source, high-level programming model
 - open source, royalty-free
 - developed by the Khronos Group 
    - 1.2 (2014), final (2015) revise 1.2.1 (2017)
    - 2.2 (2016), never finalized, C++14 and OpenCL 2.2
    - 2020 (2021), revision 9 (2024), C++17 and OpenCL 3.0
 - focused on 3P (Productivity, Portability, Performance)


# Productivity, Portability, Performance

 - **Productivity**: uses generic programming with templates and generic lambda functions.


 - **Portability**: it is a standard.


 - **Performance**: implementations aim to optimize SYCL for specific hardware platforms

# SYCL implementation


  - specific  adaptation of the SYCL programming model
    - **compilers**:  translate the SYCL code into machine code that can run on various hardware accelerators
    - **runtime library**: manages the execution of SYCL applications, handling  memory management, task scheduling, and synchronization across different devices
    - **backend support**: interface for various backends such as OpenCL, CUDA, HIP,  Level Zero, OpenMP
    - **standard template library**: interface for accesing functionalities and optimizations specific to SYCL
    - **development tools**: debuggers, profilers, etc.


# SYCL ecosystem

![https://www.khronos.org/blog/sycl-2020-what-do-you-need-to-know](img/2020-blog-sycl-03.jpg){.center width=75%}


# SYCL Implementations on Mahti and LUMI

**Intel One API** + CodePlay Plug-ins for Nvidia and AMD:

  - CPUs, Intel GPUs, Intel FPGAs (via OpenCL or Level Zero)
  - Nvidia GPUs (via CUDA), AMD GPUs (via ROCM)

**AdaptiveCpp** (former OpenSYCL, hipSYCL):

  - CPUs (via OpenMP)
  - Intel GPUs (via Level Zero)
  - Nvidia GPUs (via CUDA), AMD GPUs (via ROCM)


# Anatomy of a SYCL code

<small>
```cpp
#include <sycl/sycl.hpp>
using namespace sycl;

template <typename T>
void axpy(queue &q, const T &a, const std::vector<T> &x, std::vector<T> &y) {
  range<1> N{x.size()};
  buffer x_buf(x.data(), N);
  buffer y_buf(y.data(), N);

  q.submit([&](handler &h) {
    auto x = x_buf.template get_access<access::mode::read>(h);        // accessor x(x_buf, h, read_only);
    auto y = y_buf.template get_access<access::mode::read_write>(h);  // accessor y(y_buf, h, read_write);

    h.parallel_for(N, [=](id<1> i) {
      y[i] += a * x[i];
    });
  });
  q.wait_and_throw();
}
```
</small>

# Summary

 - single, high-level, standard C++  programming model 
 - can target various heterogenous platforms in a single application
 - Portability, Productivity, Performance
 - SYCL implementations, specific adaptions 
 - SYCL on Mahti and LUMI: Intel OneAPi+CodePlay plug-ins, AdaptiveCpp
