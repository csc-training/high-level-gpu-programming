---
title:  Queues, Command Groups, Kernels
subtitle: High-Level GPU Programming 
author:   CSC Training
date:     2024-11
lang:     en
---

#  Queues, Command Groups, Kernels{.section}


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

# GPU Programming model 

 - Program runs on the CPU (host)
 - CPU initializes the GPUs (devices), allocates the memory, and stages the -GPU transfers
    - **Note!** CPU can also be a device
 - CPU launched the parallel code (kernel) ito be executed on a device by several threads
 - Code is written from the point of view of a single thread
    - each thread has a unique ID

# Device Discovery

 - it is important to know which devices are available
 - SYCL provides methods for querying:
    - `platform::get_platforms()` gives a list of available platforms
    - `.get_devices()` gives a list of present devices in a specific platform
    - `.get_info<info::device:: <property> >()` gives invo about specific property
        - can be applied to various classes
        - wide `sycl::info` namespace 

# SYCL Queue

 - SYCL class responsible for submitting commands
 - bridge between the host and the target device (**only one**)
 - associated with a SYCL device and a SYCL context
 - enable asynchronous execution
 - has an error-handling mechanism via an optional `exception_handler`
 - are **out-of-order** (default) or **in-order** (`{property::queue::in_order()}`)
 - encapsulates operations (e.g., kernel execution or memory transfers) using **command groups**

# Choosing the Device

  - `queue q();` targets the best device
  - Pre-configured classes of devices:
    - `queue q(default_selector_v);` targets the best device 
    - `queue q(cpu_selector_v);` targets the best CPU
    - `queue q(gpu_selector_v);` targets the best GPU
    - `queue q(accelerator_selector_v);` targets the best accelerator

# Custom Selector

<small>
```cpp
using namespace sycl;
class custom_selector : public device_selector
{
public:
  int operator()(const device &dev) const override
  {
    int score = -1;
    if (dev.is_gpu()) {
      auto vendor = dev.get_info<info::device::vendor>();
      if (vendor.find("NVIDIA") != std::string::npos) score += 75;
      if (vendor.find("Intel") != std::string::npos) score += 50;
      if (vendor.find("AMD") != std::string::npos) score += 100;
    }
    if (dev.is_host()) score += 25; // Give host device points so it is used if no GPU is available.

    return score;
  }
};
``` 
```cpp
auto Q = queue { custom_selector {} };

  std::cout << "we are running on: "
            << Q.get_device().get_info<info::device::vendor>() << " "
            << Q.get_device().get_info<info::device::name>() << std::endl;
```
</small>

# Explicit Way
 - using `get_platforms()` and/or `get_devices` 
```cpp
  std::cout << "\tChecking for GPUs\n" << std::endl;

  auto gpu_devices= sycl::device::get_devices(sycl::info::device_type::gpu);
  auto n_gpus=size( gpu_devices );

  std::cout << "\t\t There are "<< n_gpus << " GPUs\n"<< std::endl;
  if(n_gpus>0){
    queue q{gpu_devices[my_rank]};
  }
  else{
    std::cout << "\t\t There are no GPUs found \n Existing"<< std::endl;
    exit(1);
  }
``` 

# Command Groups

 - created via `.submit()` member
 - containers for operations to be executed 
 - give more control over executions than:
    - `q.parallel_for(N, [=](id<1> i) { y[i] += a * x[i];});`
 - can have dependencies for ensuring desired order
 - are executed *asynchronous* within specific **context** and **queue**
<small>
```cpp  
  q.submit([&](handler &cgh) {
    auto x = x_buf.template get_access<access::mode::read>(h);        // accessor x(x_buf, h, read_only);
    auto y = y_buf.template get_access<access::mode::read_write>(h);  // accessor y(y_buf, h, read_write);

    h.parallel_for(N, [=](id<1> i) {
      y[i] += a * x[i];
    });
  });
```
</small>

# Kernels
 - code to be executed in parallel
 - written from the point of view of a work-item (gpu thread)
    - each intance gets a unique `id` using the work-item index

<div class="column">
 - lambda expressions
```cpp
    [=](id<1> i) {
      y[i] += a * x[i];
    }
```
</div>

<div class="column">
 - function object (functors)
 <small>
```cpp 
class AXPYFunctor {
public:
    AXPYFunctor(float a, accessor<T> x, accessor<T> y): a(a), x(x), y(y) {}

    void operator()(id<1> i) {
        y[i] += a * x[i];
    }

private:
    float a;
    accessor<T> x; 
    accessor<T> y;
};
```
</small>


# Summary

 - **queueus* are the bridege between host and devices
 - each queue maps to one device
 - work is enqued by submitting **command groups**
    - give lots of flexibility
 - parallel code (kernel)  is submitted as a lambda function or as a function operator