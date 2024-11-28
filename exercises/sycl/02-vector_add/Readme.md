# Vector Addition
The task is to compute the element-wise addition of two vectors (C = A + B) in parallel.

A skeleton code is provided in vector_add_<..>.cpp. You need to complete the missing parts to calculate the result in parallel. Try running the program on both CPU and GPU devices.

A typical application running on an accelerator follows these steps:

 1. Initialize data on the host.
 1. Create a queue and associate it with the desired device.
 1. Manage memory on the device by creating necessary constructs.
 1. Launch the kernel.
 1. Retrieve and verify the results on the host.
 
In this exercise, we will explore various memory models.

## I. Memory management using Buffers and Accessors

Use the skeleton provided in `vector_add_buffer.cpp`. Look for the **//TODO** lines.

### Step 1: Define a Queue
Start by defining a **queue**  and selecting the appropriate device selector. SYCL provides predefined selectors, such as: default, gpu, cpu, accelerator or you can use the procedure from the [previous exercise](../01-info/enumerate_device.cpp).

### Step 2: Create Buffers
Next, create buffers to encapsulate the data. For a one-dimensional array of integers of length `N`, with pointer `P`, a buffer can be constructed as follows:

```cpp
    sycl::buffer<int, 1> a_buf(P, sycl::range<1>(N));
```
### Step 3: Create Accessors
Accessors provide a mechanism to access data inside the buffers. Accessors on the device must be created within command groups. There are two ways to create accessors. Using the `sycl::accessor` class constructor

```cpp
   sycl::accessor a_acc{a_buf, h, read_write};
```
or  using the buffer `.getaccess<...>(h)`  member function:
```cpp
   auto a = a_buf.get_access<sycl::access::mode::read_write>(h);
```
**Important**  Use appropriate access modes for your data:
 - **Input Buffers:** Use `sycl::read_only` / `sycl::access::mode::read` to avoid unnecessary device-to-host data transfers.
 - **Output Buffers:** Use `sycl::write_only`/ `sycl::access::mode::write` to avoid unnecessary host-to-device data transfers.
 - **Input/Ouput Buffers:** Use `sycl::read_write` / `sycl::access::mode::read_write` for the variables that are input, but they also get modified during the computaions.

### Step 4: Submit the Task
Once accessors are ready, submit the task to the device using the `.parallel_for()` member function. The basic submission:

```cpp
   h.parallel_for(sycl::range{N}, [=](sycl::id<1> idx) {
        c_acc[idx] = a_acc[idx] + b_acc[idx];
      });
```  
Here: 
 - `sycl::range{N}` or `sycl::range(N)` specify number of work-items be launched 
 - `sycl::id<1>` represents the index used within the kernel.

#### Using **item** class instead of **id**
Modify the lambda function to use the  **sycl::item** class instead of the **id** class. In this case the index `idx` is obtained from the `.get_id()` member.

#### Using ND-Range
This basic launching serves our purpose for this simpler example, however it is useful to test also the **ND-RANGE**. In case we specify to the runtime the total size of the grid of work-items and size of a work-group as well:

```cpp
   h.parallel_for(sycl::nd_range<1>(sycl::range<1>(((N+local_size-1)/local_size)*local_size), sycl::range<1>(local_size)), [=](sycl::nd_itemi<1> item) {
        auto idx=item.get_global_id(0);
        c[idx] = a[idx] + b[idx];
      });
```  
**Note** that **ND-RANGE** requires that the total number of work-items to be divisible by the size of the work-group.

### Step 5: Retrieve Data
The final task in this exercise is to move the checking of the results  within the scope of the buffers (before the ending curly bracket) and add the appropriate method to access this data.

By default, buffers are automatically synchronized with the host when they go out of scope. However, if you need to access data within the buffer’s scope, use **host accessors**. 

Similar to the device  accessors, it is possible to define host accessors in two ways. By using the accessor class constructor
```cpp
    host_accessor c{c_buf, read_only};
``` 
or by using the `.get_access` member function of the buffer
```cpp
    auto c = c_buf.get_access<access::mode::read>();
```

## II. Memory management with Unified Shared Memory
 
###  IIa) **malloc_device**

Use the skeleton provided in `vector_add_usm_device.cpp`. Look for the **//TODO** lines.

### Step 1: Define a Queue
Same as using buffers

### Step 2: Allocate Memory on the Device Using `malloc_device`
Instead of creating buffers, allocate memory directly on the device using `sycl::malloc_device`. For a one-dimensional array of integers of length N, memory can be allocated as follows:

```cpp
    int* a_usm = sycl::malloc_device<int>(N, q);
```
### Step 3: Copy Data to the Device

You need to copy the data from the host to the device memory. Use sycl::memcpy to transfer data from the host memory to device memory before launching the kernel:
```cpp
    q.memcpy(a_usm, a.data(), N * sizeof(int)).wait();
``` 

### Step 4: Submit the Task
Same as using buffers.

### Step 5: Retrieve Data

After the kernel execution is complete, you need to copy the result back from the device to the host. Use `sycl::memcpy` again to transfer the result:
```cpp
    q.memcpy(c.data(), c_usm, N * sizeof(int)).wait();
```
### Step 6: Free Device Memory

Once you're done with the device memory, free the allocated memory using `sycl::free`:

```cpp
    sycl::free(a_usm, q);
```
This ensures that the allocated memory is properly released on the device.


### IIb) **malloc_shared**

Use the skeleton provided in `vector_add_usm_managed.cpp`. Look for the **//TODO** lines.

### Step 1: Define a Queue
Same as before

### Step 2: Allocate Memory on the Device Using `malloc_shared`
Allocate memory that can be migrated between host and device using `sycl::malloc_shared`. For a one-dimensional array of integers of length N, memory can be allocated as follows:

```cpp
int* a = sycl::malloc_shared<int>(N, q);
```
### Step 3: Initialize Data on Host

This part is already in the skeleton, it is done using `std::fill`. Though if you have time you can replace it with a **for loop**.

### Step 4: Submit the Task
Same as using buffers.

### Step 5: Synchronize and Check Results

Since `malloc_shared` migrates data automatically between the host and device, no explicit memory transfer is required. Ensure the queue finishes execution before accessing the results using `q.wait()`;
### Step 6: Free Device Memory

Once you're done with the device memory, free the allocated memory using `sycl::free`:

```cpp
sycl::free(a_usm, q);
```
This ensures that the allocated memory is properly released on the device.
