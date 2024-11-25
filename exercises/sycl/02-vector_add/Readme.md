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

## Memory management using Buffers and Accessors

Use the skeleton provided in `vector_add_buffer.cpp`. Look for the **//TODO** lines.

### Step 1: Define a Queue
Start by defining a **queue**  and selecting the appropriate device selector. SYCL provides predefined selectors, such as: default, gpu, cpu, accelerator or you can use the procedure from the [previous exercise](../01-info/enumerate_device.cpp).

### Step 2: Create Buffers
Next, create buffers to encapsulate the data. For a one-dimensional array of length `N`, with pointer `P`, a buffer can be constructed as follows:

```
sycl::buffer<int, 1> a_buf(P, sycl::range<1>(N));
```
### Step 3: Create Accessors
Accessors provide a mechanism to access data inside the buffers. Accessors on the device must be created within command groups. There are two ways to create accessors. Using the `sycl::accessor` class constructor

```
   sycl::accessor a{a_buf, h, sycl::read_write};
```
or  using the buffer `.getaccess<...>(h)`  member function:
```
a = a_buf.get_access<sycl::access::mode::read_write>(h);
```
**Important**  Use appropriate access modes for your data:
 - **Input Buffers:** Use `sycl::access::mode::read` to avoid unnecessary device-to-host data transfers.
 - **Output Buffers:** Use `sycl::access::mode::write` to avoid unnecessary host-to-device data transfers.
 - **Input/Ouput Buffers:** Use `sycl::access::mode::read_write` for the variables that are input, but they also get modified during the computaions.

### Step 4: Submit the Task
Once accessors are ready, submit the task to the device using the `.parallel_for()` member function. The basic submission:

```
   h.parallel_for(sycl::range{N}, [=](sycl::id<1> idx) {
        c[idx] = a[idx] + b[idx];
      });
```  
Here: 
 - `sycl::range{N}` or `sycl::range(N)` specify number of work-items be launched 
- `sycl::id<1>` represents the index used within the kernel.

#### Using **item** class instead of **id**
Modify the lambda function to use the  **sycl::item** class instead of the **id** class. In this case the index `idx` is obtained from the `.get_id()` member.

#### Using ND-Range
This basic launching serves our purpose for this simpler example, however it is useful to test also the **ND-RANGE**. In case we specify to the runtime the total size of the grid of work-items and size of a work-group as well:

```
   h.parallel_for(sycl::nd_range<1>(sycl::range<1>(((N+local_size-1)/local_size)*local_size), sycl::range<1>(local_size)), [=](sycl::nd_itemi<1> item) {
        auto idx=item.get_global_id(0);
        c[idx] = a[idx] + b[idx];
      });
```  
**Note** that **ND-RANGE** requires that the total number of work-items to be divisible by the size of the work-group.

### Step 5: Retrieve Data
The final task in this exercise is to move the checking of the results  within the scope of the buffers (before the ending curly bracket) and add the appropriate method to access this data.

By default, buffers are automatically synchronized with the host when they go out of scope. However, if you need to access data within the buffer’s scope, use **host accessors**. Host accessors can also be created in two ways:
Similar to the device, it is possible to define host accessors in two ways. By using the accessor class constructor
```
host_accessor c{c_buf, sycl::access::mode::read};
``` 
or by using the `.get_access` member function of the buffer
```
auto = c_buf.get_access<access::mode::read>();
```

## Memory management with Unified Shared Memory
 
### **malloc_device**

Use the skeleton provided in `vector_add_buffer.cpp`. Look for the **//TODO** lines.

### Step 1: Define a Queue
Same as using buffers

### Step 2: Create Buffers
Next, create buffers to encapsulate the data. For a one-dimensional array of length `N`, with pointer `P`, a buffer can be constructed as follows:

```
sycl::buffer<int, 1> a_buf(P, sycl::range<1>(N));
```
### Step 3: Create Accessors
Accessors provide a mechanism to access data inside the buffers. Accessors on the device must be created within command groups. There are two ways to create accessors. Using the `sycl::accessor` class constructor

```
   sycl::accessor a{a_buf, h, sycl::read_write};
```
or  using the buffer `.getaccess<...>(h)`  member function:
```
a = a_buf.get_access<sycl::access::mode::read_write>(h);
```
**Important**  Use appropriate access modes for your data:
 - **Input Buffers:** Use `sycl::access::mode::read` to avoid unnecessary device-to-host data transfers.
 - **Output Buffers:** Use `sycl::access::mode::write` to avoid unnecessary host-to-device data transfers.
 - **Input/Ouput Buffers:** Use `sycl::access::mode::read_write` for the variables that are input, but they also get modified during the computaions.

### Step 4: Submit the Task
Once accessors are ready, submit the task to the device using the `.parallel_for()` member function. The basic submission:

```
   h.parallel_for(sycl::range{N}, [=](sycl::id<1> idx) {
        c[idx] = a[idx] + b[idx];
      });
```  
Here: 
 - `sycl::range{N}` or `sycl::range(N)` specify number of work-items be launched 
- `sycl::id<1>` represents the index used within the kernel.

#### Using **item** class instead of **id**
Modify the lambda function to use the  **sycl::item** class instead of the **id** class. In this case the index `idx` is obtained from the `.get_id()` member.

#### Using ND-Range
This basic launching serves our purpose for this simpler example, however it is useful to test also the **ND-RANGE**. In case we specify to the runtime the total size of the grid of work-items and size of a work-group as well:

```
   h.parallel_for(sycl::nd_range<1>(sycl::range<1>(((N+local_size-1)/local_size)*local_size), sycl::range<1>(local_size)), [=](sycl::nd_itemi<1> item) {
        auto idx=item.get_global_id(0);
        c[idx] = a[idx] + b[idx];
      });
```  
**Note** that **ND-RANGE** requires that the total number of work-items to be divisible by the size of the work-group.

### Step 5: Retrieve Data
The final task in this exercise is to move the checking of the results  within the scope of the buffers (before the ending curly bracket) and add the appropriate method to access this data.

By default, buffers are automatically synchronized with the host when they go out of scope. However, if you need to access data within the buffer’s scope, use **host accessors**. Host accessors can also be created in two ways:
Similar to the device, it is possible to define host accessors in two ways. By using the accessor class constructor
```
host_accessor c{c_buf, sycl::access::mode::read};
``` 
or by using the `.get_access` member function of the buffer
```
auto = c_buf.get_access<access::mode::read>();
```
