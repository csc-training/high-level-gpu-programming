# Dependencies, `axpy` example

In this exercise, you will solve the `axpy` problem (`Y=Y+a*X`) imposing the right dependencies on the various tasks. while correctly imposing task dependencies. All operations, including initialization, must be performed on devices. You should start from the [vector addition exercise solutions](../02-vector_add/solution/).

**Structure of the Code**:
  1. define a SYCL  queue
  1. declare  the variables
  1. fill the variables with data using 2 kernels, one for each array
  1. do the final `axpy` computation in another kernel 
  1. copy data to host to check the results

There are several ways to enforce dependencies. 

## I. Automatic dependencies using Buffer and Acceesors API
When managing memory with **buffers and accessors**, dependencies are handled automatically. Accessors ensure proper synchronization by blocking access to associated buffers until kernels complete their operations. As a result, kernels that use the same buffer execute in the correct order.

**Steps**
 1. Start from the [vector_add example](../02-vector_add/solution/vector_add_buffer.cpp) or use the skeleton  [axpy_buffer.cpp](axpy_buffer.cpp)
 1. initialize the arrays `X`and `Y` with two separate kernels. Use initial values  `X=1`, and `Y=2` at the beginning. 
 1.  compute `Y=Y+a*X` using a 3rd kernel with `a=1`
 1. copy the final result back to the host to validate

## II. Dependencies using USM

### IIa) Use of **in-order** queues 
SYCL queues are **out-of-order** by default, meaning kernels can execute concurrently. When using USM, submitting multiple kernels without explicit synchronization can result in incorrect execution order. To enforce task order, you can define the queue as **in-order**, ensuring tasks are executed sequentially.

**Steps**
 1. Start from the  [vector_add example](../02-vector_add/solution/vector_add_usm_device.cpp) or use the skeleton [axpy_usm_queue_sync.cpp](axpy_usm_queue_sync.cpp).
 1. Modify the queue definition
 ```cpp
 sycl::queue queue(sycl::default_selector{}, sycl::property::queue::in_order{});
```
 1. initialize `X` and `Y` on the device using two separate kernels (no need for `.memcpy` calls).
 1. submit a third kernel to compute `Y = Y + a * X` with a = 1.
 1. copy the result back to the host and validate it


### IIb) Use `sycl::events`

Instead of using **in-order** queues can use `sycl::events` to explicitly set the order of execution. Each kernel submission returns an event, which can be used to ensure that subsequent tasks wait for the completion of preceding tasks.


**Steps**
 1. Start from the  [vector_add example](../02-vector_add/solution/vector_add_usm_device.cpp) or use the skeleton [axpy_usm_events.cpp](axpy_usm_events.cpp)
 1. Keep the default out-of-order queue definition
 1. Initialize arrays `X` and `Y` on the device using two separate kernels. Capture the events for these kernel submissions:
```cpp
auto event_x = queue.submit([&](sycl::handler &h) {
    h.parallel_for(range{N}, [=](id<1> idx) { X[idx] = 1; });
});
auto event_b = queue.submit([&](sycl::handler &h) {
    h.parallel_for(range{N}, [=](id<1> idx) { Y[idx] = 2; });
});
```
 1. submit the `axpy` kernel with an explicit dependency on the two initialization events
 ```cpp
 queue.submit([&](sycl::handler &h) {
    h.depends_on({event_y, event_y});
    h.parallel_for(range{N}, [=](id<1> idx) { Y[idx] += a * X[idx]; });
});
```
or 

 ```cpp
 queue.
    h.parallel_for(range{N},{event_y, event_y}, [=](id<1> idx) { Y[idx] += a * X[idx]; });
```
 1. as a exercise you can synch the host with the event `sycl::event::wait({event_a, event_b});`
 1. copy the final result back to the host for validation
