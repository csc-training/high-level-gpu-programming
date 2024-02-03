# SYCL2 LAB 
We will be working on Mahti and or Lumi, this is up to you!
You are all responsible for the jobscripts, Christian put examples for this on Github.
You can of course work interactively.
<br>
For all the exercises you have to write some code, compile the program and then run the program.
There are suggested solutions but don't look at them to early.
First try to solve it yourself.
<br>
# LAB2: get your hands dirty:

# Unified Shared Memory (USM)
- What is USM?
- Types of USM
- Code: Implicit USM
- Code: Explicit USM
- Data Dependency in USM
- Code: Data Dependency in-order queues
- Code: Data Dependency out-of-order queues
- Lab Exercise: Unified Shared Memory
<br>

## Learning Objectives
- Use new SYCL2020 features such as Unified Shared Memory to simplify programming.
- Understand implicit and explicit way of moving memory using USM.
- Solve data dependency between kernel tasks in optimal way.
<br>

## What is Unified Shared Memory?
Unified Shared Memory (USM) is a pointer-based memory management in SYCL. USM is a pointer-based approach that
should be familiar to C and C++ programmers who use malloc or new to allocate data. USM simplifies development for
the programmer when porting existing C/C++ code to SYCL.
<br>

## Developer view of USM
The picture below shows developer view of memory without USM and with USM.
<br>
With USM, the developer can reference that same memory object in host and device code.
![](pics/usm.png)

## Types of USM
Unified shared memory provides both explicit and implicit models for managing memory.
![](pics/usm_2.png)

## USM Syntax
USM Initialization: The initialization below shows example of shared allocation using malloc_shared, the "q" queue
parameter provides information about the device that memory is accessible.
<br>
```c++
int *data = malloc_shared<int>(N, q);
```

OR you can use familiar C++/C style malloc:

```c++
int *data = static_cast<int *>(malloc_shared(N * sizeof(int), q));
```

## Freeing USM:
```c++
free(data, q);
```

## USM Implicit Data Movement
The SYCL code below shows an implementation of USM using malloc_shared, in which data movement happens implicitly
between host and device. Useful to get functional quickly with minimum amount of code and developers will not
having worry about moving memory between host and device.
The SYCL code below demonstrates USM Implicit Data Movement:

```c++
#include <sycl/sycl.hpp>

using namespace sycl;
static const int N = 16;

int main()
{
    queue q;
    std::cout << "Device : " << q.get_device().get_info<info::device::name>() << "\n";
    //# USM allocation using malloc_shared
    int *data = malloc_shared<int>(N, q);
    //# Initialize data array
    for (int i = 0; i < N; i++) data[i] = i;
        
    //# Modify data array on device
    q.parallel_for(range<1>(N), [=](id<1> i) { data[i] *= 2; }).wait();

    //# print output
    for (int i = 0; i < N; i++) std::cout << data[i] << "\n";
        
    free(data, q);
    return 0;
        
}
```

# Exercises USM
## usm1
1. Inspect the code in usm/usm.cpp file.
2. Compile and run this code using our jobscripts for offloading to our nvidia GPUs.

## USM Explicit Data Movement
The SYCL code below shows an implementation of USM using malloc_device, in which data movement between host and
device should be done explicitly by developer using memcpy. This allows developers to have more controlled
movement of data between host and device.
<br>
The SYCL code below demonstrates USM Explicit Data Movement:

```c++
#include <sycl/sycl.hpp>

using namespace sycl;
static const int N = 16;

int main()
{
    queue q;
    std::cout << "Device : " << q.get_device().get_info<info::device::name>() << "\n";

    //# initialize data on host
    int *data = static_cast<int *>(malloc(N * sizeof(int)));
    
    for (int i = 0; i < N; i++) data[i] = i;

    //# Explicit USM allocation using malloc_device
    int *data_device = malloc_device<int>(N, q);

    //# copy mem from host to device
    q.memcpy(data_device, data, sizeof(int) * N).wait();
    
    //# update device memory
    q.parallel_for(range<1>(N), [=](id<1> i) { data_device[i] *= 2; }).wait();
    
    //# copy mem from device to host
    q.memcpy(data, data_device, sizeof(int) * N).wait();
    
    //# print output
    for (int i = 0; i < N; i++) std::cout << data[i] << "\n";
    
    free(data_device, q);
    free(data);

    return 0;
}
```

# Exercises USM
## usm2
1. Inspect the code in usm/usm_explicit.cpp file
2. Compile and run this code
3. What happens if you don’t wait on the event returned from parallel_for on line 26?
<br>

## When to use USM?
SYCL* Buffers are powerful and elegant. Use them if the abstraction applies cleanly in your application, and/or if
buffers aren’t disruptive to your development. However, replacing all pointers and arrays with buffers in a C++ program
can be a burden to programmers so in this case consider using USM

## USM provides a familiar pointer-based C++ interface:
- Useful when porting C++ code to SYCL by minimizing changes
- Use shared allocations when porting code to get functional quickly. Note that shared allocation is not intended to provide peak performance out of box.
- Use explicit USM allocations when controlled data movement is needed.

## Data dependency in USM
When using unified shared memory, dependences between tasks must be specified using events since tasks execute
asynchronously and multiple tasks can execute simultaneously.
<br>
Programmers may either explicitly wait on event objects or use the depends_on method inside a command group to
specify a list of events that must complete before a task may begin.
<br>
In the example below, the two kernel tasks are updating the same data array, these two kernels can execute
simultaneously and may cause undesired result. The first task must be complete before the second can begin, the next
section will show different ways the data dependency can be resolved.

```c++
q.parallel_for(range<1>(N), [=](id<1> i) { data[i] += 2; });

q.parallel_for(range<1>(N), [=](id<1> i) { data[i] += 3; });
```

## Different options to manage data dependency when using USM:
- wait() on kernel task
- use in_order queue property
- use depends_on method

### wait()
- Use q.wait() on kernel task to wait before the next dependent task can begin, however it will block execution on host.

```c++
q.parallel_for(range<1>(N), [=](id<1> i) { data[i] += 2; });
q.wait(); // <--- wait() will make sure that task is complete before continuing
q.parallel_for(range<1>(N), [=](id<1> i) { data[i] += 3; });
```

### in_order queue property
- Use in_order queue property for the queue, this will serialize all the kerenel tasks. Note that execution will not overlap even if the queues have no data dependency.

```c++
queue q{property::queue::in_order()}; // <--- this will serialize all kernel tasks
q.parallel_for(range<1>(N), [=](id<1> i) { data[i] += 2; });
q.parallel_for(range<1>(N), [=](id<1> i) { data[i] += 3; });
```

### depends_on
- Use h.depends_on(e) method in command group to specify events that must complete before a task may begin.

```c++
auto e = q.submit([&](handler &h)
{ // <--- e is event for kernel task
    h.parallel_for(range<1>(N), [=](id<1> i) { data[i] += 2; });
});

q.submit([&](handler &h)
{
    h.depends_on(e); // <--- waits until event e is complete
    h.parallel_for(range<1>(N), [=](id<1> i) { data[i] += 3; });
});
```

- You can also use a simplified way of specifying dependencies by passing an extra parameter in parallel_for

```c++
auto e = q.parallel_for(range<1>(N), [=](id<1> i) { data[i] += 2; });

q.parallel_for(range<1>(N), e, [=](id<1> i) { data[i] += 3; });
```

# Exercises USM
## usm3
Code Example: USM and Data dependency 1
<br>
The code in usm/usm_data.cpp uses USM and has three kernels that are submitted to the device. Each kernel modifies
the same data array. There is data dependency between the three queue submissions, so the code needs to be fixed to
get desired output of 20.

1. Inspect the code in usm/usm_data.cpp file and fix the bug.
<br>
There are three solutions: use in_order queue property or use wait() event or use depends_on() method.
<br>
### HINT
- Add wait() for each queue submit
- Implement depends_on() method in second and third kernel task
- Use in_order queue property instead of regular queue: queue q{property::queue::in_order()};

2. Compile and run this code using job scripts on the GPU.

# Exercises USM
## usm4
Code Example: USM and Data dependency 2
<br>
The code in lab2/usm_data2.cpp uses USM and has three kernels that are submitted to device. The first two kernels
modify two different memory objects and the third one has a dependency on the first two. There is data dependency
between the three queue submissions, so the code needs to be fixed to get the desired output of 25.

1. Inspect the code in usm/usm_data2.cpp file and implement the solution.
<br>
- Implementing depends_on() method gets the best performance
- Using in_order queue property or wait() will get results but not the most efficient

### HINT:
```c++
auto e1 = ...   ;
auto e2 = ...   ;
    
q.parallel_for(range<1>(N), {e1, e2}, [=](id<1> i)
{
    ...
});
```

2. Compile and run this code

# Exercises USM
## usm5
Complete the coding exercise using Unified Shared Memory concepts.
<br>
1. Complete the code in usm/usm_lab.cpp file by writing the missing code (look for comments)
<br>
- The code has two arrays data1 and data2 initialized on host
- Create USM device allocation for data1 and data2 and copy data to device
- Create two kernel tasks, one to update data1 with sqrt of values and another to update data2 with sqrt of values
- Create a third kernel task to add data2 into data1
- Copy data1 back to host and verify results
<br>
2. Compile and run this code as usual
<br>

# Buffer Memory Model
Buffers encapsulate data in a SYCL application across both devices and host. Accessors is the mechanism to access
buffer data.
As explained earlier in USM model section, offloading computation requires copying data between host and
device. In Buffer Memory model SYCL does not require the programmer to manage the data copies. By
creating Buffers and Accessors, SYCL ensures that the data is available to host and device without any programmer
effort. SYCL also allows the programmer explicit control over data movement when it is necessary to achieve best
performance. The code below shows Simple Vector addition using SYCL and Buffers. Read through the comments
addressed in step 1 through step 6.
<br>
Buffers encapsulate data in a SYCL application across both devices and host. Accessors is the mechanism to access
buffer data.
<br>
As explained earlier in USM model section, offloading computation requires copying data between host and
device. In Buffer Memory model SYCL does not require the programmer to manage the data copies. By
creating Buffers and Accessors, SYCL ensures that the data is available to host and device without any programmer
effort. SYCL also allows the programmer explicit control over data movement when it is necessary to achieve best
performance. The code below shows Simple Vector addition using SYCL and Buffers. Read through the comments
addressed in step 1 through step 6.

```c++
void SYCL_code(int* a, int* b, int* c, int N)
{
    //Step 1: create a device queue
    //(developer can specify a device type via device selector or use default selector)
    queue q;

    //Step 2: create buffers (represent both host and device memory)
    buffer buf_a(a, range<1>(N));
    buffer buf_b(b, range<1>(N));
    buffer buf_c(c, range<1>(N));

    //Step 3: submit a command for (asynchronous) execution
    q.submit([&](handler &h)
    {

        //Step 4: create buffer accessors to access buffer data on the device
        accessor A(buf_a,h,read_only);
        accessor B(buf_b,h,read_only);
        accessor C(buf_c,h,write_only);

        //Step 5: send a kernel (lambda) for execution
        h.parallel_for(N, [=](auto i)
        {
            //Step 6: write a kernel
            //Kernel invocations are executed in parallel
            //Kernel is invoked for each element of the range
            //Kernel invocation has access to the invocation id
            C[i] = A[i] + B[i];
        });
    });
}
```

## Vector Add implementation using USM and Buffers
The SYCL code below shows vector add computation implemented using USM and Buffers memory model:

```c++
#include <sycl/sycl.hpp>

using namespace sycl;

// kernel function to compute vector add using Unified Shared memory model (USM)
void kernel_usm(int* a, int* b, int* c, int N)
{
    //Step 1: create a device queue
    queue q;

    //Step 2: create USM device allocation
    auto a_device = malloc_device<int>(N, q);
    auto b_device = malloc_device<int>(N, q);
    auto c_device = malloc_device<int>(N, q);

    //Step 3: copy memory from host to device
    q.memcpy(a_device, a, N*sizeof(int));
    q.memcpy(b_device, b, N*sizeof(int));
    q.wait();

    //Step 4: send a kernel (lambda) for execution
    q.parallel_for(N, [=](auto i)
    {
        //Step 5: write a kernel
        c_device[i] = a_device[i] + b_device[i];
    }).wait();
    
    //Step 6: copy the result back to host
    q.memcpy(c, c_device, N*sizeof(int)).wait();

    //Step 7: free device allocation
    free(a_device, q);
    free(b_device, q);
    free(c_device, q);
}

// kernel function to compute vector add using Buffer memory model
void kernel_buffers(int* a, int* b, int* c, int N)
{
    //Step 1: create a device queue
    queue q;

    //Step 2: create buffers
    buffer buf_a(a, range<1>(N));
    buffer buf_b(b, range<1>(N));
    buffer buf_c(c, range<1>(N));

    //Step 3: submit a command for (asynchronous) execution
    q.submit([&](handler &h)
    {
        //Step 4: create buffer accessors to access buffer data on the device
        accessor A(buf_a, h, read_only);
        accessor B(buf_b, h, read_only);
        accessor C(buf_c, h, write_only);
        //Step 5: send a kernel (lambda) for execution
        h.parallel_for(N, [=](auto i)
        {
            //Step 6: write a kernel
            C[i] = A[i] + B[i];
        });
    });
}


int main()
{
    // initialize data arrays on host
    constexpr int N = 256;
    int a[N], b[N], c[N];
    for (int i=0; i<N; i++)
    {
        a[i] = 1;
        b[i] = 2;
    }
    
    // initialize c = 0 and offload computation using USM, print output
    for (int i=0; i<N; i++) c[i] = 0;

    kernel_usm(a, b, c, N);
    std::cout << "Vector Add Output (USM): \n";
    for (int i=0; i<N; i++)std::cout << c[i] << " ";std::cout << "\n";

    // initialize c = 0 and offload computation using USM, print output
    for (int i=0; i<N; i++) c[i] = 0;
    std::cout << "Vector Add Output (Buffers): \n";
    kernel_buffers(a, b, c, N);
    
    for (int i=0; i<N; i++)std::cout << c[i] << " ";std::cout << "\n";
}
```

###  Exercise Buffer1
1. Inspect the code in buffer/vector_add_usm_buffers.cpp file showing vector add computation implemented using USM and Buffers memory model.
2. Compile and run this code using jobscripts
3. Set the environment variable SYCL_PI_TRACE to enable the tracing of plugins/devices discovery. You should be able to check on which device you are running:


## Exercise Buffer 2

1. Try to solve the exercise described in the cpp file buffer/buffer.cpp.

```c++
#include <sycl/sycl.hpp>
using namespace sycl;
constexpr int N = 42;

// Buffers may not be directly accessed by the host and device (except
// through advanced and infrequently used mechanisms not described here).
// Instead, we must create accessors in order to read and write to buffers.
// Accessors provide the runtime with information about how we plan to use
// the data in buffers, allowing it to correctly schedule data movement.

// In this exercise the hostVector is declarated and initialized
// on the host.

// In the device block a buffer buf is created to interact with
// hostVector on the host.

//-----------------TASK-----------------------------
// Create buffer and an accessor to update the
// buffer buf on the device
// and write a h.parallel_for in which the indices
// are written to a and thereby also to buf
//-----------------TASK-----------------------------

int main()
{
    std::vector<int> hostVector(N, 3);

    std::cout << "printing hostVector before computation \n" ;
    for (int i = 0; i < N; i++) std::cout << hostVector[i] << " ";
    std::cout << "\n" ;

    {
        queue q;
        // Create a buffer buf for the hostVector
        // YOUR CODE GOES HERE
        
	
	q.submit( [&] (handler& h)
         {
        // Create an accessor a for buf
        // and write a h.parallel_for in which the indices
        // are written to a and thereby also to buf
        // YOUR CODE GOES HERE
         
	 
	 } );

    }

    // When exiting the scope of the devices destroys the buffer
    // and updating the hostVector

    std::cout << "printing hostVector after computation \n" ;
    for (int i = 0; i < N; i++) std::cout << hostVector[i] << " ";
    std::cout << "\n" ;

    return 0;
}

```

2. Build and test the program

## Exercise Buffer and Accessor 
1. Try to solve the exercise described in the cpp file buffer/buffer_accessor.cpp

```c++
#include <sycl/sycl.hpp>

using namespace sycl;

int main() {
  const int N = 16;

  //# Initialize a vector and print values
  std::vector<int> v1(N, 11);
  std::vector<int> v2(N, 22);
  std::vector<int> v3(N, 0);

  std::cout<<"\nInput V1: ";
  for (int i = 0; i < N; i++) std::cout << v1[i] << " ";
  std::cout<<"\nInput V2: ";
  for (int i = 0; i < N; i++) std::cout << v2[i] << " ";
  std::cout<<"\nInput V3: ";
  for (int i = 0; i < N; i++) std::cout << v3[i] << " ";

  {

    //# STEP 1 : Create buffers for the three vectors

    //# YOUR CODE GOES HERE




    //# Submit task to add vector
    queue q;
    q.submit([&](handler &h) {

      //# STEP 2 - create accessors for buffers with access permissions

      //# YOUR CODE GOES HERE



      h.parallel_for(range<1>(N), [=](id<1> i) {

        //# STEP 3 : Implement kernel code to add v3 = v1 + v2

        //# YOUR CODE GOES HERE




      });
    });

  }

  //# Print Output values
  std::cout<<"\nOutput V3: ";
  for (int i = 0; i < N; i++) std::cout<< v3[i] << " ";
  std::cout<<"\n";

  return 0;
}

```

2. Build and test the program
