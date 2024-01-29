# High-Level GPU Programming

Course material for the CSC course "High-Level GPU ProgrammingP". The course is
part of the EuroCC training activities at CSC.

## Agenda

### Day 1

| Time | Topic |
| ---- | ----- |
| 09:00-09:15 | Welcome 
| 09:15-09:20 | Break 
| 09:20-09:50 | Introduction to GPUs
| 09:50-10:00 | Break 
| 10:00-10:30 | GPU parallel programming model
| 10:50-11:20 | GPU memory hierarchy
| 11:25-12:00 | Refresher of C++ concepts (TR)
12:00-13:00 Lunch break
13:00-13:30 Mahti and LUMI  Computing Platforms (login & edit files, Slurm,SYCL installation, usage) (CVA)
13:30-15:00 SYCL I(a)  (Soner) 
15:00-15:20 Coffee break
13:00-16:45 SYCL I(b) (Soner) 
16:45-17:00 Day 1 wrap-up (CVA)
## Day 2, Thursday 15.02, 9:00-17:00

09:00-12:00 SYCL II (Soner) (advance features, shared memory)
12:00-13:00 Lunch break
13:00-15:30 SYCL III (Soner) (syclomatic,dependency and order of execution)
15:00-15:20  Coffee Break
15:50-16:00 Break
16:00-16:45 Exercises (simple exercises, heat equation, intro & cuda to sycl demo)   
16:45-17:00 Day 2 wrap-up
## Day 3 Friday 16.02, 9:00-17:00

09:00-9:30 Kokkos (JH)
09:30-11:00 Kokkos exercises (JH)
11:00-12:00 Interoperability with third-party libraries,  and multi-gpu, multi-node programming
12:00-13:00 Lunch break
13:00-16:45 Bring your own code
16:45-17:00 Day 3 wrap-up & Course closing



### Day 2

| Time | Topic |
| ---- | ----- |
| 09:00–09:35 | Fortran and HIP |
| 09:35–09:45 | Break |
| 09:45-10:30 | Multi-GPU programming, HIP+MPI |
| 10:30–10:45 | Break & Snacks |
| 10:45-11:30 | Exercises |
| 11:30-12:15 | Kernel optimizations |
| 12:15-13:00 | Lunch |
| 13:00-13:30 | Exercises |
| 13:30-14:15 | Code design, conditional compilation, lambdas, hipify  |
| 14:15-15:45 | Exercises & Break |
| 15:45-16:00 | Close-up | 

Link to [slides](https://kannu.csc.fi/s/gZSBE8DbeEKZjRw)
## Exercises

[General instructions](exercise-instructions.md)

### Introduction and GPU kernels

- [Hello world](kernels/01-hello-world)
- [Error checking](kernels/02-error-checking)
- [Kernel saxpy](kernels/03-kernel-saxpy)
- [Kernel copy2d](kernels/04-kernel-copy2d)

### Streams, events, and synchronization

- [Understanding asynchronity using events](streams/01-event-record)
- [Investigating streams and events](streams/02-concurrency)

### Memory allocations, access, and unified memory

- [Memory management strategies](memory/01-prefetch)
- [The stream-ordered memory allocator and memory pools](memory/02-mempools)
- [Unified memory and structs](memory/03-struct)

### Fortran and HIP

- [Hipfort exercise](hipfort)

### Optimization

- [Matrix Transpose](optimization/01-matrix_transpose)

### Multi-GPU programming and HIP+MPI

- [Peer to peer device access](multi-gpu/01-p2pcopy)
- [Vector sum on two GPUs without MPI](multi-gpu/02-vector-sum)
- [Ping-pong with multiple GPUs and MPI](multi-gpu/03-mpi)

### Code design, conditional compilation, lambdas, hipify

- [Host-device lambda functions and general kernels](lambdas/01-lambda)
- [Reductions with host-device lambdas and hipCUB](lambdas/02-reduction)
- [Monte Carlo simulation with hipRAND library](lambdas/03-hipify)

### Bonus
- [Heat equation with HIP](bonus/heat-equation)
 

Course material for the CSC course "GPU programming with HIP". The course is
part of the EuroCC training activities at CSC.

## Agenda

### Day 1

| Time | Topic |
| ---- | ----- |
| 09:00–09:35 | Welcome & Introduction to GPU programming |
| 09:35–09:45 | Break |
| 09:45-10:30 | HIP and GPU kernels |
| 10:30–10:45 | Break & Snacks |
| 10:45-11:30 | Exercises |
| 11:30-12:15 | Streams, events, and synchronization |
| 12:15-13:00 | Lunch |
| 13:00-13:30 | Exercises |
| 13:30-14:15 | Memory allocations, access and unified memory  |
| 14:15-15:45 | Exercises & Break |
| 15:45-16:00 | Day summary |


### Day 2

| Time | Topic |
| ---- | ----- |
| 09:00–09:35 | Fortran and HIP |
| 09:35–09:45 | Break |
| 09:45-10:30 | Multi-GPU programming, HIP+MPI |
| 10:30–10:45 | Break & Snacks |
| 10:45-11:30 | Exercises |
| 11:30-12:15 | Kernel optimizations |
| 12:15-13:00 | Lunch |
| 13:00-13:30 | Exercises |
| 13:30-14:15 | Code design, conditional compilation, lambdas, hipify  |
| 14:15-15:45 | Exercises & Break |
| 15:45-16:00 | Close-up | 

Link to [slides](https://kannu.csc.fi/s/gZSBE8DbeEKZjRw)
## Exercises

[General instructions](exercise-instructions.md)

### Introduction and GPU kernels

- [Hello world](kernels/01-hello-world)
- [Error checking](kernels/02-error-checking)
- [Kernel saxpy](kernels/03-kernel-saxpy)
- [Kernel copy2d](kernels/04-kernel-copy2d)

### Streams, events, and synchronization

- [Understanding asynchronity using events](streams/01-event-record)
- [Investigating streams and events](streams/02-concurrency)

### Memory allocations, access, and unified memory

- [Memory management strategies](memory/01-prefetch)
- [The stream-ordered memory allocator and memory pools](memory/02-mempools)
- [Unified memory and structs](memory/03-struct)

### Fortran and HIP

- [Hipfort exercise](hipfort)

### Optimization

- [Matrix Transpose](optimization/01-matrix_transpose)

### Multi-GPU programming and HIP+MPI

- [Peer to peer device access](multi-gpu/01-p2pcopy)
- [Vector sum on two GPUs without MPI](multi-gpu/02-vector-sum)
- [Ping-pong with multiple GPUs and MPI](multi-gpu/03-mpi)

### Code design, conditional compilation, lambdas, hipify

- [Host-device lambda functions and general kernels](lambdas/01-lambda)
- [Reductions with host-device lambdas and hipCUB](lambdas/02-reduction)
- [Monte Carlo simulation with hipRAND library](lambdas/03-hipify)

### Bonus
- [Heat equation with HIP](bonus/heat-equation)
