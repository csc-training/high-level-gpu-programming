# High-Level GPU Programming

Course material for the CSC course "High-Level GPU Programming". The course is
part of the EuroCC training activities at CSC.

## Agenda

### Day 1, Wednesday 27.11

| Time         | Topic |
| ----         | ----- |
| 09:00-09:15  | Welcome
| 09:15-10:00  | Introduction to GPUs
| 10:00-10:40  | GPU execution model
| 10:40-11:30  | GPU memory hierarchy
| 11:20-12:00  | [Mahti and LUMI Computing Platforms](Exercises_Instructions.md)
| 12:00-13:00  | Lunch break
| 13:00-13:30  | Refresher of C++ concepts
| 13:30-15:00  | Standard C++ parallelism
| 14:00-15:00  | Kokkos and Kokkos exercises
| 15:00-15:30  | Coffee break
| 15:30-16:45  | Kokkos and Kokkos exercises
| 16:45-17:00  | Day 1 wrap-up

### Day 2, Thursday 28.11

| Time         | Topic |
| ----         | ----- |
| 09:00-10:30  | Kokkos and Kokkos exercises
| 10:30-12:00  | SYCL Essentials 
| 12:00-13:00  | Lunch break
| 13:00-14:00  | SYCL exercises (Essentials)
| 13:00-14:00  | SYCL advance features
| 15:00-15:30  | Coffee break
| 15:30-16:45  | SYCL exercises (Advance Features & Essentials)
| 16:45-17:00  | Day 2 wrap-up

### Day 3, Friday 29.11

| Time         | Topic |
| ----         | ----- |
| 09:00-09:30  | SYCL Review
| 09:30-11:00  | Syclomatic (theory & heat equation demo)
| 11:00-12:00  | Memory optimizations
| 12:00-13:00  | Lunch break
| 13:00-13:30  | Memory optimization
| 13:30-15:00  | Interoperability with mpi, ping-pong and pi
| 15:00-15:30  | Coffee break
| 15:30-16:00  | Interoperability with third-party libraries
| 15:30-16:45  | Exercises & Bring your own code
| 16:45-17:00  | Day 3 wrap-up & Course closing

The lectures in this repository are published under [CC-BY-SA license](https://creativecommons.org/licenses/by-nc/4.0/). Some of the lectures and exercise are based on Intel copyrighted work and they have their own license.  

Link to [HedgeDoc](https://siili.rahtiapp.fi/High-Level-GPU-Programming)

## Exercises

[General instructions](Exercises_Instructions.md)

### SYCL Essentials
- [Intel DevCloud Intructions](exercises/sycl-optimization-performance-c2s/sycl1/1_Intel_Devcloud_20240203.pdf)
- [SYCL Basics](exercises/sycl-optimization-performance-c2s/sycl1/Readme.md)

### SYCL Performance and Optimization
- [Unified Shared Memory](exercises/sycl-optimization-performance-c2s/sycl2/Readme.md)
- [Profiling on Nvidia platform](exercises/sycl-optimization-performance-c2s/sycl3/NBody-nvidia-profiling/Readme.md)

### SYCL Migrate from CUDA to SYCL
- [CUDA to SYCL migration](exercises/sycl-optimization-performance-c2s/sycl3/Readme.md)

### Syclomatic
- [Heat equation from CUDA to SYCL](exercises/sycl/10-heat-equation-from-cuda/)

### Memory Optimization
- [Jacobi Iterations](exercises/sycl/07-jacobi)
- [Dense matrix matrix multiplication](exercises/sycl/04-matrix-matrix-mul)
- [Reductions](exercises/sycl/06-reduction-direct)

### Kokkos
- [Kokkos](/exercises/kokkos)

### SYCL and MPI
- [Ping-pong with 2 GPUs and MPI](exercises/sycl/08-ping-pong)
- [Pi computing with multiple GPU and MPI](exercises/sycl/11-pi/)
  
### SYCL interoperability

- [SYCL and 3rd party libraries](exercises/sycl/09-interoperability/)
