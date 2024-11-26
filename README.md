# High-Level GPU Programming

Course material for the CSC course "High-Level GPU Programming". The course is
part of the EuroCC training activities at CSC.

## Presentation slides

The slides are available [here](https://csc-training.github.io/high-level-gpu-programming/).

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
| 13:00-13:30  | C++ Refresher
| 13:30-14:00  | C++ Standard Parallelism
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
| 09:30-11:00  | Converting CUDA to SYCL
| 11:00-12:00  | Memory optimizations
| 12:00-13:00  | Lunch break
| 13:00-13:30  | Memory optimization
| 13:30-15:00  | Interoperability with mpi, ping-pong and pi
| 15:00-15:30  | Coffee break
| 15:30-16:00  | Interoperability with third-party libraries
| 15:30-16:45  | Exercises & Bring your own code
| 16:45-17:00  | Day 3 wrap-up & Course closing

The lectures in this repository are published under [CC-BY-SA license](https://creativecommons.org/licenses/by-nc/4.0/). Some of the lectures and exercises are based on Intel copyrighted work and they have their own license ([MIT](https://fedoraproject.org/wiki/Licensing:MIT?rd=Licensing/MIT)).

## Exercises

[General instructions](Exercises_Instructions.md)

### Kokkos
- [Kokkos](/exercises/kokkos)

### SYCL Essentials
- [Hello World](/exercises/sycl/00-hello/)
- [Getting Device Info](/exercises/sycl/01-info/)
- [Vector Addition](/exercises/sycl/02-vector_add)

### SYCL Advance Features
- [Dependencies](exercises/sycl/03-axpy/)
- [Basic Profiling](exercises/sycl/12-basic-profiling)
- [Error Handling](exercises/sycl/13-error-handling/)
  
### SYCL Memory Optimizations
- [Jacobi Iterations](exercises/sycl/07-jacobi)
- [Dense matrix matrix multiplication](exercises/sycl/04-matrix-matrix-mul)
- [Reductions](exercises/sycl/06-reduction-direct) (Optional)

### SYCL and MPI
- [Ping-pong with 2 GPUs and MPI](exercises/sycl/08-ping-pong)
- [Pi computing with multiple GPU and MPI](exercises/sycl/11-pi/)

### SYCL interoperability
- [SYCL and 3rd party libraries](exercises/sycl/09-interoperability/)
