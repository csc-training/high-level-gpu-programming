# High-Level GPU Programming

Course material for the CSC course "High-Level GPU Programming". The course is
part of the EuroCC training activities at CSC.

## Agenda

### Day 1, Wednesday 14.2.

| Time         | Topic |
| ----         | ----- |
| 09:00-09:15  | Welcome
| 09:15-10:00  | Introduction to GPUs
| 10:00-10:40  | GPU parallel programming model
| 10:40-11:20  | GPU memory hierarchy
| 11:20-12:00  | Refresher of C++ concepts
| 12:00-13:00  | Lunch break
| 13:00-13:30  | [Mahti and LUMI Computing Platforms](Exercises_Instructions.md)
| 13:30-15:00  | SYCL I(a)
| 15:00-15:30  | Coffee break
| 15:30-16:45  | SYCL I(b)
| 16:45-17:00  | Day 1 wrap-up

### Day 2, Thursday 15.2.

| Time         | Topic |
| ----         | ----- |
| 09:00-12:00  | SYCL II
| 12:00-13:00  | Lunch break
| 13:00-15:00  | SYCL III
| 15:00-15:30  | Coffee Break
| 15:30-16:45  | Exercises (on Mahti & LUMI )   
| 16:45-17:00  | Day 2 wrap-up

### Day 3, Friday 16.2.

| Time         | Topic |
| ----         | ----- |
| 09:00-09:30  | Kokkos
| 09:30-11:00  | Kokkos exercises
| 11:00-12:00  | Interoperability with third-party libraries and mpi
| 12:00-13:00  | Lunch break
| 13:00-14:00  | Heat equation, cuda to sycl demo
| 14:00-15:00  | Exercises & Bring your own code
| 15:00-15:30  | Coffee Break
| 15:30-16:45  | Exercises & Bring your own code
| 16:45-17:00  | Day 3 wrap-up & Course closing


Link to [slides](https://kannu.csc.fi/s/gZSBE8DbeEKZjRw)
## Exercises

[General instructions](Exercises_Instructions.md)

### SYCL 

- 

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

### Syclomatic
- [Heat equation from CUDA to SYCL](exercises/sycl/10-heat-equation-from-cuda/)
 
