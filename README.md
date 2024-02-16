# High-Level GPU Programming

Course material for the CSC course "High-Level GPU Programming". The course is
part of the EuroCC training activities at CSC.

**Zoom:** <https://cscfi.zoom.us/j/67534098583>

## Agenda

### Day 1, Wednesday 14.2.

| Time         | Topic |
| ----         | ----- |
| 09:00-09:15  | Welcome
| 09:15-10:00  | [Introduction to GPUs](docs/01-introduction.pdf)
| 10:00-10:40  | [GPU execution model](docs/02-execution-model.pdf)
| 10:40-11:20  | [GPU memory hierarchy](docs/03-memory-access-hierarchy.pdf)
| 11:20-12:00  | [Refresher of C++ concepts](docs/04-cpp-concepts.pdf)
| 12:00-13:00  | Lunch break
| 13:00-13:30  | [Mahti and LUMI Computing Platforms](Exercises_Instructions.md)
| 13:30-15:00  | [SYCL I](exercises/sycl-optimization-performance-c2s/sycl1/sycl_1_sonersteiner_helsinki_FINAL.pdf)(a)
| 15:00-15:30  | Coffee break
| 15:30-16:45  | SYCL I(b)
| 16:45-17:00  | Day 1 wrap-up

### Day 2, Thursday 15.2.

| Time         | Topic |
| ----         | ----- |
| 09:00-12:00  | [SYCL II](exercises/sycl-optimization-performance-c2s/sycl2/sycl_2_sonersteiner_helsinki_FINAL.pdf)
| 12:00-13:00  | Lunch break
| 13:00-15:00  | [SYCL III](exercises/sycl-optimization-performance-c2s/sycl3/CUDA_To_SYCL_SYCLomatic.pdf)
| 15:00-15:30  | Coffee Break
| 15:30-16:45  | [Heat equation, cuda to sycl demo](exercises/sycl/10-heat-equation-from-cuda/) 
| 16:45-17:00  | Day 2 wrap-up

### Day 3, Friday 16.2.

| Time         | Topic |
| ----         | ----- |
| 09:00-09:30  | [Kokkos](docs/06-kokkos.pdf)
| 09:30-11:00  | [Kokkos exercises](/exercises/kokkos)
| 11:00-12:00  | Memory optimizations [Jacobi Iterations](exercises/sycl/07-jacobi/), [Matrix-Matrix multiply](exercises/sycl/04-matrix-matrix-mul/) & [Reductions](/exercises/sycl/06-reduction-direct/)
| 12:00-13:00  | Lunch break
| 13:00-13:00  | Memory optimization
| 13:30-15:00  | Interoperability with mpi [ping-pong](exercises/sycl/08-ping-pong) & [pi](/exercises/sycl/11-pi/)
| 15:00-15:30  | Coffee Break
| 15:30-16:00  | Interoperability with [third-party libraries](exercises/sycl/09-interoperability/)
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
- 
### SYCL Migrate from CUDA to SYCL
- [CUDA to SYCL migration](exercises/sycl-optimization-performance-c2s/sycl3/Readme.md)

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
 
