# Overview

 Many curent HPC systems are heterogeneous  a mix of CPUs, GPUs, FPGAs, and other accelerators. Creating code that delivers optimal performance while remaining portable across a diverse set of platforms can be expensive and time-consuming to achieve the best result.  
 To address this challenge, cross-platform portability ecosystems typically offer a higher-level abstraction layer, which presents a convenient and portable programming model for parallel programming in shared memory environments. These ecosystems can significantly streamline the process of developing, maintaining, and deploying accelerated applications. Their primary objective is to achieve performance portability by allowing developers to write a single-source application. In C++ two (among many) notable   cross-platform portability ecosystems are **Kokkos** and **SYCL**.

 **SYCL** is an open standard specification published by the Khronos Group. This specification defines a unified C++ programming layer that empowers developers to harness modern C++ capabilities across a diverse array of heterogeneous devices. By utilizing support from OpenCL and various other backends, SYCL facilitates parallel execution on a wide spectrum of hardware, encompassing CPUs, GPUs, DSPs, FPGAs, AI accelerators, and custom-designed chips. This capability forms the basis for the development of efficient, portable, and reusable middleware libraries and applications.

 **Kokkos** Core is a C++ programming framework designed to enable the development of high-performance applications that can run seamlessly across various prominent HPC platforms. This framework offers abstractions for concurrent code execution and efficient data management. Kokkos is specifically engineered to address the challenges posed by intricate node architectures featuring multiple memory levels and diverse execution resources. Currently, Kokkos supports a range of backend programming models, including CUDA, HIP, SYCL, HPX, OpenMP, and C++ threads, with ongoing development efforts to incorporate additional backend options.

This course basic to intermediate level on GPU programming using SYCL and Kokkos. It starts with half a day introduction to GPU parallel programming model and C++ concepts. Then it continues with SYCL for two days. Building on this more advance topics are introduced. These enabe the developer to utilize the whole range of features to write portable and performant applications  

# Learning outcomes
After the course, participant will be able to write hardware-agnostic code to express parallelism using SYCL and Kokkos. Manage memory across devices, do basic performance analysis, and evaluate the drawbacks between different approches for programming GPUs.

# Prerequisites and content level:
This course is targets developers who know C++ and would like to learn how to programm GPUs or for developers who are already doing GPU programming using a non-portable approach such like CUDA or HIP and would like to write performant code which runs on various computing platforms. In order to be able to follow the course the participant should be familiar with C++ concepts such as raw and  smart pointers, classes, structures, templates, lambdas, functors, and understand in general the concept of parallel programming.

The content level of the course is broken down as: beginner's - 0%, intermediate - 60%, advanced - 40%, community-targeted content - 0%.

The event is organised at the CSC Training Facilities located in the premises of CSC at Keilaranta 14, Espoo, Finland.

## Agenda 
Day 1, Wednesday 14.02, 9:00-16:00
- Introduction to GPUs and GPU parallle programming model
- C++ concepts
- Introduction to SYCL, queues, command, command groups, kernels
- Memory management in SYCL 

Day 2, Thursday 15.02, 9:00-16:00
- Expressing parallelism
- Dependencies, synchronization

Day 3 Friday 16.02, 9:00-16:00
- SYCL profiling
- Interoperability with other libraries
- Multi-GPU programming
- Kokkos

  
Lecturers: Cristian-Vasile Achim (CSC), Jaro Hokkanen (CSC), Tuomas Rossi (CSC)

Language: English

This event is ON-PREMISE and the participation is limited to 28. The registration is at a first come, first served basis and it closes on February 7 at 23:59!
