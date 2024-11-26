---
title:    C++ Standard Parallelism
subtitle: High-Level GPU Programming
date:     November 2024
lang:     en
---

# Outline

- Introduction to parallelization with C++ standard library
- Performance considerations of GPU acceleration
- Examples


# Introduction

- ''Traditional'' way of processing data:

```cpp
#include <vector>

double a = 5;
std::vector<double> x = {1, 2, 3, 4}, y(4);

for (int i = 0; i < 4; ++i) {
    y[i] = a * x[i];
}

```

# Introduction cont'd

- Separating computation and iteration:

```cpp
// Kernel: what to do for each data element
auto kernel = [=](const double x) {
    return a * x;
};

// Loop: how to process through all data
for (int i = 0; i < 4; ++i) {
    y[i] = kernel(x[i]);
}

```

# C++ algorithms library

- Algorithms abstract the looping part

```cpp
#include <algorithm>

auto kernel = [=](const double x) {
    return a * x;
};

// Process through all data
std::transform(begin(x), end(x), begin(y), kernel);

```

# C++ standard parallelism

- Since C++17, C++ algorithms have an optional execution policy

```cpp
#include <execution>

// Process through all data in parallel
std::transform(std::execution::par_unseq, begin(x), end(x), begin(y), kernel);

```

- This kernel can now run in parallel on GPU or CPU!
  - Only a suitable compiler needed

- Note! With `std::execution::par_unseq`, the compiler *assumes* that the operations defined by kernel are independent
  - It is the responsibility of the programmer to ensure that this is the case


# Full example

- Code snippets collected from previous slides:

```cpp
#include <algorithm>
#include <execution>
#include <vector>

double a = 5;
std::vector<double> x = {1, 2, 3, 4}, y(4);

auto kernel = [=](const double x) {
    return a * x;
};

std::transform(std::execution::par_unseq, begin(x), end(x), begin(y), kernel);

```

# Notes on GPU acceleration

- The call `std::transform(std::execution::par_unseq, ...)` launches a kernel on GPU
  - Memory copied implicitly to GPU (managed memory)
  - The host CPU thread waits until GPU finishes calculation (see also [P2300](https://wg21.link/p2300))

- Kernel should not hold references to CPU memory to avoid memory access faults or performance hits
  - Capture everything by value: `[=](...) {...}` &rarr; values get copied to GPU memory in kernel launch


# Available C++ algorithms

- C++ standard library has algorithms for generic batch operation, reductions, searching, sorting, ...
  - [List of algorithms](https://en.cppreference.com/w/cpp/algorithm)

- Use existing algorithms when possible
  - Shorter and more efficient code than hand-written custom code

- Example: parallel inner product 

```cpp
std::vector<double> x(N), y(N);

auto prod = std::transform_reduce(std::execution::par_unseq,
                                  begin(x), end(x), begin(y),
                                  0.0, std::plus<>(), std::multiplies<>());

```

# Calling functions in the kernel based on an index

- Example using C++20 views

```cpp
#include <ranges>

std::vector<double> x(N);

auto kernel = [x = x.data()](size_t i) {
    x[i] = func(i);
};

using std::begin;
auto indices = std::views::iota(0);
std::for_each_n(std::execution::par_unseq, begin(indices), size(x), kernel);

```

# Demo: DAXPY

- DAXPY is Level 1 BLAS call
  - `y[i] += a * x[i]` in double precision

- See `demos/daxpy_stdpar.cpp`


# Summary

- C++ standard library enables parallel programming without external dependencies
- Fully portable C++ code
- No full control over GPU
  - No explicit management of GPU memory
    - Simplifies programming
    - Low performance due to suboptimal memory access possible
- Performance depends on the compiler

