// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: MIT
#include <iostream>
#include <sycl/sycl.hpp>
#include <algorithm>
using namespace sycl;

int main() {
  // Set up an in-order queue on any available device
  queue q{default_selector{}, property::queue::in_order{}};

  // Initialize input and output memory on the host
  constexpr size_t N = 256;
  int a = 1;

  // Allocate Unified Shared Memory (USM) on the device
  int* X = malloc_device<int>(N, q);
  int* Y = malloc_device<int>(N, q);

  // Allocate host memory for results validation
  int* host_Y = new int[N];

  // Initialize X and Y arrays on the device
  q.parallel_for(range<1>(N), [=](id<1> idx) {
    X[idx] = 1;  // Initialize X with value 1
  });


  // Initialize X and Y arrays on the device
  q.parallel_for(range<1>(N), [=](id<1> idx) {
    Y[idx] = 2;  // Initialize Y with value 2
  });
  // Perform the AXPY operation: Y = X + a * Y
  q.parallel_for(range<1>(N), [=](id<1> idx) {
    Y[idx] = X[idx] + a * Y[idx];
  });

  // Copy results from device to host memory
  q.memcpy(host_Y, Y, sizeof(int) * N).wait();

  // Check that all outputs match the expected value
  bool passed = std::all_of(host_Y, host_Y + N, [](int i) { return i == 3; });
  std::cout << ((passed) ? "SUCCESS" : "FAILURE") << std::endl;

  // Free device memory
  free(X, q);
  free(Y, q);

  // Free host memory
  delete[] host_Y;

  return (passed) ? 0 : 1;
}
