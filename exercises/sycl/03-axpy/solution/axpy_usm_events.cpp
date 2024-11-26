// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: MIT
#include <iostream>
#include <sycl/sycl.hpp>
#include <algorithm>
using namespace sycl;

int main() {
  // Set up a queue on any available device
  queue q{default_selector{}};

  // Initialize input and output memory on the host
  constexpr size_t N = 256;
  int a = 1;

  // Allocate Unified Shared Memory (USM) on the device
  int* X = malloc_device<int>(N, q);
  int* Y = malloc_device<int>(N, q);

  // Allocate host memory for results validation
  int* host_Y = new int[N];

  // Initialize X array on the device
  event init_X_event = q.parallel_for(range<1>(N), [=](id<1> idx) {
    X[idx] = 1;  // Initialize X with value 1
  });

  // Initialize Y array on the device, depending on X initialization
  event init_Y_event = q.parallel_for(
      range<1>(N), 
      [=](id<1> idx) {
        Y[idx] = 2;  // Initialize Y with value 2
      });

  // Perform the AXPY operation: Y = X + a * Y, depending on Y initialization
  event axpy_event = q.parallel_for(
      range<1>(N), 
       {init_Y_event,init_Y_event},  // Wait for Y initialization to complete
      [=](id<1> idx) {
        Y[idx] = X[idx] + a * Y[idx];
      });

  // Copy results from device to host memory, depending on AXPY completion
  event memcpy_event = q.memcpy(host_Y, Y, sizeof(int) * N, axpy_event);

  // Wait for all events to complete
  memcpy_event.wait();

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
