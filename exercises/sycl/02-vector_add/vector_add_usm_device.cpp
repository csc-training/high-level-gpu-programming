// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: MIT
#include <iostream>
#include <sycl/sycl.hpp>
using namespace sycl;

int main() {
  // Set up queue on any available device
  queue q;

  // Initialize input and output memory on the host
  constexpr size_t N = 256;
  std::vector<int> a(N), b(N), c(N);
  std::fill(a.begin(), a.end(), 1);
  std::fill(b.begin(), b.end(), 2);
  std::fill(c.begin(), c.end(), 0);
  
  // Allocate the memory using universal share memory
  int* a_usm = malloc_device<int>(N, q);
  int* b_usm = malloc_device<int>(N, q);
  int* c_usm = malloc_device<int>(N, q);
  
  // Copy data from host to USM
  q.memcpy(a_usm, a.data(), N * sizeof(int)).wait();
  q.memcpy(b_usm, b.data(), N * sizeof(int)).wait();
  q.memcpy(c_usm, c.data(), N * sizeof(int)).wait();
  
  // Submit the kernel to the queue
  q.submit([&](handler& h) {

    h.parallel_for(range{N}, [=](id<1> idx) {
      c_usm[idx] = a_usm[idx] + b_usm[idx];
    });

  });

  // Wait for the kernel to finish
  q.wait();

  // Copy data from USM to host
  q.memcpy(c.data(), c_usm, N * sizeof(int)).wait();

  // Free USM allocations
  free(a_usm, q);
  free(b_usm, q);
  free(c_usm, q);

  // Check that all outputs match the expected value
  bool passed = std::all_of(c.begin(), c.end(),
                            [](int i) { return (i == 3); });
  std::cout << ((passed) ? "SUCCESS" : "FAILURE") << std::endl;
  return (passed) ? 0 : 1;
}
