// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: MIT
#include <iostream>
#include <sycl/sycl.hpp>
using namespace sycl;

int main() {
  // Set up queue on any available device
  //TODO

  // Initialize input and output memory on the host
  constexpr size_t N = 256;
  std::vector<int> a(N), b(N), c(N);
  std::fill(a.begin(), a.end(), 1);
  std::fill(b.begin(), b.end(), 2);
  std::fill(c.begin(), c.end(), 0);
  
  // Allocate the memory using malloc_device
  //TODO
  
  // Copy data from host to USM
  //TODO
  
  // Submit the kernel to the queue
  q.submit([&](handler& h) {

    h.parallel_for(
      //TODO
    );

  });

  // Wait for the kernel to finish
  q.wait();

  // Copy data from USM to host
  //TODO

  // Free USM allocations
  //TODO

  
  // Check that all outputs match the expected value
  bool passed = std::all_of(c.begin(), c.end(),
                            [](int i) { return (i == 3); });
  std::cout << ((passed) ? "SUCCESS" : "FAILURE") << std::endl;
  return (passed) ? 0 : 1;
}
