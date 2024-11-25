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
  
  // Allocate memory using malloc_managed
  //TODO

  // Initialize input memory on the host
  std::fill(a, a + N, 1);
  std::fill(b, b + N, 2);
  std::fill(c, c + N, 0);
  
  // Submit the kernel to the queue
  q.submit([&](handler& h) {

    h.parallel_for(
      //TODO
      );
  });

  // Wait for the kernel to finish
  q.wait();

  // Check that all outputs match the expected value
  bool passed = std::all_of(c.begin(), c.end(),
                            [](int i) { return (i == 3); });
  std::cout << ((passed) ? "SUCCESS" : "FAILURE") << std::endl;

  // Free the memory
  //TODO

  return (passed) ? 0 : 1;
}
