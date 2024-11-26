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
  int a=1;
  std::vector<int> X(N), Y(N);

{
  // Create buffers for data
  //TODO
  
  // Submit kernel to initialize X
  q.submit([&](handler& h) {
    //TODO
  });

  // Submit kernel to initialize Y
  q.submit([&](handler& h) {
    //TODO
  });

  // Submit kernel to perform the AXPY operation: a = X + Y
  q.submit([&](handler& h) {
    //TODO
}

  // Check that all outputs match expected value
  bool passed = std::all_of(Y.begin(), Y.end(),
                            [](int i) { return (i == 3); });
  std::cout << ((passed) ? "SUCCESS" : "FAILURE")
            << std::endl;

  return (passed) ? 0 : 1;
}
