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
  buffer<int, 1> X_buf(X.data(), range<1>(N));
  buffer<int, 1> Y_buf(Y.data(), range<1>(N));

  // Submit kernel to initialize X
  q.submit([&](handler& h) {
      //Create accessors
    accessor X_acc{X_buf, h,write_only};

    // Kernel to initialize X with value 1
    h.parallel_for(range{N}, [=](id<1> idx) {
      X_acc[idx] = 1;
    });
  });

  // Submit kernel to initialize Y
  q.submit([&](handler& h) {
      //Create accessors
    accessor Y_acc{Y_buf, h,write_only};

    // Kernel to initialize Y with value 2
    h.parallel_for(range{N}, [=](id<1> idx) {
      Y_acc[idx] = 2;
    });
  });

  // Submit kernel to perform the AXPY operation: a = X + Y
  q.submit([&](handler& h) {
      //Create accessors
    accessor X_acc{X_buf, h,read_only};
    accessor Y_acc{Y_buf, h,read_write};
    // Kernel to compute Y = X + Y
    h.parallel_for(range{N}, [=](id<1> idx) {
      Y_acc[idx] = X_acc[idx] + a*Y_acc[idx];
    });
  });
}

  // Check that all outputs match expected value
  bool passed = std::all_of(Y.begin(), Y.end(),
                            [](int i) { return (i == 3); });
  std::cout << ((passed) ? "SUCCESS" : "FAILURE")
            << std::endl;

  return (passed) ? 0 : 1;
}
