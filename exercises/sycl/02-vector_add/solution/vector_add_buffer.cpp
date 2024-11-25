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

  {
   // Create buffers for data and sum
    buffer<int, 1> a_buf(a.data(), range<1>(N));
    buffer<int, 1> b_buf(b.data(), range<1>(N));
    buffer<int, 1> c_buf(c.data(), range<1>(N));

    // Submit the kernel to the queue
    q.submit([&](handler& h) {
      accessor a{a_buf, h};
      accessor b{b_buf, h};
      accessor c{c_buf, h};

      h.parallel_for(range{N}, [=](id<1> idx) {
        c[idx] = a[idx] + b[idx];
      });
    });
  }

  // Check that all outputs match expected value
  bool passed = std::all_of(c.begin(), c.end(),
                            [](int i) { return (i == 3); });
  std::cout << ((passed) ? "SUCCESS" : "FAILURE")
            << std::endl;
  return (passed) ? 0 : 1;
}
