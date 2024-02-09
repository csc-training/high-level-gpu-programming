#include <iostream>
#include <sycl/sycl.hpp>

using namespace sycl;

template <typename T>
void axpy(queue &q, const T &a, const std::vector<T> &x, std::vector<T> &y) {
  range<1> N{x.size()};
  buffer x_buf(x.data(), N);
  buffer y_buf(y.data(), N);

  q.submit([&](handler &h) {
    auto x = x_buf.template get_access<access::mode::read>(h);        // accessor x(x_buf, h, read_only);
    auto y = y_buf.template get_access<access::mode::read_write>(h);  // accessor y(y_buf, h, read_write);

    h.parallel_for(N, [=](id<1> i) {
      y[i] += a * x[i];
    });
  });
  q.wait_and_throw();
}


int main() {
  size_t N = 10;

  queue q{property::queue::enable_profiling{}};

  // Call double kernel
  double a = 5.5;
  auto x = std::vector<double>(N, 1.1);
  auto y = std::vector<double>(N, 2.2);

  axpy(q, a, x, y);

  for (int i = 0; i < N; i++){
    std::cout << y[i] << std::endl;
  }

  // Call int kernel
  int ai = 5;
  auto xi = std::vector<int>(N, 1);
  auto yi = std::vector<int>(N, 2);

  axpy(q, ai, xi, yi);

  for (int i = 0; i < N; i++){
    std::cout << yi[i] << std::endl;
  }

  return 0;
}
