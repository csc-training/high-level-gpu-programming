// We use built-in sycl::reduction mechanism in this example.
// The manual implementation of the reduction kernel can be found in folder 06-reduction
#include <sycl/sycl.hpp>

using namespace sycl;

int main() {
  queue q;
  unsigned n = 10;

  // Initialize sum
  int sum = 0;
  {
    // Create a buffer for sum to get the reduction results
    buffer<int> sum_buf{&sum, 1};

    // Submit a SYCL kernel into a queue
    q.submit([&](handler &cgh) {
      // Create temporary object describing variables with reduction semantics
      auto sum_acc = sum_buf.get_access<access_mode::read_write>(cgh);
      // We can use built-in reduction primitive
      auto sum_reduction = reduction(sum_acc, plus<int>());

      // A reference to the reducer is passed to the lambda
      cgh.parallel_for(range<1>{n}, sum_reduction,
                      [=](id<1> idx, auto &reducer) { reducer.combine(idx[0]); });
    }).wait();
    // The contents of sum_buf are copied back to sum by the destructor of sum_buf
  }
  // Print results

  std::cout << "sum = " << sum << "\n";
  bool passed = (sum == (((n-1) * n) / 2));
  std::cout << ((passed) ? "SUCCESS" : "FAILURE") << "\n";
}
