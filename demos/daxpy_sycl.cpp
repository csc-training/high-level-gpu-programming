#include <chrono>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include <sycl/sycl.hpp>


void init(sycl::queue &q, auto &x_buf, auto &y_buf)
{
    q.submit([&](sycl::handler& h) {
        sycl::accessor x{x_buf, h, sycl::read_write};
        sycl::accessor y{y_buf, h, sycl::read_write};

        auto kernel = [=](sycl::id<1> i) {
            x[i] = sin((double)i) * 2.3;
            y[i] = cos((double)i) * 1.1;
        };
        h.parallel_for(sycl::range{x_buf.size()}, kernel);
    });
}

void daxpy(sycl::queue &q, const double a, auto &x_buf, auto &y_buf)
{
    q.submit([&](sycl::handler& h) {
        sycl::accessor x{x_buf, h, sycl::read_only};
        sycl::accessor y{y_buf, h, sycl::read_write};

        auto kernel = [=](sycl::id<1> i) {
            y[i] += a * x[i];
        };
        h.parallel_for(sycl::range{x_buf.size()}, kernel);
    });
}

bool check(const auto &y);

int main(int argc, char *argv[])
{
    const size_t n = argc > 1 ? (size_t)std::stoll(argv[1]) : 1000000;
    const size_t nit = argc > 2 ? (size_t)std::stoll(argv[2]) : 4;
    const size_t nbytes = sizeof(double) * n;
    double a = 3.4;

#ifndef BENCHMARK
    printf("Vector size %zu (%.2f MiB)\n", n, (double)nbytes / (1024*1024));
    fflush(stdout);
#endif

    std::vector<double> x_(n), y_(n);

    // Set up sycl
    sycl::queue q;
    sycl::buffer<double, 1> x(x_.data(), sycl::range<1>(n));
    sycl::buffer<double, 1> y(y_.data(), sycl::range<1>(n));

    // Initialize data on GPU
    init(q, x, y);

    // Calculate on GPU
    daxpy(q, a, x, y);

#ifndef BENCHMARK
    // Check correctness
    {
        sycl::host_accessor h_y{y, sycl::read_only};
        if (check(h_y)) {
            printf("Correctness OK!\n");
            fflush(stdout);
        } else {
            printf("Correctness ERROR!\n");
            return 1;
        }
    }
#endif

    // Measure performance
    q.wait();
    using clock = std::chrono::high_resolution_clock;
    auto t0 = clock::now();
    for (size_t i = 0; i < nit; i++) {
        daxpy(q, a, x, y);
    }
    q.wait();
    auto t1 = clock::now();
    double time = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count() / 1e9;
    double gflops = 2.0 * n * nit / time / 1e9;
    double gbytess = 3.0 * nbytes * nit / time / 1e9;

#ifndef BENCHMARK
    printf("Time: %.4f s\n", time);
    printf("Performance: %.3f GFLOPS, %.3f GiB/s\n", gflops, gbytess);
#else
    printf("%16.4f %16.3f %16.3f\n", (double)nbytes / pow(1024, 2), gflops, gbytess);
#endif

    return 0;
}

bool check(const auto &y)
{
    double tolerance = 1e-6;
    for (size_t i = 0; i < y.size(); i++) {
        double y_ref = 3.4 * sin(i) * 2.3 + cos(i) * 1.1;

        if (abs(y_ref - y[i]) > tolerance) {
            return false;
        }
    }
    return true;
}
