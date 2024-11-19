#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <execution>
#include <ranges>
#include <string>
#include <vector>


void init(std::vector<double> &x, std::vector<double> &y)
{
    auto kernel = [=, x = x.data(), y = y.data()](size_t i) {
        x[i] = sin(i) * 2.3;
        y[i] = cos(i) * 1.1;
    };

    using std::begin;
    auto int_range = std::views::iota(0);
    std::for_each_n(std::execution::par_unseq, begin(int_range), size(x), kernel);
}

void daxpy(const double a, const std::vector<double> &x, std::vector<double> &y)
{
    auto kernel = [=](const double x, const double y) {
        return a * x + y;
    };
    std::transform(std::execution::par_unseq, begin(x), end(x), begin(y), begin(y), kernel);
}

bool check(const std::vector<double> &y);

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

    std::vector<double> x(n), y(n);

    // Initialize data on GPU
    init(x, y);

    // Calculate on GPU
    daxpy(a, x, y);

#ifndef BENCHMARK
    // Check correctness; this migrates memory to CPU
    if (check(y)) {
        printf("Correctness OK!\n");
        fflush(stdout);
    } else {
        printf("Correctness ERROR!\n");
        return 1;
    }

    // Migrate memory back go to GPU before measuring performance
    daxpy(a, x, y);
#endif

    // Measure performance
    using clock = std::chrono::high_resolution_clock;
    auto t0 = clock::now();
    for (size_t i = 0; i < nit; i++) {
        daxpy(a, x, y);
    }
    auto t1 = clock::now();
    double time = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count() / 1e9;
    double gflops = 2.0 * n * nit / time / 1e9;
    double gbytess = 3.0 * nbytes * nit / time / 1e9;

#ifndef BENCHMARK
    printf("Time: %.4f s\n", time);
    printf("Performance: %.3f GFLOPS, %.3f GB/s\n", gflops, gbytess);
#else
    printf("%16zu %16.4f %16.3f %16.3f\n", n, (double)nbytes / pow(1024, 2), gflops, gbytess);
#endif

    return 0;
}

bool check(const std::vector<double> &y)
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
