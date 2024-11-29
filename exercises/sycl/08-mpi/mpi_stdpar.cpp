#include <algorithm>
#include <cstdio>
#include <execution>
#include <iostream>
#include <ranges>
#include <sstream>
#include <vector>

#include <mpi.h>

template <typename T>
void init(T &x)
{
    auto kernel = [=, x = x.data()](size_t i) {
        x[i] = 1.1 * i;
    };

    using std::begin;
    auto indices = std::views::iota(0);
    std::for_each_n(std::execution::par_unseq, begin(indices), size(x), kernel);
}

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);

    const size_t n = argc > 1 ? (size_t)std::stoll(argv[1]) : 1024;

    int size, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    printf("Hello from MPI rank %d/%d\n", rank, size);

    // Data
    std::vector<double> x(n);

    // Fill with zeros on GPU
    std::fill(std::execution::par_unseq, begin(x), end(x), 0);

    if (rank == 0) {
        // Initialize data on rank 0
        init(x);

        // Send with rank 0
        MPI_Send(x.data(), n, MPI_DOUBLE, 1, 123, MPI_COMM_WORLD);
        printf("Rank %d sent\n", rank);

    } else if (rank == 1) {
        // Receive with rank 1
        MPI_Recv(x.data(), n, MPI_DOUBLE, 0, 123, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        printf("Rank %d received\n", rank);
    }

    // Print; this migrates memory to CPU
    std::stringstream ss;
    ss << "Rank " << rank << " has";
    for (int i = 0; i < std::min(8ul, n); ++i) ss << " " << x[i];
    if (n > 8) ss << " ...";
    ss << "\n";
    std::cout << ss.str();

    MPI_Finalize();
}

