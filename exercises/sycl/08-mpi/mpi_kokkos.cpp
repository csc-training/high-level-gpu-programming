#include <cstdio>
#include <iostream>
#include <sstream>
#include <vector>

#include <mpi.h>
#include <Kokkos_Core.hpp>

using std::size;

template <typename T>
void init(T &x)
{
    auto kernel = KOKKOS_LAMBDA(const size_t i) {
        x[i] = 1.1 * i;
    };

    Kokkos::parallel_for(size(x), kernel);
}

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);

    int size, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Kokkos sets device by MPI rank by default (tunable with command line arguments); below is example for setting it manually
//  Kokkos::InitializationSettings settings;
//  settings.set_device_id(rank);
//  Kokkos::initialize(settings);
    Kokkos::initialize(argc, argv);
    {
        const size_t n = argc > 1 ? (size_t)std::stoll(argv[1]) : 1024;

        int device = Kokkos::device_id();
        int count = Kokkos::num_devices();
        printf("Hello from MPI rank %d/%d with GPU %d/%d\n", rank, size, device, count);

        // Device data
        Kokkos::View<double*> x("x", n);

        if (rank == 0) {
            // Initialize data on rank 0
            init(x);

            // Send with rank 0
            Kokkos::fence();
            MPI_Send(x.data(), n, MPI_DOUBLE, 1, 123, MPI_COMM_WORLD);
            printf("Rank %d sent\n", rank);
        } else if (rank == 1) {
            // Receive with rank 1
            Kokkos::fence();
            MPI_Recv(x.data(), n, MPI_DOUBLE, 0, 123, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            printf("Rank %d received\n", rank);
        }

        // Copy result to CPU and print
        auto h_x = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), x);
        std::stringstream ss;
        ss << "Rank " << rank << " has";
        for (int i = 0; i < std::min(8ul, n); ++i) ss << " " << h_x[i];
        if (n > 8) ss << " ...";
        ss << "\n";
        std::cout << ss.str();

    }
    Kokkos::finalize();

    MPI_Finalize();
}

