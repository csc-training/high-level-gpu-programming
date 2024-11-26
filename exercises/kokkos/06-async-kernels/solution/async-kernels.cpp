#include <Kokkos_Core.hpp>

int main(int argc, char* argv[]) {

  // Initialize Kokkos
  Kokkos::initialize(argc, argv);

  {
    // Number of regions
    int n = 5;
    // Problem size
    int nx = 20;

    // Allocate on Kokkos default memory space (Unified Memory)
    Kokkos::View<int*, Kokkos::SharedSpace> a("a", nx);

    // Create 'n' execution space instances (maps to streams in CUDA/HIP)
    auto ex = Kokkos::Experimental::partition_space(
      Kokkos::DefaultExecutionSpace(), 1,1,1,1,1);

    // Launch 'n' potentially asynchronous kernels
    // Each kernel has their own execution space instances
    for(int region = 0; region < n; region++) {
      Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(ex[region],
        nx / n * region, nx / n * (region + 1)), KOKKOS_LAMBDA(const int i) {
          a(i) = 100*region + i;
        });
    }

    // Sync execution space instances (maps to streams in CUDA/HIP)
    for(int region = 0; region < n; region++)
      ex[region].fence();

    // Print results
    for (int i = 0; i < nx; i++)
      printf("a(%d) = %d\n", i, a[i]);

  }

  // Finalize Kokkos
  Kokkos::finalize();
  return 0;
}
