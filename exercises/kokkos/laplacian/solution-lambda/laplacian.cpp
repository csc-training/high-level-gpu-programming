#include <Kokkos_Core.hpp>
#include <cstdlib>
#include <cstdio>

#define NX 4096
#define NY 4096

int main(int argc, char** argv)
{
  Kokkos::initialize();
  {
  Kokkos::Timer timer;

  int nx, ny;
  if (2 == argc) {
    nx = atoi(argv[1]);
    ny = atoi(argv[1]);
  } else {
     nx = NX;
     ny = NY;
  }
    
  constexpr int niters = 50;

  Kokkos::View<double**> A("A", nx, ny);
  Kokkos::View<double**> L("L", nx, ny);

  double dx = 1.0 / double(nx);
  double dy = 1.0 / double(ny);

  double inv_dx2 = 1.0 / (dx*dx);
  double inv_dy2 = 1.0 / (dy*dy);

  // Initialize arrays
  Kokkos::parallel_for("init",
    Kokkos::MDRangePolicy<Kokkos::Rank<2> >({0, 0}, {nx, ny}), 
        KOKKOS_LAMBDA(const int i, const int j) {
          A(i,j) = (i*dx)*(i*dx) + (j*dy)*(j*dy); 
          L(i,j) = 0.0;
        });

  double t0 = timer.seconds();
  // Compute Laplacian
  #pragma nounroll
  for (int iter = 0; iter < niters; iter++)
  Kokkos::parallel_for("laplacian",
    Kokkos::MDRangePolicy<Kokkos::Rank<2> >({1, 1}, {nx-1, ny-1}), 
        KOKKOS_LAMBDA(const int i, const int j) {
          L(i,j) = (A(i-1,j) - 2.0*A(i,j) + A(i+1,j)) * inv_dx2 +
                   (A(i,j-1) - 2.0*A(i,j) + A(i,j+1)) * inv_dy2;
        });

  double t1 = timer.seconds();

  // Check the result
  double meanL = 0.0;
  Kokkos::parallel_reduce("reduce",
    Kokkos::MDRangePolicy<Kokkos::Rank<2> >({1, 1}, {nx-1, ny-1}), 
        KOKKOS_LAMBDA(const int i, const int j, double& meanL) {
           meanL += L(i,j);
        }, meanL);

  meanL /= ((nx - 1) * (ny - 1));

  printf("Numerical solution %6.4f\n", meanL);
  printf("Analytical solution %6.4f\n", 4.0);

  printf("Time %7.4f\n", t1 - t0);

  }
  Kokkos::finalize();

  return 0;
}
