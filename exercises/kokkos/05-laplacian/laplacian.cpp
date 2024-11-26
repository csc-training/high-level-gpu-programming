#include <cstdio>
#include <cstdlib>
#include <omp.h>

#define NX 4096
#define NY 4096

int main(int argc, char** argv)
{

  int nx, ny;
  if (2 == argc) {
    nx = std::atoi(argv[1]);
    ny = std::atoi(argv[1]);
  } else {
     nx = NX;
     ny = NY;
  }
    
  constexpr int niters = 50;

  // Use static to ensure allocation from heap; allocation from stack can segfault
  static double A[NX][NY];
  static double L[NX][NY];

  double dx = 1.0 / double(nx);
  double dy = 1.0 / double(ny);

  // Initialize arrays
  for (int i = 0; i < nx; i++)
      for (int j = 0; j < ny; j++)
        {
          A[i][j] = (i*dx)*(i*dx) + (j*dy)*(j*dy); 
          L[i][j] = 0.0;
        }

  double t0 = omp_get_wtime();
  // Compute Laplacian
  #pragma nounroll
  for (int iter = 0; iter < niters; iter++)
    for (int i = 1; i < nx-1; i++)
      for (int j = 1; j < ny-1; j++)
        L[i][j] = (A[i-1][j] - 2.0*A[i][j] + A[i+1][j]) / (dx*dx) +
                  (A[i][j-1] - 2.0*A[i][j] + A[i][j+1]) / (dy*dy);

  double t1 = omp_get_wtime();

  // Check the result
  double meanL = 0.0;
  for (int i = 1; i < nx-1; i++)
    for (int j = 1; j < ny-1; j++)
      meanL += L[i][j];

  meanL /= ((nx - 1) * (ny - 1));

  printf("Numerical solution %6.4f\n", meanL);
  printf("Analytical solution %6.4f\n", 4.0);

  printf("Time %7.4f\n", t1 - t0);

  return 0;
}
