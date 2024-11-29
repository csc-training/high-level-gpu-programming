#include <cstdio>
#include <cstdlib>
#include <omp.h>
#include <vector>
#include <sycl/sycl.hpp>

#define NX 4096
#define NY 4096

int main(int argc, char** argv)
{

  int nx, ny;
  if (3 == argc) {
    nx = std::atoi(argv[1]);
    ny = std::atoi(argv[2]);
  } else {
     nx = NX;
     ny = NY;
  }
    
  constexpr int niters = 50;

  sycl::range<2> range_full(nx, ny);
  sycl::buffer<double, 2> A{range_full};
  sycl::buffer<double, 2> L{range_full};

  // L.set_final_data(L_host.data());

  double dx = 1.0 / double(nx);
  double dy = 1.0 / double(ny);

  sycl::queue q;

  // Initialize arrays
  q.submit([&](sycl::handler& h) {

    auto acc_A = A.get_access<sycl::access::mode::write>(h);
    auto acc_L = L.get_access<sycl::access::mode::write>(h);
  
    h.parallel_for(range_full, [=](auto idx) {
      auto i = idx[0];
      auto j = idx[1];
      acc_A[i][j] = (i*dx)*(i*dx) + (j*dy)*(j*dy); 
      acc_L[i][j] = 0.0;
    });
  });

  sycl::range<2> range_inner(nx-2, ny-2);
  double t0 = omp_get_wtime();
  q.submit([&](sycl::handler& h) {

    auto acc_A = A.get_access<sycl::access::mode::read>(h);
    auto acc_L = L.get_access<sycl::access::mode::write>(h);
  
    h.parallel_for(range_inner, [=](auto idx) {
      auto i = idx[0] + 1;
      auto j = idx[1] + 1;
        acc_L[i][j] = (acc_A[i-1][j] - 2.0*acc_A[i][j] + acc_A[i+1][j]) / (dx*dx) +
                  (acc_A[i][j-1] - 2.0*acc_A[i][j] + acc_A[i][j+1]) / (dy*dy);
    });
  });

  q.wait();
  double t1 = omp_get_wtime();

  // Check the result
  //sycl::host_accessor L_host{L};
  auto L_host = L.get_host_access();
  double meanL = 0.0;
  for (int i = 1; i < nx-1; i++)
    for (int j = 1; j < ny-1; j++)
      meanL += L_host[i][j];

  meanL /= ((nx - 1) * (ny - 1));

  printf("Numerical solution %f\n", meanL);
  printf("Analytical solution %6.4f\n", 4.0);

  printf("Time %7.4f\n", t1 - t0);

  return 0;
}
