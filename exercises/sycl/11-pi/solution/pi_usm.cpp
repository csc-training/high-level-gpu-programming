
// AdaptiveCpp compilation with cpu & nvidia targets: syclcc -O3 --hipsycl-targets="omp;cuda:sm_80" <code>.cpp 
// AdaptiveCpp compilation with cpu & amd targets:    syclcc -O3 --hipsycl-targets="omp;hip:gfx90a" <code>.cpp 
// OneAPI with cpu & nvidia targets:clang++ -std=c++17 -O3 -fsycl -fsycl-targets=nvptx64-nvidia-cuda,spir64_x86_64 -Xsycl-target-backend=nvptx64-nvidia-cuda --cuda-gpu-arch=sm_80  <code>.cpp 
// OneAPI with cpu & amd targets:      icpx -std=c++17 -O3 -fsycl -fsycl-targets=amdgcn-amd-amdhsa,spir64_x86_64 -Xsycl-target-backend=amdgcn-amd-amdhsa  --offload-arch=gfx90a  <code>.cpp 

// Compilation with mpi
// export  MPI_flags="-I/appl/spack/v017/install-tree/gcc-11.2.0/openmpi-4.1.2-bylozw/include -I/appl/spack/v017/install-tree/gcc-11.2.0/openmpi-4.1.2-bylozw/include/openmpi -I/appl/spack/syslibs/include -pthread -L/appl/spack/v017/install-tree/gcc-11.2.0/openmpi-4.1.2-bylozw/lib -L/appl/spack/syslibs/lib -Wl,-rpath,/appl/spack/v017/install-tree/gcc-8.5.0/gcc-11.2.0-zshp2k/lib/gcc/x86_64-pc-linux-gnu/11.2.0 -Wl,-rpath,/appl/spack/v017/install-tree/gcc-8.5.0/gcc-11.2.0-zshp2k/lib64 -Wl,-rpath -Wl,/appl/spack/v017/install-tree/gcc-11.2.0/openmpi-4.1.2-bylozw/lib -Wl,-rpath -Wl,/appl/spack/syslibs/lib -lmpi"
// clang++ $MPI_flags    -std=c++17 -O3 -fsycl -fsycl-targets=nvptx64-nvidia-cuda,spir64_x86_64 -Xsycl-target-backend=nvptx64-nvidia-cuda --cuda-gpu-arch=sm_80 pi_usm.cpp 
// This is a port of a CUDA code written for the CUDA training at CSC https://github.com/csc-training/CUDA

//clang++ `mpicxx --showme:compiler` `mpicxx --showme:link` -std=c++17 -O3 -fsycl -fsycl-targets=nvptx64-nvidia-cuda,spir64_x86_64 -Xsycl-target-backend=nvptx64-nvidia-cuda --cuda-gpu-arch=sm_80 

#include <cassert>
#include <iostream>
#include <numeric>
#include <vector>
#include <ctime>
#include <chrono>
#include <getopt.h>
#include <mpi.h>


#include <sycl/sycl.hpp>

using namespace sycl;

constexpr int n = 840;


void ShowDevice(queue &q) 
{


  // Output platform and device information.
  
  auto device = q.get_device();
  auto p_name = device.get_platform().get_info<info::platform::name>();
  std::cout << "\t\t\t\tPlatform Name: " << p_name << "\n";
  auto p_version = device.get_platform().get_info<info::platform::version>();
  std::cout << "\t\t\t\tPlatform Version: " << p_version << "\n";
  auto d_name = device.get_info<info::device::name>();
  std::cout << "\t\t\t\tDevice Name: " << d_name << "\n";
  auto max_work_group = device.get_info<info::device::max_work_group_size>();
  std::cout << "\t\t\t\tMax Work Group: " << max_work_group << "\n";
  auto max_compute_units = device.get_info<info::device::max_compute_units>();
  std::cout << "\t\t\t\tMax Compute Units: " << max_compute_units << "\n\n";
}


int main(int argc, char** argv)
{
   int myid=0, ntasks;
   int M=256;
   MPI_Status status;
   char machine_name[50];
   int name_len=0;
   if (MPI_Init(&argc, &argv) != MPI_SUCCESS) 
   {
        std::cout << "Failed to initialize MPI\n";
        exit(-1);
   }
   MPI_Comm_size(MPI_COMM_WORLD, &ntasks);
   MPI_Comm_rank(MPI_COMM_WORLD, &myid);
    // Get the machine name.
    MPI_Get_processor_name(machine_name, &name_len);
   //Check how many gpus per node are present
    property_list q_prof{property::queue::enable_profiling{}, sycl::property::queue::in_order{}};
    auto gpu_devices= device::get_devices(sycl::info::device_type::gpu);
    auto devcount=size( gpu_devices ); 
     // Assign the gpu to each task based on the mpi rank
    queue q{gpu_devices[ntasks%devcount],q_prof};
   
   if (0 == myid) {
      printf("Computing approximation to pi with N=%d\n", n);
      printf("Using %d MPI processes\n", ntasks);
      printf("%zu  GPUs/node were detected\n ", devcount);
      std::cout << "Rank #" << myid << " runs on: " << machine_name  << "\n";
      ShowDevice(q);
   }
   

   int chunksize = n / ntasks;
   int istart = myid * chunksize + 1;
   int istop = (myid + 1) * chunksize;

   // Handle possible uneven division
   int remainder = n % ntasks;
   if (remainder > 0) {
       if (myid < remainder) {
          // Assign this task one element more
          istart += myid;
          istop += myid + 1;
       } else {
          istart += remainder;
          istop += remainder;
       }
    }
         
   double *localpi = malloc_shared<double>(1, q);
   *localpi = 0.0;
   
   // Execute the work in a SYCL kernel 
  q.submit([&](handler &cgh) {
    // Create temporary object describing variables with reduction semantics
    auto sum_acc = reduction(localpi, plus<double>());
    size_t myn=istop-istart+1;
    // A reference to the reducer is passed to the lambda
    cgh.parallel_for(range<1>{myn}, sum_acc,
                    [=](id<1> idx, auto &reducer) 
      { 
         double x=(idx[0]+istart-0.5)/n;
         reducer.combine(1.0/(1.0+x*x)); });
  }).wait();
  
   // Reduction to rank 0
   if (0 == myid) {
      double pi = *localpi;
      for (int i=1; i < ntasks; i++) {
        MPI_Recv(localpi, 1, MPI_DOUBLE, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &status);
        pi += *localpi;
      }
      pi *= 4.0 / n;
      printf("Approximate pi=%18.16f (exact pi=%10.8f)\n", pi, M_PI);
   } else {
      MPI_Send(localpi, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
   }
   // Free the allocated memory
   free(localpi, q);
   MPI_Finalize();

}
