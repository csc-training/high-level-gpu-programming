//==============================================================
// Copyright © Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
// =============================================================
#include <CL/sycl.hpp>

using namespace sycl;

int main()
{
  const int N = 16;
    
  //# Initialize a vector and print values
  std::vector<int> v1(N, 11);
  std::vector<int> v2(N, 22);
  std::vector<int> v3(N, 0);

  std::cout<<"\nInput V1: ";    
  for (int i = 0; i < N; i++) std::cout << v1[i] << " ";
  std::cout<<"\nInput V2: ";    
  for (int i = 0; i < N; i++) std::cout << v2[i] << " ";
  std::cout<<"\nInput V3: ";    
  for (int i = 0; i < N; i++) std::cout << v3[i] << " ";
    
  {

    //# STEP 1 : Create buffers for the three vectors
    
    buffer v1_buffer(v1);
    buffer v2_buffer(v2);
    buffer v3_buffer(v3);

    //# Submit task to add vector
    queue q;
    q.submit([&](handler &h)
    {

      //# STEP 2 - create accessors for buffers with access permissions

      accessor v1_accessor (v1_buffer, h, read_only);
      accessor v2_accessor (v2_buffer, h, read_only);
      accessor v3_accessor (v3_buffer, h, write_only);

      h.parallel_for(range<1>(N), [=](id<1> i)
      {

        //# STEP 3 : Implement kernel code to add v3 = v1 + v2

        v3_accessor[i] = v1_accessor[i] + v2_accessor[i];

      });
    });

  }

  //# Print Output values 
  std::cout<<"\nOutput V3: ";
  for (int i = 0; i < N; i++) std::cout<< v3[i] << " ";
  std::cout<<"\n";

  return 0;
}


