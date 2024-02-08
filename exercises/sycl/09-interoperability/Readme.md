# Using 3rd party libraries. 
Third-party libraries play a crucial role in GPU programming by providing access to optimized functions and algorithms tailored for specific tasks. These libraries abstract low-level GPU hardware details, enabling developers to focus on algorithm design and application development. 

Intel® oneAPI Math Kernel Library provides provides a comprehensive set of low-level routines for math operations. It is highly optimized and extensively parallelized computing and it has a SYCL interface (it is callable from SYCL codes), but it can be used only on CPUs (bothe Intel and AMD) or Intel GPUs. 
For Nvidia hardware the CUDA toolkit provides a equivalent suit of libraries that are highly optimized (for example cuBlas, cuFFT, etc.) and similarly for AMD hardware we have roc/hip libraries. 
