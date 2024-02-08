# Using 3rd party libraries. 
Third-party libraries play a crucial role in GPU programming by providing access to optimized functions and algorithms tailored for specific tasks. These libraries abstract low-level GPU hardware details, enabling developers to focus on algorithm design and application development. 

Intel® oneAPI Math Kernel Library provides provides a comprehensive set of low-level routinesfor math operations. It highly optimized and extensively parallelized computing and it has a SYCL interface (it is callable from SYCL codes), but it can be used only on CPUs (bothe Intel and AMD) or Intel GPUs. 
