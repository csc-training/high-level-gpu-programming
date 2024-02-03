# Memory Optimization III

We saw in the [previous exercise](04-matrix-matrix-mul/) how local share memory can be used to improve the performance by reducing the amount of extra memory operations and by improving the access pattern. This codes show another way to use the local share memory for reductions.
This material is based on [ENCCS lecture](https://enccs.github.io/gpu-programming/9-non-portable-kernel-models/#reductions) 
