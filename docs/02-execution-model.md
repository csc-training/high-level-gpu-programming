---
title:  GPU Execution Model
subtitle: Higher Level GPU programming 
author:   CSC Training
date:     2024-02
lang:     en
---


# Accelerator model today


- Local memory in GPU
    - Smaller than main memory (32 GB in Puhti, 64GB in LUMI)
    - Very high bandwidth (up to 3200 GB/s in LUMI)
    - Latency high compared to compute performance

![](img/gpu-bws.png){width=100%}

- GPUs are connected to CPUs via PCIe
- Data must be copied from CPU to GPU over the PCIe bus


# Lumi - Pre-exascale system in Finland

 ![](img/lumi.png){.center width=50%}


# GPU architecture
<div class="column">
- Designed for running tens of thousands of threads simultaneously on
  thousands of cores
- Very small penalty for switching threads
- Running large amounts of threads hides memory access penalties
- Very expensive to synchronize all threads
</div>

<div class="column">
![](img/mi100-architecture.png)
<small>AMD Instinct MI100 architecture (source: AMD)</small>
</div>


#  Heterogeneous Programming Model

- GPUs are co-processors to the CPU
- CPU controls the work flow:
  - *offloads* computations to GPU by launching *kernels*
  - allocates and deallocates the memory on GPUs
  - handles the data transfers between CPU and GPUs
- CPU and GPU can work concurrently
   - kernel launches are normally asynchronous

# Example: axpy

<div class="column">

Serial cpu code of `y=y+a*x`:

- have a loop going over the each index


<small>
```cpp
void axpy_(int n, double a, double *x, double *y)
{
    for(int id=0;id<n; id++) {
        y[id] += a * x[id];
    }
}
```
</small>
</div>

<div class="column">

On a accelerator:

- no loop 
- we create instances of the same function, **kernels**
<small>

```cpp
GPU_K void axpy_(int n, double a, double *x, double *y, int id)
{
        y[id] += a * x[id]; // id<n
}

```
</small>

</div>



# Work-items

<div class="column">


![](img/work_item.png){.center width=5%}

<div align="center"><small>A work-item is running on a simd lane</small></div>

</div>

<div class="column">

![](img/amd_simd_lanet.png){.center width=31%} 

<div align="center"><small>The smallest compuational element in a GPU.</small></div>
</div>

- the work-items are very light execution contexts.
- contain all information needed to execute a stream of instructions.
- for each work-item there is an instance of the **kernel**. 
- each work-item processes different elements of the data (SIMD).

# Sub-Group

<div class="column">


![](img/sub_group.png){.center width=15%}

<div align="center"><small>Execution is done per sub-groups.</small></div>

</div>

<div class="column">

![](img/amd_simd_unit.png){.center width=55%} 

<div align="center"><small>Scheme of a SIMD unit in an AMD GPU.</small></div>
</div>
- the work-items are physically locked in sub-groups
- the size is locked  by hardware, 64 for AMD and 32 for Nvidia GPUs.
- an instruction is executed by all items in the sub-group (in 4 cycles).
- in the case of branching, each branch has to be handled separetely.
- memory accesses are done per sub-group.

# Work-Group

<div class="column">


![](img/work_group.png){.center width=25%}

<div align="center"><small>Work-groups of work-items.</small></div>

</div>

<div class="column">
![](img/CU2.png){.center width=26%}

<div align="center"><small>Compute Unit in an AMD GPU.</small></div>
</div>
- the work-items are divided in groups of fixed size.
- the hardware gives the maximum size, 1024 in GPUS or 8912 in CPUs.
- each work-group is assign to a CU and it can not be split. 
- synchronization and data exchange is possible inside a group.


# Grid of Work-Items

<div class="column">


![](img/Grid_threads.png){.center width=35%}

<div align="center"><small>A grid of work-items executing the same **kernel**</small></div>

</div>

<div class="column">
![](img/mi100-architecture.png){.center width=10%}

<div align="center"><small>AMD Instinct MI100 architecture (source: AMD)</small></div>
</div>

- a grid of threads is created on a specific device to perform the work. 
- each work-item executes the same kernel
- each work-item processes different elements of the data. 
- there is no global synchronization or data exchange.

# Summary
- GPUs are hardware with high degree of parallelism.
- many threads execute the same instruction (SIMD).
- there is a hierarchy of the work-items (*work-groups*, *sub-groups*).
- all items in the sub-group execute the same instruction (in 4 cycles).
- branching in a *sub-group* should be avoided
- memory accesses are done per *sub-group*.

