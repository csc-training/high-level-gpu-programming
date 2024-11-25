---
title:  GPU Execution Model
subtitle: High-Level GPU Programming 
date:     Novermber 2024
lang:     en
---

# GPU Execution Model{.section}

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

On an accelerator:

- we create instances of the same function, **kernels**
- kernel is called typically in implicit loop
- SIMT (single instruction multiple threads)
<small>

```cpp
GPU_K void axpy_(int n, double a, double *x, double *y, int id)
{
        y[id] += a * x[id]; // id<n
}

```
</small>

</div>



# GPU threads

<div class="column">


![](img/work_item.png){.center width=5%}

<div align="center"><small>A thread is running on execution unit</small></div>

</div>

<div class="column">

![](img/amd_simd_lanet.png){.center width=31%} 

<div align="center"><small>The smallest execution unit in a GPU.</small></div>
</div>

- GPU threads are very light execution contexts.
- Threads execute a stream of instructions running on different execution units
- Each thread runs the same **kernel** (SIMT). 
- Each thread processes different elements of the data (SIMD).

# Warp / wavefront

<div class="column">


![](img/sub_group.png){.center width=15%}

<div align="center"><small>Execution is done per warp / wavefront</small></div>

</div>

<div class="column">

![](img/amd_simd_unit.png){.center width=55%} 

<div align="center"><small>Scheme of a SIMD unit in an AMD GPU</small></div>
</div>
- GPU threads are grouped together in hardware level
    - warp (NVIDIA, 32 threads), wavefront (AMD, 64 threads)
- All members of group execute the same instruction
- In the case of branching, each branch is executed sequentially
- Memory accesses are done per group

# Thread blocks

<div class="column">

![](img/work_group.png){.center width=20%}

<div align="center"><small>Thread blocks</small></div>

</div>

<div class="column">
![](img/CU2.png){.center width=20%}

<div align="center"><small>Compute Unit in an AMD GPU.</small></div>
</div>
- Threads are grouped in so called blocks
- Each block is executed in specific unit
    - Streaming multiprocessor, SMP (NVIDIA), compute unit, CU (AMD)
- Maximum number of threads in  a block limited by hardware
- Synchronization and data exchange is possible inside a block


# Grid of thread blocks

<div class="column">

![](img/Grid_threads.png){.center width=33%}

<div align="center"><small>A grid of thread blocks executing the same **kernel**</small></div>

</div>

<div class="column">
![](img/mi100-architecture.png){.center width=52%}

<div align="center"><small>AMD Instinct MI100 architecture (source: AMD)</small></div>
</div>

- Thread blocks are organized into a grid
    - Total number of threads = number of blocks $\mathrm{\times}$ threads per block
- In order to hide latencies, there should be more blocks than SMPs / CUs
- There is no synchronization or data exchange between blocks

# Terminology with different vendors


<table class="docutils align-center" id="id7">
<thead>
<tr class="row-odd"><th class="head"><p>NVIDIA</p></th>
<th class="head"><p>AMD</p></th>
<th class="head"><p>SYCL</p></th>
</tr>
</thead>
<tbody>
<tr class="row-even"><td style="text-align: center"; colspan="2"><p>grid of threads</p></td>
<td><p>NDRange</p></td>
</tr>
<tr class="row-odd"><td style="text-align: center"; colspan="2"><p>block</p></td>
<td><p>work-group</p></td>
</tr>
<tr class="row-even"><td><p>warp</p></td>
<td><p>wavefront</p></td>
<td><p>sub-group</p></td>
</tr>
<tr class="row-odd"><td style="text-align: center"; colspan="2"><p>thread</p></td>
<td><p>work-item</p></td>
</tr>
</tbody>
</table>

# Summary

- GPU hardware has high degree of parallelism
- Many threads execute the same instruction (SIMT), working on different data (SIMD)
- Threads are organized in a grid of blocks 
- All threads within a group (*warp / wavefront*) execute the same instructions
    - branching within a *warp / wavefront* should be avoided
- Memory accesses are done per *warp / wavefront*
- High-level frameworks aim to hide these low level details



