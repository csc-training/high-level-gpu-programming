# General exercise instructions

## Getting the materials

All course materials, slides and hands-out are available in the github repository. They can be downloaded with the command

```
https://github.com/csc-training/higher-level-gpu-programming.git
```

If you have a GitHub account you can also **Fork** this repository and clone
then your fork. That way you can easily commit and push your own solutions
to exercises.

### Repository structure

The exercise assignments are provided in various `README.md`s.
For most of the exercises, some skeleton codes are provided as starting point. In addition, all of the exercises have exemplary full codes
(that can be compiled and run) in the `solutions` folder. **Note that these are
seldom the only or even the best way to solve the problem.**

## Using supercomputers

Exercises can be carried out using the [LUMI](https://docs.lumi-supercomputer.eu/)  supercomputer, [Mahti](https://docs.csc.fi/computing/systems-mahti/), and [Intel DevCloud](https://console.cloud.intel.com/).

 ![](docs/img/cluster_diagram.jpeg)

LUMI can be accessed via ssh using the provided username and ssh key pair:
```
ssh -i <path-to-private-key> <username>@lumi.csc.fi
```
Mahti can be accessed via ssh using the provided username and CSC password:

```
ssh  <username>@mahti.csc.fi
```
The Intel DevCloud can be acces via the [web interface](https://console.cloud.intel.com/).

### Disk area

The  (computing and storage)  resources can be accessed on on supercomputers via project-based allocation system, where users are granted access based on the specific needs and goals of their projects. Running applications and storage area are directly linked ot this projects. For this event we have been granted access to the training `project_2008874` on Mahti and `project_462000456` on LUMI.

All the exercises in the supercomputers have to be carried out in the **scratch** disk area. The name of the scratch directory can be queried with the commands `csc-workspaces` on Mahti and `lumi-workspaces` onLUMI. As the base directory is shared between members of the project, you should create your own
directory:

on Mahti
```
cd /scratch/project_2008874
mkdir -p $USER
cd $USER
```

on LUMI
```
cd /scratch/project_2008874
mkdir -p $USER
cd $USER
```
The `scratch` area has quota of 1-2TB per project. More than enough for the training. In addition to this other areas are disks areas available. The `projappl/project_xyz` area is faster and can be used for storing the project applications (should not be used for data storage) and on LUMI the so called `flash/project_xyz` disk area can be used for IO intensive runs.

### Editors

For editing program source files you can use e.g. *nano* editor:

```
nano prog.f90
```
(`^` in nano's shortcuts refer to **Ctrl** key, *i.e.* in order to save file and exit editor press `Ctrl+X`)
Also other popular editors such as emacs and vim are available.

## Compilation

Mahti and LUMI have several programming environments. For training, we recommend that you use the two SYCL implementations:

### Intel oneAPI compilers
oneAPI is a collection of tool and library supporting a wide range of programming languange and parallel programming paradigms. It includes a SYCL implementation which supports all  Intel devices (CPUs, FPGAs, and GPUs) and has SYCL plug-ins for targeting Nvidia and AMD GPUs.
In order to use the intel SYCL compiler one has to  set the environment varibles first:

on Mahti:
```
. /projappl/project_2008874/intel/oneapi/setvars.sh --include-intel-llvm
module load cuda # This is needed for compiling sycl code for nvidia gpus
```

on LUMI:
```
. /projappl/project_462000456/intel/oneapi/setvars.sh --include-intel-llvm
```
After this one can load other modules that might be needed for compiling the codes. With the environment set-up we can compile and run the SYCL codes.

On Mahti:
```
icpx -fsycl -fsycl-targets=nvptx64-nvidia-cuda,spir64_x86_64 -Xsycl-target-backend=nvptx64-nvidia-cuda --cuda-gpu-arch=sm_80 <sycl_code>.cpp
```
on LUMI
```
icpx -fsycl -fsycl-targets=amdgcn-amd-amdhsa,spir64_x86_64 -Xsycl-target-backend=amdgcn-amd-amdhsa --offload-arch=gfx90a <sycl_code>.cpp
```
Where `-fsycl` flag indicates that a sycl code is compiled and `-fsycl-targets` instructs the compiler to generate optimized code for both CPU and GPU SYCL devices.

### AdaptiveCpp
This is another SYCL  implementation with support for many type of devices. NO sepcial set-up is needed
on Mahti:
```
/projappl/project_2008874/AdaptiveCpp/bin/acpp -O3 -L/appl/spack/v017/install-tree/gcc-8.5.0/gcc-11.2.0-zshp2k/lib64 <sycl_code>.cpp
```

on LUMI:
```
export LD_PRELOAD=/pfs/lustrep4/appl/lumi/SW/LUMI-22.08/G/EB/rocm/5.3.3/llvm/lib/libomp.so
 /projappl/project_462000456/AdaptiveCpp/bin/acpp -O3 <sycl_code>.cpp
```
AdaptiveCpp was set-up so that on Mahti the `acpp` compiler will generate code for CPU and Nvidia GPUs, while on LUMI for CPU nd AMD GPUs.

### MPI
MPI (Message Passing Interface) is a standardized and portable message-passing standard designed for parallel computing architectures. It allows communication between processes running on separate nodes in a distributed memory environment. MPI plays a pivotal role in the world of High-Performance Computing (HPC), this is why is important to know we could combine SYCL and MPI.

The SYCL implementation do not know anything about MPI. Intel oneAPI contains mpi wrappers, however they were not configure for Mahti and LUMI. Both Mahti and LUMI provide wrappers that can compile applications which use MPI, but they can not compile SYCL codes. We can however extract the MPI related flags and add them to the SYCL compilers.

For exampl on Mahti in order to use CUDA-aware MPI we would first load the modules:
```
module load cuda
module load openmpi/4.1.2-cuda
```
The environment would be setup for compiling a CUDA code which use GPU to GPU communications. We can inspect the `mpicxx` wrapper:
```
$ mpicxx -showme
/appl/spack/v017/install-tree/gcc-8.5.0/gcc-11.2.0-zshp2k/bin/g++ -I/appl/spack/v017/install-tree/gcc-11.2.0/openmpi-4.1.2-bylozw/include -I/appl/spack/v017/install-tree/gcc-11.2.0/openmpi-4.1.2-bylozw/include/openmpi -I/appl/spack/syslibs/include -pthread -L/appl/spack/v017/install-tree/gcc-11.2.0/openmpi-4.1.2-bylozw/lib -L/appl/spack/syslibs/lib -Wl,-rpath,/appl/spack/v017/install-tree/gcc-8.5.0/gcc-11.2.0-zshp2k/lib/gcc/x86_64-pc-linux-gnu/11.2.0 -Wl,-rpath,/appl/spack/v017/install-tree/gcc-8.5.0/gcc-11.2.0-zshp2k/lib64 -Wl,-rpath -Wl,/appl/spack/v017/install-tree/gcc-11.2.0/openmpi-4.1.2-bylozw/lib -Wl,-rpath -Wl,/appl/spack/syslibs/lib -lmpi
```
We note that underneath `mpicxx` is calling `g++` with a lots of MPI related flags. We can obtain and use these programmatically with `mpicxx --showme:compile` and `mpicxx --showme:link`
for compiling the SYCL+MPI codes:
```
icpx -fsycl -fsycl-targets=nvptx64-nvidia-cuda -Xsycl-target-backend=nvptx64-nvidia-cuda --cuda-gpu-arch=sm_80 `mpicxx --showme:compile` `mpicxx --showme:link` <sycl_mpi_code>.cpp
```
or
```
/projappl/project_2008874/AdaptiveCpp/bin/acpp -O3 -L/appl/spack/v017/install-tree/gcc-8.5.0/gcc-11.2.0-zshp2k/lib64 `mpicxx --showme:compile` `mpicxx --showme:link` <sycl_mpi_code>.cpp
```

Similarly on LUMI. First we set up the envinronment as indicated above, then load the modules :
```
module load LUMI/22.08
module load partition/G
#module load Boost/1.79.0-cpeCray-22.08 # This is needed only for AdaptiveCpp
module load rocm/5.3.3
module load cce/16.0.1
```
We also activate the GPU-aware MPI via:
```
export MPICH_GPU_SUPPORT_ENABLED=1
```
Now compile with intel compilers:

```
icpx -fsycl -fsycl-targets=amdgcn-amd-amdhsa,spir64_x86_64 -Xsycl-target-backend=amdgcn-amd-amdhsa  --offload-arch=gfx90a `CC --cray-print-opts=cflags` <sycl_mpi_code>.cpp `CC --cray-print-opts=libs`
```
Or with AdaptiveCpp:
```
export LD_PRELOAD=/pfs/lustrep4/appl/lumi/SW/LUMI-22.08/G/EB/rocm/5.3.3/llvm/lib/libomp.so
/projappl/project_462000456/AdaptiveCpp/bin/acpp -O3  `CC --cray-print-opts=cflags` <sycl_mpi_code>.cpp `CC --cray-print-opts=libs`
```

## Running applications in supercomputers
Programs need to be executed via the batch job system.
```
sbatch job.sh
```
The `job.sh` file contains all the necessary information (number of nodes, tasks per node, cores per taks, number of gpus per node, etc.)  for the `slurm` to execute the program.

### Useful environment variables

Use [`SYCL_PI_TRACE`](https://intel.github.io/llvm-docs/EnvironmentVariables.html#sycl-pi-trace-options) to enable runtime tracing (e.g. device discovery):

    export SYCL_PI_TRACE=1


### Running on Mahti

#### CPU applications
```
#!/bin/bash
#SBATCH --job-name=example
#SBATCH --account=project_2008874
#SBATCH --partition=medium
#SBATCH --reservation=hlgp-cpu-f2024
#SBATCH --time=00:05:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1

srun my_cpu_exe
```

Save the script *e.g.* as `job.sh` and submit it with `sbatch job.sh`.
The output of job will be in file `slurm-xxxxx.out`. You can check the status of your jobs with `squeue -u $USER` and kill possible hanging applications with
`scancel JOBID`.

The reservation `hlgp-cpu-f2024` for parition `medium` is available during the training days and it
is accessible only if the users are part of `project_2008874`.

Some applications use MPI, in this case the number of node and number of tasks per node will have to be adjusted accordingly.

#### GPU applications

When running GPU programs, few changes need to made to the batch job
script. The `partition` is are now different, and one must also request explicitly given number of GPUs per node. As an example, in order to use a
single GPU with single MPI task and a single thread use:
```
#!/bin/bash
#SBATCH --job-name=example
#SBATCH --account=project_2008874
#SBATCH --partition=gpu_medium
#SBATCH --reservation=hlgp-gpu-f2024-wed
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=00:05:00
#SBATCH --gres=gpu:a100:1

srun my_gpu_exe
```
The reservation `hlgp-gpu-f2024-wed` is valid on Wednesday, 15:00 to 17:00. On Thursday we will use `hlgp-gpu-f2024-thu` , while on Friday `hlgp-gpu-f2024-fri`.


### Running on LUMI

LUMI is similar to Mahti.

#### CPU applications

```
#!/bin/bash
#SBATCH --job-name=example
#SBATCH --account=project_462000456
#SBATCH --partition=standard
#SBATCH --reservation=hlgp-cpu-f2024
#SBATCH --time=00:05:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1

srun my_cpu_exe
```


#### GPU applications

```
#!/bin/bash
#SBATCH --job-name=example
#SBATCH --account=project_462000456
#SBATCH --partition=standard-g
#SBATCH --reservation=hlgp-gpu-f2024
#SBATCH --time=00:05:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-node=1

srun my_gpu_exe
```

**NOTE** Some exercises have additional instructions of how to run!
