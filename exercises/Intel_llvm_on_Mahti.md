
# Mahti

## Compile the intell llvm from source

First create folder to install `llvm`

```
mkdir <your_scratch_folder>/sycl_workspace
export DPCPP_HOME=<your_scratch_folder>/llvmIntel/sycl_workspace
```

Clone the repository:

```
cd $DPCPP_HOME
git clone https://github.com/intel/llvm -b sycl
```

**Note**! The repository has more than 100k files. On Mahti the we want to keep the applications in `/projappl/project_zzzz` folder which is very fast, but has a limit of `50GB` and `100k` files. 

The compiler uses `cuda` as a back-end and also needs python:

``` 
module load python-data
module load gcc/11.2.0
module load cuda/11.5.0
```

Now the configure and compile:

```
CUDA_LIB_PATH=${CUDA_HOME}/lib64/stubs CC=gcc CXX=g++ python $DPCPP_HOME/llvm/buildbot/configure.py --cuda --cmake-opt="-DCUDA_TOOLKIT_ROOT_DIR=${CUDA_HOME}"

CUDA_LIB_PATH=${CUDA_HOME}/lib64/stubs CC=gcc CXX=g++ python $DPCPP_HOME/llvm/buildbot/compile.py

```

## Compile `sycl` codes 
In order to compile  `sycl` codes first set the environment:

``` 
module load python-data
module load gcc/11.2.0
module load cuda/11.5.0
export DPCPP_HOME=<your_scratch_folder>/llvmIntel/sycl_workspace
```

```
$DPCPP_HOME/llvm/build/bin/clang++ -std=c++17 -O3 -fsycl -fsycl-targets=nvptx64-nvidia-cuda --cuda-path=$CUDA_HOME -L  $GCC_INSTALL_ROOT/lib64 <sycl_code>.cpp 

```
## Execution
```
LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$DPCPP_HOME/llvm/build/lib srun  --time=00:15:00 --partition=gputest --account=project_2001498 --nodes=1 --ntasks-per-node=1  --cpus-per-task=1 --gres=gpu:a100:4  ./a.out
```

