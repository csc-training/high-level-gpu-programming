
# Mahti

## Compile the intell llvm from source
```
export DPCPP_HOME=/scratch/project_2001659/cristian/llvmIntel/sycl_workspace
cd $DPCPP_HOME
git clone https://github.com/intel/llvm -b sycl

module load python-data
module load gcc/11.2.0
module load cuda/11.5.0
# which nvvc # use this folder
CUDA_LIB_PATH=${CUDA_HOME}/lib64/stubs CC=gcc CXX=g++ python $DPCPP_HOME/llvm/buildbot/configure.py --cuda --cmake-opt="-DCUDA_TOOLKIT_ROOT_DIR=${CUDA_HOME}"

CUDA_LIB_PATH=${CUDA_HOME}/lib64/stubs CC=gcc CXX=g++ python $DPCPP_HOME/llvm/buildbot/compile.py

```

## Compile `sycl` codes 
```
$DPCPP_HOME/llvm/build/bin/clang++ -std=c++17 -O3 -fsycl -fsycl-targets=nvptx64-nvidia-cuda --cuda-path=$CUDA_HOME -L  /appl/spack/v017/install-tree/gcc-8.5.0/gcc-11.2.0-zshp2k/lib64 enumerate_gpu.cpp 

```
## Execution
```
LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$DPCPP_HOME/llvm/build/lib srun  --time=00:15:00 --partition=gputest --account=project_2001498 --nodes=1 --ntasks-per-node=1  --cpus-per-task=1 --gres=gpu:a100:4  ./a.out
```

