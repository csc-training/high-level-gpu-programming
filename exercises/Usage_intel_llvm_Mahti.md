## Compile `sycl` codes with intel llvm
In order to compile  `sycl` codes first set the environment:

```
module load python-data
module load gcc/11.2.0
module load cuda/11.5.0
export DPCPP_HOME=/scratch/project_2008874/sycl_workspace
```
### Compile the code
```
$DPCPP_HOME/llvm/build/bin/clang++ -std=c++17 -O3 -fsycl -fsycl-targets=nvptx64-nvidia-cuda --cuda-path=$CUDA_HOME -L  $GCC_INSTALL_ROOT/lib64 <sycl_code>.cpp 

```
### Execution
```
LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$DPCPP_HOME/llvm/build/lib srun  --time=00:15:00 --partition=gputest --account=project_2008874 --nodes=1 --ntasks-per-node=1  --cpus-per-task=1 --gres=gpu:a100:4  ./a.out
```

