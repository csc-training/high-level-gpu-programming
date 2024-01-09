# LUMI


## Compile the intel llvm from source

First create folder to install `llvm`

```
module load LUMI/22.08
module load partition/G
module load Boost/1.79.0-cpeCray-22.08
module load rocm/5.3.3
module load cce/16.0.1
module load cray-python
```

```
mkdir <your_scratch_folder>/sycl_workspace
export DPCPP_HOME=<your_scratch_folder>/sycl_workspace
```

Clone the repository:

```
cd $DPCPP_HOME
git clone https://github.com/intel/llvm -b sycl
```

```
export PATH=/appl/lumi/SW/LUMI-22.12/common/EB/buildtools/22.12/bin:$PATH
python $DPCPP_HOME/llvm/buildbot/configure.py --hip --cmake-opt=-DSYCL_BUILD_PI_HIP_ROCM_DIR=/appl/lumi/SW/LUMI-22.08/G/EB/rocm/5.3.3
python $DPCPP_HOME/llvm/buildbot/compile.py
```

