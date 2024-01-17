## Usage
After installation:
If the initial set-up was succesful there should be modules files available. 

```
module purge
module use /scratch/project_2008874/spack/share/spack/modules/linux-rhel8-x86_64_v3/
module load gcc/11.2.0-gcc-8.5.0-gpvckmb 
module load hipsycl
export LD_LIBRARY_PATH=/scratch/project_2008874/spack/opt/spack/linux-rhel8-x86_64_v3/gcc-8.5.0/gcc-11.2.0-gpvckmbfwjw5vhti5pznbiwpgzap2qld/lib64/:$LD_LIBRARY_PATH
```
Alternatively if there are no modules files, one can use the `spack load <package>`. 
Compile with `syclcc` or `sycl-clang`. 
```
syclcc -O2 --hipsycl-targets="omp;cuda:sm_80" <syl-code>.cpp
```
Run on without gpus  with:
```
srun  --time=00:15:00 --partition=test --account=project_2008874 --nodes=1 --ntasks-per-node=1  --cpus-per-task=6 ./a.out
```
or with gpus:
```
srun  --time=00:15:00 --partition=gputest --account=project_2008874 --nodes=1 --ntasks-per-node=1  --cpus-per-task=1 --gres=gpu:a100:4 ./a.out
```
