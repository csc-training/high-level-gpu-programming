# Ping-Pong test

In order for this code to work one needs to load the GPU aware mpi:
```
module load openmpi/4.1.2-cuda
```
# MPI ping pong 

Rank 0 sends an array filled with `1` to rank 1, which adds `1` the the values
and sends the data back.

