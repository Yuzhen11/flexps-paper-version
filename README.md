
FlexPS is a Parameter-Server based system that introduces a novel multi-stage abstraction to support flexible parallelism control, which is crucial for the efficient execution of machine learning tasks with dynamic workloads. Distinguishing system designs such as stage scheduler, stage-aware consistency controller, flexible model preparation, and direct model transfer are introduced to support efficient multi-stage machine learning. FlexPS includes optimizations such as customizable parameter partition, customizable KV-server, and repetitive get avoidance to support general machine learning. FlexPS achieves significant speedups and resource saving compared with existing PS systems. 

## Build

Download the source code with submodule Husky.
```sh
git clone --recursive https://github.com/Yuzhen11/husky-45123.git
```

Go to the project root and do an out-of-source build using CMake:
```sh
mkdir debug
cd debug
cmake -DCMAKE_BUILD_TYPE=Debug ..
make help               # List all build target
make $ApplicationName   # Build application
make ClusterManagerMainWithContext # Build ClusterManagerMainWithContext
make -j                 # Build all applications with all threads
```

## Configuration

The config file is almost the same as husky config file with some extra parameters. We need to add the following parameters:

1. cluster_manager_host (the same with master_host)
2. cluster_manager_port (should be a different port from master_port)
3. worker_port (a new port for worker to listen for the instructions)
4. task_scheduler (greedy or sequential)

A sample config file is attached (Make sure to set your hostname and port accordingly):
```
master_host=proj10
master_port=45124
worker_port=12345
comm_port=33244

cluster_manager_host=proj10
cluster_manager_port=45123

hdfs_namenode=proj10
hdfs_namenode_port=9000

serve=0

task_scheduler_type=sequential

[worker]
info=proj10:8
```

## Run a Program

Check the examples in examples directory.

First make sure that the ClusterManagerMainWithContext is running. Use the following to start the ClusterManagerMainWithContext

```sh
./ClusterManagerMainWithContext -C /path/to/your/conf
```

In the distributed environment, use the following to execute workers on all machines,

```sh
./exec.sh <executable> --conf /path/to/your/conf
```

In the single-machine environment, use the following,

```sh
./<executable> --conf /path/to/your/conf
```

