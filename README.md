Gpu Perf Toolkit
=================

This directory contains performance benchmarking tools for Nvidia GPU workloads, focusing on testing
inference with Triton Inference Server.

It provides several scripts to setup the environment, split Gpus in MIG slices, deploy Triton servers with cpu affinity
and run inference benchmarks using several concurrent instances of `perf_analyser` against the Triton servers.

Setup
-----
The installation steps below assume you are using Ubuntu (either bare metal or a VM).

This does not cover installing Nvidia drivers, in order for this to work you need to have a system with Nvidia GPUs
and have the drivers installed (you need to be able to run `nvidia-smi`).

1. Setup docker using `./setup/setup_docker.sh` (run as sudo)
    * Caveat: This does not configure docker rootless access.
2. Setup nvidia-docker using `./setup/setup_nvidia_plugin.sh` (run as sudo)
    * You can alter the `NVIDIA_CONTAINER_TOOLKIT_VERSION` in the script to install a different version if needed.
3. (Optional) Setup MIG partitions on your GPUs using `./setup/setup_mig_mode.sh` (run as sudo)
    * This is a hardcoded script, edit it to suit your needs.

Run Triton instances
---------------------
Multiple triton instances can be run using the provided docker-compose file in the `./triton/` directory.
The docker-compose configures each container to run on a specifc GPU (or MIG slice) and binds
specific CPU cores to each container to achieve a proper Cpu isolation.
Each container is started a separate service, which allows to easily target a specific instance with the perf_analyser
(see related section below)

The triton servers are started in polling mode so that any modification of config.pbtxt in your model repository
is automatically picked up by the server.

1. Provide your Triton model repository as a local folder
2. Edit the `./triton/docker-compose.yaml` file to suit your needs.
    * You can change various env variable defaults, GPU placement and Cpusets for each container.
3. Start all triton instances using `MODEL_REPO=/path/to/your/repo docker-compose -f docker compose.yaml up`
    * **Caveat**: don't use `docker-compose` (not installed if you followed the setup instructions), but `docker compose`
which targets the new docker cli plugin.
    * you can also select which services to start by appending the service names at the end of the command line, e.g.:
`MODEL_REPO=/path/to/your/repo docker-compose -f docker compose.yaml up triton_0 triton_1`

Run Triton instances
---------------------
Multiple triton instances can be run using the provided docker-compose file in the `./triton/` directory.
The docker-compose configures each container to run on a specific GPU (or MIG slice) and binds
specific CPU cores to each container to achieve a proper Cpu isolation.
Each container is started a separate service, which allows to easily target a specific instance with the perf_analyser
(see related section below)

The triton servers are started in polling mode so that any modification of config.pbtxt in your model repository
is automatically picked up by the server.

1. Provide your Triton model repository as a local folder
2. Edit the `./triton/docker-compose.yaml` file to suit your needs.
    * You can change various env variable defaults, GPU placement and Cpusets for each container.
3. Start all triton instances using `MODEL_REPO=/path/to/your/repo docker-compose -f docker compose.yaml up`
    * **Caveat**: don't use `docker-compose` (not installed if you followed the setup instructions), but `docker compose` which
targets the new docker cli plugin.
    * you can also select which services to start by appending the service names at the end of the command line, e.g.:
`MODEL_REPO=/path/to/your/repo docker-compose -f docker compose.yaml up triton_0 triton_1`
    * You can also override env variables from the command line in the same way the MODEL_REPO is set.

Run Perf Analyser benchmarks
--------------------------------
Multiple perf_analyser instances can be run using the provided docker-compose file in the `./perf_analyser/` directory.
Each perf_analyser instance can be configured to target a specific triton server instance (with the TRITON_URL env variable
set for each perf-analyser service).

1. Edit the `./perf_analyser/docker-compose.yaml` file to suit your needs.
2. Start all perf_analyser instances using e.g. `BEGIN=80000 MODEL=<model_loaded_on_triton docker compose -f docker compose.yaml up`
    * As for other docker-compose commands, you can select which services to start by appending the service names at the end of the command line, e.g.:
`BEGIN=80000 MODEL=<model_loaded_on_triton docker compose -f docker-compose.yaml up triton0_perf0 triton0_perf1`
    * with high throughput, it can be useful to have several perf_analyser instances targeting the same triton server instance to saturate it.
   In the proposed file, 3 perf_analyser instances target each triton server, but you can modify that as needed (or select which ones you want to start
   in the docker compose up command line).
   * You can also override env variables from the command line in the same way the BEGIN and MODEL variables are set.


Cpu isolation remarks
----------------------
To achieve proper Cpu isolation between triton servers and perf_analyser instances, the existing docker compose files
use `cpuset` to bind each container to specific Cpu cores.
Make sure that the Cpu cores assigned to the perf_analyser instances do not overlap with the Cpu cores assigned to the triton servers.
You can modify the `cpuset` values in the docker-compose files to adapt them to the actual cpu Ids available in your system.

Another remark (at least) for AMD Turing architecture CPUs: the Cpu cores are grouped in CCDs (Core Complex Die), which contain
a certain number of cores sharing some resources (L3 cache, memory controllers, etc..). We have noted that for workloads where the inference
is not heavy enough, the bottleneck comes more from data management (memory copies, dynamic batching) than from actual computation.
In that case, locating the Triton server on a single CCD (e.g. using a single L3 cache group), even if it means using less cpu cores, can
lead to a performance improvement of around 25% compared to spreading the workload across multiple CCDs.

to know how you cpu cores are organized, you can use the `lscpu -p` command. Example of output:
```
# CPU,Core,Socket,Node,,L1d,L1i,L2,L3
0,0,0,0,,0,0,0,0
1,1,0,0,,1,1,1,0
2,2,0,0,,2,2,2,0
3,3,0,0,,3,3,3,0
4,4,0,0,,4,4,4,0
5,5,0,0,,5,5,5,0
6,6,0,0,,6,6,6,0
7,7,0,0,,7,7,7,0
8,8,0,0,,8,8,8,1
9,9,0,0,,9,9,9,1
10,10,0,0,,10,10,10,1
11,11,0,0,,11,11,11,1
12,12,0,0,,12,12,12,1
(...)
46,46,0,0,,46,46,46,5
47,47,0,0,,47,47,47,5
48,0,0,0,,0,0,0,0
49,1,0,0,,1,1,1,0
50,2,0,0,,2,2,2,0
51,3,0,0,,3,3,3,0
52,4,0,0,,4,4,4,0
53,5,0,0,,5,5,5,0
54,6,0,0,,6,6,6,0
55,7,0,0,,7,7,7,0
56,8,0,0,,8,8,8,1
57,9,0,0,,9,9,9,1
58,10,0,0,,10,10,10,1
59,11,0,0,,11,11,11,1
60,12,0,0,,12,12,12,1
(...)
85,37,0,0,,37,37,37,4
86,38,0,0,,38,38,38,4
87,39,0,0,,39,39,39,4
88,40,0,0,,40,40,40,5
```
first column is the cpu id, second is the core id (same core id means hyperthreading), last column is the L3 cache group.

So as an example, the cpuset `0-7,48-55` binds the container to all cores of CCD 0 only (8 cores, with hyperthreading).
All these cores share the same L3 cache (cache group 0).
cpu 0 and cpu 48 are two hyper-threads sharing the same physical core (core 0).


