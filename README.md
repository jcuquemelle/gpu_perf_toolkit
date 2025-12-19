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

