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

