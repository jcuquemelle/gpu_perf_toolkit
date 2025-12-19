#!/usr/bin/env bash
# This is hardcoded to configure 2 RTX6000 Gpus
# See https://docs.nvidia.com/datacenter/tesla/mig-user-guide/getting-started-with-mig.html
# for more information on how MIG works and available MIG profiles for your Gpu

# activate MIG on GPU 0
nvidia-smi -i 0 -mig 1
# Create 2 "half-Gpu" Mig slices and associated compute instances on GPU 0
nvidia-smi mig -cgi 2g.48gb-me,2g.48gb-me -C

# activate MIG on GPU 1
nvidia-smi -i 1 -mig 1
# Create 2 "quarter-Gpu" Mig slices and associated compute instances on GPU 1
nvidia-smi mig -i 1 -cgi 1g.24gb,1g.24gb,1g.24gb,1g.24gb -C

