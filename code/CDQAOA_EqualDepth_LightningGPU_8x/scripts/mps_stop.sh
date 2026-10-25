#!/usr/bin/env bash
# Stop the MPS daemon started by scripts/mps_start.sh
export CUDA_MPS_PIPE_DIRECTORY=${CUDA_MPS_PIPE_DIRECTORY:-$HOME/.cdqaoa-mps/pipe}
echo quit | nvidia-cuda-mps-control && echo "MPS stopped"
