#!/usr/bin/env bash
# OPTIONAL. Start CUDA MPS so several processes on one GPU run their kernels concurrently.
# Use with "source":   source scripts/mps_start.sh 0,1,2,3,4,5,6,7
# The training command must run in the SAME shell (it needs the CUDA_MPS_* variables).
gpus=${1:-0,1,2,3,4,5,6,7}
export CUDA_MPS_PIPE_DIRECTORY=${CUDA_MPS_PIPE_DIRECTORY:-$HOME/.cdqaoa-mps/pipe}
export CUDA_MPS_LOG_DIRECTORY=${CUDA_MPS_LOG_DIRECTORY:-$HOME/.cdqaoa-mps/log}
mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
if ! command -v nvidia-cuda-mps-control >/dev/null; then
  echo "nvidia-cuda-mps-control not found; continuing without MPS"; return 0 2>/dev/null || exit 0
fi
CUDA_VISIBLE_DEVICES=$gpus nvidia-cuda-mps-control -d && echo "MPS started on GPUs $gpus (pipe $CUDA_MPS_PIPE_DIRECTORY)"
