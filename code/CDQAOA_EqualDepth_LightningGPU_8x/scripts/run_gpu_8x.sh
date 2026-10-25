#!/usr/bin/env bash
# Usage: bash scripts/run_gpu_8x.sh CONFIG OUT_DIR [extra cdtransfer.cli options]
#   extra options: --gpus 0,1,2,3,4,5,6,7  --slots-per-gpu 4  --task-weights 18:1,20:1,22:2,24:4  --reuse-from DIR
# Run inside tmux with the conda env active. Rerun the SAME command to resume after an interruption.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
config=${1:-configs/eqdepth_smoke.json}
out=${2:-results/eqdepth_smoke}
shift $(( $# >= 2 ? 2 : $# ))
mkdir -p "$out"
if [ ! -f "$out/preflight_validation.json" ]; then
  # Lightning-GPU vs NumPy reference (state, energy, adjoint gradient); stops here on mismatch.
  dtype=$(python -c "import json,sys; print(json.load(open(sys.argv[1]))['execution'].get('dtype','complex128'))" "$config")
  python -u scripts/18_lightning_validate_benchmark.py --gpu 0 --dtype "$dtype" --out "$out/preflight_validation.json"
fi
python -u -m cdtransfer.cli pipeline --config "$config" --out "$out" "$@" 2>&1 | tee -a "$out/launcher.log"
