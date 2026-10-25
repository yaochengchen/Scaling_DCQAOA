#!/usr/bin/env bash
# Formula-initialized refinement on the 12 Fig.-3 graphs (n=12, p=2).
# Usage (repo root, conda env cdqaoa active, inside tmux):
#   bash scripts/run_formula_init.sh            # both configurations, GPUs 0 and 1 in parallel
# Outputs: results/prr_formula_init_c128/  and  results/prr_formula_init_c64/
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p results/prr_formula_init_c128 results/prr_formula_init_c64

# (1) double precision, tight tolerances (the code's complex128 defaults)
python -u scripts/26_formula_init_refine.py --backend lightning.gpu --dtype complex128 \
    --maxiter 200 --ftol 1e-10 --gtol 1e-5 --random-starts 20 --gpu 0 \
    --out results/prr_formula_init_c128 2>&1 | tee results/prr_formula_init_c128/log.txt &

# (2) exactly the campaign optimizer settings (configs/rounds/eqdepth_small_N20.json)
python -u scripts/26_formula_init_refine.py --backend lightning.gpu --dtype complex64 \
    --maxiter 200 --ftol 1e-6 --gtol 1e-3 --random-starts 20 --gpu 1 \
    --out results/prr_formula_init_c64 2>&1 | tee results/prr_formula_init_c64/log.txt &

wait
echo "done: results/prr_formula_init_c128  results/prr_formula_init_c64"
