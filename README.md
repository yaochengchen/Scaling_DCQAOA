# Scaling-Guided Parameter Setting for Digitized Counterdiabatic QAOA

Code, frozen parameter table, and archived results for

> **Scaling-Guided Parameter Setting for Digitized Counterdiabatic QAOA: A Formula-Based Approach to Weighted MaxCut**
> Yaocheng Chen and Jiapei Zhuang

The question: can one frozen angle table, rescaled only by the target graph's mean degree *D* and edge-weight RMS *s_J*, set all (γ, β, α) parameters of first-order digitized counterdiabatic QAOA (DC-QAOA / CD-QAOA) on weighted MaxCut — with **no optimization on the target instance**?

Offline training used 528,000 instance-level optimizations (PennyLane Lightning-GPU, 8 × RTX PRO 6000), covering 10 graph families, 5 positive weight distributions, n = 10–22 and p ≤ 6. The frozen unified table was then evaluated on 7,000 held-out graphs (32,000 graph–depth pairs).

## Main result (held-out, same number of layers)

| Cohort | p | Pairs | QAOA ratio | DC-QAOA ratio (frozen table) | Gain / ΣJ [95% CI] | DC wins |
|---|---:|---:|---:|---:|---:|---:|
| n = 10–16 | 1 | 4,000 | 0.7756 | 0.8400 | 0.1151 [0.1136, 0.1165] | 98.7% |
| n = 10–16 | 2 | 4,000 | 0.8436 | 0.8723 | 0.0504 [0.0489, 0.0519] | 92.4% |
| n = 10–16 | 4 | 4,000 | 0.9035 | 0.9165 | 0.0220 [0.0205, 0.0236] | 88.9% |
| n = 10–16 | 6 | 4,000 | 0.9303 | 0.9417 | 0.0187 [0.0164, 0.0208] | 85.9% |
| n = 18–22 | 1 | 3,000 | 0.7668 | 0.8290 | 0.1111 [0.1096, 0.1126] | 99.2% |
| n = 18–22 | 4 | 3,000 | 0.8910 | 0.9020 | 0.0174 [0.0152, 0.0194] | 84.0% |

Full table: [`analysis/outputs_unified/REPORT.md`](analysis/outputs_unified/REPORT.md). Model SHA-256: `f2be3446fd992a826dcfec8f6048b27a8ccf21fff146d06faa84fd507d7f1eb4`.

## Quick start: angles for a new graph

Pure Python standard library; no quantum simulator needed.

```bash
python exports/cd-qaoa-gpu-transfer/scripts/transfer.py predict \
    --graph exports/cd-qaoa-gpu-transfer/examples/graph.json --p 2
```

The rule (templates exist for p = 1, 2, 3, 4, 6):

```
D = 2|E|/n,   s_J = sqrt(mean J_ij^2),   f_D(D) = arctan(1/sqrt(D-1))  (D > 1; else 1)
gamma_l = gamma_l^unit * f_D(D) / s_J
beta_l  = beta_l^unit
alpha_l = alpha_l^unit / s_J * (D / D0)^(-1/2),   D0 = 3.4545...
```

Unit angles are in [`exports/cd-qaoa-gpu-transfer/assets/unified_gpu_model.json`](exports/cd-qaoa-gpu-transfer/assets/unified_gpu_model.json). Circuit and sign conventions, training ranges and domain limits are in [`exports/cd-qaoa-gpu-transfer/GUIDE_zh.md`](exports/cd-qaoa-gpu-transfer/GUIDE_zh.md) (Chinese). In short: H_C = Σ J_ij Z_i Z_j with J_ij > 0, initial state |+⟩^⊗n, each layer applies e^{-iγH_C}, then e^{-iαJ_ij Y_iZ_j} and e^{-iαJ_ij Z_iY_j} per edge, then e^{-iβH_B}. Check whether your framework's rotation gates halve the angle.

## Repository layout

```
exports/cd-qaoa-gpu-transfer/   frozen unified angle table, transfer.py, training optima (compressed), tests
code/CDQAOA_EqualDepth_LightningGPU_8x/
    cdtransfer/                 circuits, instance generation, optimization, transfer rule, GPU scheduling
    scripts/                    experiment, analysis and audit scripts (numbered); run_gpu_8x.sh, run_rounds.sh
    configs/rounds/             configurations of the three main GPU training runs
    tests/                      pytest suite
analysis/
    summarize_unified_table.py, evaluate_unified_table.py, plot_*.py   held-out evaluation and figures
    compare_optimizers_n12p2.py, pennylane_n12p2.py                    12-graph optimizer comparison
    prr_figures.py              validation grid, angle collapse, fitted exponents, transfer loss
    analyze_eqdepth.py, analyze_largen.py                              earlier equal-depth analysis
    outputs_unified/            T1–T5 tables of the held-out evaluation
    outputs_prr/                outputs of prr_figures.py
    outputs/                    tables/figures of the earlier equal-depth analysis
results/
    gpu_eqdepth/                three GPU training cohorts (manifests, frozen models, restart summaries,
                                held-out results), template aggregation, concentration, n = 20–28 check, benchmarks
    unified_table_heldout.zip   per-graph evaluation of the frozen table (7,000 files) — see below
    unified_variational_n12p2_optimizer_compare_pennylane_starts20_eval500/
                                12 graphs × 20 starts × 3 optimizers; runs.zip holds the 720 trajectories
    prr_formula_init_c64/, prr_formula_init_c128/   L-BFGS-B refinement started from the formula angles
    cpu_stage/                  earlier CPU-stage study
review_tools/reproduce.py       re-verify archived numbers and redraw figures without quantum simulation
```

## Data archives — extract before re-running

Two result directories with many small files are stored as zips. Extract them in place first:

| Archive | Extract to | Needed by |
|---|---|---|
| `results/unified_table_heldout.zip` | `results/unified_table_heldout/` | `analysis/summarize_unified_table.py`, `review_tools/reproduce.py verify` |
| `results/unified_variational_n12p2_optimizer_compare_pennylane_starts20_eval500/runs.zip` | `.../runs/` (same folder) | `review_tools/reproduce.py verify`, `analysis/compare_optimizers_n12p2.py` |

```bash
unzip results/unified_table_heldout.zip -d results/unified_table_heldout
unzip results/unified_variational_n12p2_optimizer_compare_pennylane_starts20_eval500/runs.zip \
      -d results/unified_variational_n12p2_optimizer_compare_pennylane_starts20_eval500/runs
```

Everything else (aggregated tables, frozen models, compressed manifests and held-out results) is available directly.

## Reproducing the analysis

Python 3.11 or 3.12 is recommended.

**Re-check numbers and redraw figures (no quantum simulation):**

```bash
python -m pip install -r review_tools/requirements-review.txt
python review_tools/reproduce.py verify     # rebuilds the 32,000 pairs and checks the optimizer summary
python review_tools/reproduce.py figures    # redraws the figures into review_generated/
python analysis/prr_figures.py              # validation grid, collapse, exponents, transfer loss
```

**Re-evaluate the frozen table on the held-out graphs:** `analysis/evaluate_unified_table.py` (Qulacs), then `analysis/summarize_unified_table.py`. Figures are written to `paper/figures/`, which is created if missing.

**Re-run GPU training:** install the package in `code/CDQAOA_EqualDepth_LightningGPU_8x/`:

```bash
cd code/CDQAOA_EqualDepth_LightningGPU_8x
conda env create -f environment.yml && conda activate cdqaoa
python -m pip install -e '.[gpu,dev]'
bash scripts/run_gpu_8x.sh configs/eqdepth_smoke.json results/eqdepth_smoke      # smoke test
```

The main runs used `configs/rounds/*.json` through `scripts/run_rounds.sh`. Step-by-step server instructions (CUDA ≥ 12.8, cuStateVec, MPS, tmux) are in `README_GPU_等深_中文.md` (Chinese).

### Backends by stage

- Offline GPU training: PennyLane Lightning-GPU.
- Held-out evaluation of the unified table: Qulacs.
- 12-graph optimizer comparison: PennyLane `lightning.qubit` 0.45.0.

The `environment_*.json` file in each result directory records the exact software versions, GPU and driver, and source hashes.

## Scope and caveats

- Templates exist only for p = 1, 2, 3, 4, 6. Training covered n = 10–22 for p ≤ 4 and n = 10–18 for p = 6. Positive weights only.
- The unified table was built from training records only and frozen before the held-out evaluation. However, the test graphs had appeared in earlier per-cohort analyses, so they are not a never-seen test set for the whole project.
- Same-depth comparison: a DC layer contains three times the two-qubit rotations of a QAOA layer.
- Not every per-step training trace or GPU checkpoint is included (several GB). Best optima, per-restart summaries, graph definitions and code are, which is enough to rebuild every reported table.

## Citation

See [`CITATION.cff`](CITATION.cff).

## License

MIT — see [`LICENSE`](LICENSE).
