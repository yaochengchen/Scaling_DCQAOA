> **等深（CD p 层 对 QAOA p 层）8 卡 GPU 扩展实验：请看 `README_GPU_等深_中文.md`（conda + PennyLane Lightning-GPU）。历史 CPU 结果未改写。**

# Training-Free Explicit CD-QAOA — Reproducible Research Prototype v0.1

**Scope:** Discover transferable fixed-angle rules for *explicit* CD-QAOA; evaluate on held-out instances with **zero target-instance optimization** against fixed-angle standard QAOA at a matched two-qubit decomposition gate budget. **Not** a completed scientific result or a proven CD advantage. GPU scaling explicitly deferred to collaborators.

## 2026-10-11 CPU research closure

The unattended CPU study is complete. The main locked run contains 81/54/81 train/validation/test instances, 1,458 offline optimization records, and 162 held-out pairs. Fixed CD-QAOA improved strongly at the same layer count but was worse than fixed QAOA under matched two-qubit resources in the tested n<=20, p=1/2 regime. Start with `reports/FINAL_RESEARCH_REPORT.md`, `paper/manuscript_draft.md`, and `results/codex_cpu10h_confirmatory_locked/analysis/final_statistics.json`. Portable frozen angles are under `assets/cd_transfer_table_v1.{csv,json}`.

## Quick start (Windows PowerShell)

Recommended: Python 3.10–3.12, CPU, 16 GB RAM for first checks. From the unpacked project root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
python -m pytest -q
python -m cdtransfer.cli pipeline --config configs/pilot.json --out results/pilot_reproduced
python scripts/00_validate_and_benchmark_backends.py --sizes 6 8 10 --repetitions 5
python scripts/08_check_scale_law.py
python scripts/09_analyze_concentration.py --out results/pilot_reproduced
```

If installation is unavailable but NumPy, SciPy, NetworkX and Matplotlib are already installed, use `python -m pytest -q` and `python -m cdtransfer.cli ...` from project root; for `scripts/*.py`, set `$env:PYTHONPATH = (Get-Location).Path` before running them.

With more CPU time:

```powershell
python -m cdtransfer.cli pipeline --config configs/codex_fallback_small.json --out results/fallback_run
python -m cdtransfer.cli pipeline --config configs/codex_cpu10h.json --out results/codex_cpu10h_run
```

**Important:** these last two configurations can be expensive. Prefer the 10-hour Codex supervisor instructions in `CODEX_10H_AUTORUN.md`. Never launch the full `configs/extended.json` blindly; use it only if smaller experiments finish within the time budget.

## Scientific definitions

- Ising objective: `H_C = Σ_(i,j) J_ij Z_i Z_j` to **minimize**. Positive J favors cut edges; no constant is included. Signed-J objectives are energy-minimization tasks, not standard nonnegative MaxCut ratios.
- Mixer: `H_B = Σ_i X_i`; initial state `|+>^n`; **little-endian** bit i = qubit i.
- Explicit CD: `H_CD^raw = (i/2)[H_B,H_C] = Σ J_ij(Y_i Z_j + Z_i Y_j)`.
- Layer: `exp(-i beta H_B) [Π_edges exp(-i alpha J Y_iZ_j) exp(-i alpha J Z_iY_j)] exp(-i gamma H_C)`; the simulator applies **cost first, then CD rotations in listed edge order, then mixer**. Conventional matrix products act right-to-left; the written CD product is an instruction-order specification. Edge order lexicographic. This is **first-order Trotterized**, not exact `exp(-i alpha H_CD)`.
- Angles layout: `[gamma_1...gamma_p, beta_1...beta_p, alpha_1...alpha_p]`. `alpha=0` exactly reduces to ordinary QAOA.
- Hamiltonian **global** weight scaling `J -> sJ` is exactly compensated at the circuit level by `(gamma,alpha)->(gamma/s,alpha/s)` if beta is unchanged. **This exact invariance is not empirical novelty**. Research addresses independent graph/weight-distribution changes and fair-performance implications.
- Historical code uses a different cost evolution sign; CSV parsing is repaired. Default CSV sign is -1, selected among `±1` offline on validation data before testing. See `RESEARCH_PLAN.md` and `assets/original_transfer_data.csv`.

## Data, fitting and leakage rules

1. `generate`: write seeded `manifest.json` with disjoint **labeled instance** fingerprints. Some graphs with fixed structure (notably unit-weight rings) can remain *isomorphic* across partitions despite label changes; restrict or account for them in paper-level OOD claims.
2. `train`: train the **train split only**, including standard QAOA depths `p` and `3p` plus explicit CD depths `p`, with multistart L-BFGS-B. Saves each completed restart atomically to `offline_optima.json` to resume after interruption.
3. `fit`: build weight-normalized medoid templates, select CSV reference sign, fit `alpha_degree_power`, `alpha_multiplier`, and BCH scale **on validation only**; save `FROZEN_MODEL.json`. Incomplete training cannot be frozen.
4. `evaluate`: use `cdtransfer.transfer.predict(inst,p,method,frozen_model)` to predict strictly from the input instance static features and saved template. `predict` never invokes the statevector evaluator. The experiment driver computes energies **after** predicting, solely to score methods.
5. `report`: generate `summary.json`, bootstrap CIs, `EXPERIMENT_REPORT.md`, and plots. Existing test progress checkpoints can be resumed, with no re-fitting from held-out results.

Manual stages:

```powershell
python -m cdtransfer.cli generate --config configs/codex_fallback_small.json --out results/run01
python -m cdtransfer.cli train    --config configs/codex_fallback_small.json --out results/run01
python -m cdtransfer.cli fit      --config configs/codex_fallback_small.json --out results/run01
python -m cdtransfer.cli evaluate --config configs/codex_fallback_small.json --out results/run01
python -m cdtransfer.cli report   --config configs/codex_fallback_small.json --out results/run01
```

**Freeze boundary:** once `FROZEN_MODEL.json` and held-out results exist, do not change hyperparameters based on their test gain. For the next experiment, create a **new** seed, new manifest, new frozen model and clearly mark exploratory vs confirmatory.

## Baselines and proposed models

| Name | What is fixed? | When trained? |
|---|---|---|
| `qaoa_csv` | historical 2305.15201-inspired CSV + RMS/degree scaling | no target training; offline source angle table |
| `qaoa_csv_rms` | same CSV, RMS scaling only | same |
| `qaoa_trained` | medoid of **offline-optimized QAOA** template | train split only |
| `cd_zero` | CD alpha=0; invariant check | none |
| `cd_bch` | gamma/beta transfer; alpha=c gamma beta | c tuned on validation |
| `cd_reference` | medoid of **offline-optimized explicit CD** angles, gamma/alpha RMS-corrected | train split only |
| `cd_proposed` | CD reference + low-dimensional degree exponent and alpha multiplier | validated and frozen |

The strongest fixed QAOA baseline `qaoa_best_budget` is chosen **once per depth on validation** from CSV, RMS-only CSV or trained medoid, and uses a QAOA depth of `3*p_CD`. Target test instances are not used for selection. Offline optima are **diagnostic** and not a deployable zero-training baseline.

## What is and is not fair yet?

Every two-qubit ZZ / YZ / ZY Pauli rotation costs exactly 2 CX in the **defined unoptimized decomposition**. Thus ordinary QAOA depth `3p` has the same abstract 2-CX count as explicit CD depth `p` when all CD alphas are nonzero, with all-to-all couplings and no cancellation/routing. This is a useful controlled **model-budget** comparison, but it is **not an actual native-hardware transpiled result**. `cdtransfer.qiskit_optional.transpile_cost` provides genuine basis transpilation when Qiskit is available; for the paper, report both, under matched native basis/coupling/optimization level.

The project also reports same-depth QAOA and best-of-K shots for the fixed protocols. Time-to-solution / noise comparisons are left to the main research phase.

## Code map

```text
cdtransfer/instances.py         seeded weighted-Ising generator and features
cdtransfer/simulator.py         pure NumPy explicit CD statevector and energy
cdtransfer/transfer.py          CSV parser and zero-target-eval predictors
cdtransfer/optimization.py      OFFLINE L-BFGS-B multi-start only
cdtransfer/experiment.py        split-safe train / val fit / frozen test / report
cdtransfer/resources.py         explicit unoptimized two-qubit gate budget
cdtransfer/qulacs_optional.py   optional independent PauliRotation backend
cdtransfer/qiskit_optional.py   optional circuit + native basis compilation
cdtransfer/plots.py            matplotlib research figures
cdtransfer/cli.py              reproducible multi-stage interface
scripts/00_validate_and_benchmark_backends.py
scripts/08_check_scale_law.py
scripts/09_analyze_concentration.py
configs/                     pilot, smoke, bounded CPU, discovery, extended
assets/                      original CSV and historical competition README
results/pilot/               ACTUALLY EXECUTED pilot, not fabricated
reports/BACKENDS.md          ACTUALLY EXECUTED CPU micro-benchmark
CODEX_10H_AUTORUN.md        unattended Codex execution instructions
CODEX_FINAL_REPORT_TEMPLATE.md
RESEARCH_PLAN.md             full original research plan (copied)
paper/manuscript_draft.md    scaffold; not a finished paper
```

## Known limitations (before submission)

- CPU NumPy backend is correctness/reference code; do not assume it scales to n=24–32 or even n=20 for many optimizations.
- Qulacs/Aer are optional and were **not installed** in the initial test environment; their tests skip automatically. Codex should install when allowed and verify the rotations agree numerically.
- The optimizer is multistart *local* L-BFGS-B, not guaranteed global; optimized-angle variance mixes distinct basins and periodicities.
- Some training/validation unit ring graphs are isomorphic despite distinct labeled fingerprints. Final claims need non-overlapping graph families/degree stratification or graph-isomorphism awareness.
- Medoid is a simple baseline; find equivalent angle branches/optimum families before declaring strong or weak concentration.
- The initial pilot is n=6,8 / p=1 / four held-out instances and **did not favor CD** at equal fixed-decomposition budget. Do not cite this as definitive positive or negative evidence.
- The training, fitting and OOD results, not arbitrary large GPU simulations, are the paper bottleneck.

## Ownership of original assets and references

`assets/original_transfer_data.csv` comes from user-supplied `tutorial_parameter_transfer.rar`. `assets/legacy_competition_README.md` is user-supplied background. `RESEARCH_PLAN.md` reproduces the prior generated plan so the ZIP is standalone. See the cited papers therein. No GPU result is claimed.
