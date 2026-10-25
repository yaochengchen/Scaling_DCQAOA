# Reproducibility record — final CPU research run

## Machine and software

- Client date/time zone: 2026-10-11, Asia/Shanghai (UTC+08:00).
- Host: Lenovo 83DG, Windows 11 `10.0.26200`, Intel Core i7-14650HX (16 physical cores, 24 logical processors), 34,053,414,912 bytes installed RAM.
- Present but unused for simulation: NVIDIA GeForce RTX 4060 Laptop GPU. All reported experiments are CPU runs.
- Python 3.13.9 (Anaconda build), NumPy 2.3.5, SciPy 1.16.3, NetworkX 3.5, Matplotlib 3.10.6, pytest 8.4.2.
- Independent backends: Qiskit 2.0.0, Qiskit Aer 0.17.0, Qulacs 0.6.14.
- The unpacked source was not a Git repository. Reproducibility therefore relies on the final ZIP SHA256, immutable config hashes, manifests, frozen models, and raw JSON records rather than a commit hash.

## Frozen conventions

- Objective: `H_C = sum_(i,j) J_ij Z_i Z_j`, minimized; no MaxCut constant.
- Mixer: `H_B = sum_i X_i`; initial state `|+>^n`; bit/qubit order is little-endian.
- Layer instruction order: cost, then for each lexicographically ordered edge `exp(-i alpha J Y_i Z_j)` and `exp(-i alpha J Z_i Y_j)`, then mixer.
- Angle packing: all gamma, then all beta, then all alpha. `alpha=0` exactly recovers ordinary QAOA.
- Historical CSV parser: first p entries are gamma and next p are beta; frozen cost sign is −1.
- Target-instance `predict()` reads static features and a frozen JSON only. `tests/test_pipeline.py::test_predict_stays_pure` replaces state/evaluation methods with failures and proves inference does not call them.

## Final verification commands

Run from the project root in Windows PowerShell:

```powershell
python -m pip install -e ".[dev,qiskit]"
python -m pip install qulacs
python -m pytest -q
$env:PYTHONPATH = (Get-Location).Path
python scripts/08_check_scale_law.py
python scripts/00_validate_and_benchmark_backends.py --sizes 6 8 10 --repetitions 5
```

Observed final tests: `12 passed`. Scale covariance errors were zero for s=0.25,0.5,1,2,4. Maximum cross-backend state-vector errors were `1.55e-16` (Qulacs) and `1.13e-15` (Qiskit).

## Reproduce the main locked experiment

Use a new output directory if rerunning from scratch. Reusing the supplied directory intentionally resumes from checkpoints and refuses a changed config.

```powershell
python -m cdtransfer.cli pipeline `
  --config configs/codex_cpu10h.json `
  --out results/reproduced_cpu10h

$env:PYTHONPATH = (Get-Location).Path
python scripts/09_analyze_concentration.py --out results/reproduced_cpu10h
python scripts/10_make_final_analysis.py `
  --out results/reproduced_cpu10h `
  --paper-dir paper_reproduced
python scripts/11_compiled_budget_sensitivity.py --out results/reproduced_cpu10h
```

Supplied main run:

- `results/codex_cpu10h_confirmatory_locked/manifest.json`, config SHA256 `b9addf9e543d3ca0ea491dc416a3c284cdc219d3f2da58fe628842b389bf2fd0`.
- 81 train / 54 validation / 81 test instances; 1,458 offline restart records; 162 held-out instance-depth pairs.
- File-timestamp wall time for the complete main pipeline: 328.37 seconds.
- `offline_optima.json` contains all angles, objective evaluations, optimizer status, static features, seeds, and fingerprints.
- `FROZEN_MODEL.json` is the pre-test freeze boundary; `heldout_results.json` is the raw target evaluation.
- `analysis/final_statistics.json`, `analysis/subgroup_results.csv`, and `analysis/native_gate_counts.json` contain the derived statistics used in the manuscript.

## Additional locked and diagnostic runs

```powershell
python -m cdtransfer.cli pipeline `
  --config configs/codex_fallback_small.json `
  --out results/codex_fallback_confirmatory_locked

python -m cdtransfer.cli pipeline `
  --config configs/codex_crossscale_diagnostic_round2.json `
  --out results/codex_crossscale_diagnostic_round2
```

- Fallback confirmation: 24/16/36 instances, 288 optimization records, 72 test pairs, 28.85 seconds.
- Cross-scale diagnostic: same training/validation design, 12 n=18/20 test instances, 24 test pairs, 643.74 seconds. It is explicitly exploratory because it was initiated after the main result.
- The Qiskit optimization-level-3 rematching under `analysis/compiled_level3_sensitivity` is also explicitly post-confirmatory. It freezes QAOA scaling on validation before test scoring.

## Statistical unit and caveats

- Bootstrap intervals resample held-out instance-depth rows. Because the same instance appears at p=1 and p=2, these intervals do not model within-instance depth correlation; a submission version should add cluster bootstrap by instance as a sensitivity check.
- Graph fingerprints prevent literal graph+weight duplication. They do not prove non-isomorphism, particularly for regular and ring families.
- L-BFGS-B is local. The main run retained three restarts; 1,310/1,458 optimizer calls reported success.
- The primary gate budget is exact for the declared unoptimized decomposition and also matched on a representative circuit under Qiskit optimization levels 0/1. It is not a routed hardware-topology claim.
- No GPU, device noise, or external quantum hardware result is claimed.

## Portable angle assets

- `assets/cd_transfer_table_v1.csv`
- `assets/cd_transfer_table_v1.json`
- `assets/CD_TRANSFER_V1_FROZEN_MODEL.json`

The final archive and its checksum sidecar are in `dist/`. Verify with:

```powershell
Get-FileHash dist/TrainingFree_CDQAOA_Codex_20261011_Complete.zip -Algorithm SHA256
Get-Content dist/TrainingFree_CDQAOA_Codex_20261011_Complete.zip.sha256.txt
```
