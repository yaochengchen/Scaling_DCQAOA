# PRR additions (2026-10-24)

Files added to this repository for the PRR revision of the formula-based DC-QAOA paper; nothing else was modified.

- `assets/prr_formula_init_inputs.json` — the 12 held-out graphs of the paper's Fig. 3 (n=12, p=2) with exact extrema, the archived unified-table CD energy, and the frozen unified angle table (sha256 f2be3446…).
- `scripts/26_formula_init_refine.py` — L-BFGS-B (adjoint gradients, `cdtransfer.optimization.run_lbfgs`) started from the frozen-table angles, plus random-start control chains. Run from the repo root with the `cdqaoa` env.
- `scripts/run_formula_init.sh` — runs the complex128 (ftol 1e-10, gtol 1e-5) and the campaign complex64 (ftol 1e-6, gtol 1e-3) configurations on GPUs 0 and 1.
- `scripts/27_analyze_formula_init.py` — figures and summary from the two result directories (default: `../../results/prr_formula_init_*`, i.e. `Clean/results`).

Results (executed 2026-10-24 on one RTX PRO 6000 each): `Clean/results/prr_formula_init_c128/` and `Clean/results/prr_formula_init_c64/`.
Analysis and paper text: `Jiapei_final_draft/PRR_revision_2026-10-24/`.
