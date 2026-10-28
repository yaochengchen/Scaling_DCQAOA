# Scaling Laws for Parameter Setting in Digitized Counterdiabatic QAOA
## Collaborator reproducibility package — 28 October 2026

This package contains the latest accepted manuscript, including the hardware results and all approved language edits through Results, Discussion and Conclusion. Revision colors and deletions have been removed. The main PDF has 14 pages and the supplement has 12 pages. There are eight main figures and seven supplementary figures.

Start with:
- `manuscript/paper/latex_preview_v4_merged/main.pdf`
- `manuscript/paper/latex_preview_v4_merged/supplementary.pdf`
- `RESULTS_INDEX.md` for the figure/data/code map.

## 1. Reproduce all figures from archived data (no quantum simulations)

Use a fresh Python environment. The tested environment is Python 3.13.9 on Windows; dependencies are pinned to the versions used for this handoff. The portable entry points use relative paths and are designed for Windows/Linux/macOS, but only Windows has been execution-tested.

```sh
python -m venv .venv
# Activate .venv using your shell's normal activation command.
python -m pip install -r requirements-figures.txt
python reproduce.py verify
python reproduce.py figures
python reproduce.py build
```

Run these commands from the extracted package root. An installed TeX distribution with `pdflatex`, TikZ/PGF, `standalone`, and standard AMS/graphics packages is needed for Figure 1 and manuscript compilation. The IEEE class files and logo assets are included. No Qulacs or GPU is needed by the portable commands.

Outputs:
- `reproduced/figures/`: all **15** publication figures, with the exact filenames used by the paper.
- `reproduced/paper/main.pdf` and `supplementary.pdf`: rebuilt papers using those generated figures.
- `figures/` and `editing_audit/`: generated working plots/statistics (not archived inputs).

The manuscript directory is never overwritten by these commands. `build` compiles main twice, then supplementary twice, because supplementary imports main's LaTeX labels. If only `build` is requested, it uses the supplied formal figure PDFs. Run `figures` first to compile from newly generated artwork.

The figure command recomputes the main probability bootstrap checks from paired data in the original manifest order, the depth-averaged/size and family statistics, the supplementary 600-graph bootstrap, and Figure 5 curves/attainment from 720 archived trajectories. It regenerates training-angle coordinates, Figure 6 from 252 chains, and all plots. S2 uses the archived 500-bootstrap exponent table; see the optional refit command below to repeat that bootstrap as well.

This route is sufficient to reproduce the figures and reported aggregates without repeating the long quantum simulations. The supplied publication PDFs are retained as the reference, not copied as substitutes for regeneration (Figure 1 is recompiled from TikZ).

## 2. Refit the power-law coefficients from training angles

```sh
python refit_parameters.py
# Optional, slower: repeat all 500-resample training-exponent bootstrap fits:
python refit_parameters.py --bootstrap
```

The default reconstructs and checks all 64 cost/CD amplitudes and exponents from the three archived training-angle CSV files. The 16 mixer entries and reference degree are retained from the original pooled schedule, as in the paper. Outputs go to `reproduced/parameter_fit/`; the frozen model is not replaced.

`models/U1_power_law.json` is the authoritative current rule for all positive degrees. `models/A_joint_candidate.json` and `inputs/original_model.json` are historical provenance needed to trace construction/refinement; do not substitute their historical low-degree policy for U1. The first-layer S2 confidence intervals characterize the training fit, not independent transfer validation.

## 3. Optional: rerun probability circuits with PennyLane

```sh
python -m pip install -r requirements-simulation.txt
# Short checks:
python rerun_probabilities.py --limit 2 --workers 2
python rerun_probabilities.py --depth 6 --limit 1
python rerun_probabilities.py --low-degree --depth 6 --limit 1
# Full rerun of all unique published graph–depth pairs (long):
python rerun_probabilities.py --all --workers 8
```

Default is one pair, not a full overnight run. A pair includes **power-law DC-QAOA and fixed-angle QAOA**. The complete union has 4,850 graph–depth pairs on 1,060 graphs, thus 9,700 method-specific circuits. This excludes historical previous-rule comparisons from new execution.

The runner uses PennyLane `lightning.qubit`, `complex128`, `shots=None`, the original cost–ordered YZ/ZY–mixer circuit, frozen QAOA selection, and the numerical optimum-mask check from the archived implementation. It enumerates the diagonal spectrum, includes all degenerate optima, verifies normalization, and compares every new probability with the frozen paired result (tolerance 1e-8). A mismatch stops the run; do not silently relax it.

Each worker uses one numerical thread. Each completed pair is written atomically to `reproduced/probability_records/`. Repeating the same command verifies and reuses completed files; at worst an interrupted current pair must be recomputed. Do not start two independent supervisors into the same output folder. Limit worker count to your memory budget: large statevectors require appreciable RAM even when CPU cores are free. A network disconnect does not remove saved local records, but keep the process in a persistent local terminal or `tmux`/`screen` on a remote host. Sleep/shutdown stops computation; restart the command after resuming.

## 4. Optional: rerun the energy optimization experiments

```sh
python rerun_optimization.py benchmark --limit 1
python rerun_optimization.py refinement --limit 1
# Long runs; each resumes completed task files:
python rerun_optimization.py benchmark --all
python rerun_optimization.py refinement --all
```

These commands run the same PennyLane circuits using the archived starts/protocols and write separate outputs. Benchmark: 720 runs, 500 energy evaluations maximum, three optimizers. Refinement: 12 formula starts plus 240 random starts, analytic adjoint gradients. The two protocols intentionally have different initialization distributions and call accounting.

Optimizer paths/stopping behavior can depend on SciPy/platform versions, even if circuits agree; endpoint drift is recorded, not automatically hidden. Archived trajectories define the exact figures supplied here. These reruns do not replace plotted data automatically. Figure 6 plots the first 60 calls in panel (a) and complete best-so-far endpoints in panel (b).

## 5. Data scope and interpretation

- 1,050 balanced test graphs, depths 1–4: 4,200 paired records.
- Same 600 small graphs, depths 1,2,3,4,6: 3,000 records, including reuse of the first four depths.
- Eleven low-degree graphs, five depths: 55 records; one graph overlaps the main/small samples at all five depths.
- Deduplicated: 1,060 graphs and 4,850 pairs.
- Training angle pools: 7,000 vectors at each depth 1–4 and 5,000 at depth 6, with D>1 filters/bins as documented in Methods II-C and Supplementary S2/S8. Do not change p=6 to 4,000 without rebuilding the model.
- Formal probability results are PennyLane results. Old energy/Qulacs fields and previous-rule columns in historical manifests/CSVs are provenance, not additional methods in the current plots.
- Comparators were selected by **mean normalized Ising energy**, not by maximizing test optimal-solution probability.
- Probability differences are signed paired differences, in percentage points; they are not averages of absolute differences.

`inputs/optimizer_benchmark/runs/` contains historical trajectories whose stored fixed reference predates the current rule; `results/benchmark_12graphs.json` supplies the checked current formula references. The reproduction code applies those references before checking Figure 5 attainment counts. The trajectories themselves did not need reoptimization.

## 6. Verification and limits of this handoff

See `VERIFICATION.md` for checks performed on this release. The packaging checks regenerate figures and rebuild both papers; full circuit and optimizer simulations are not repeated for this release. Optional rerun commands above are included for independent checks.

The package contains retained training vectors, graph/cohort manifests, frozen models, optimizer trajectories, paired probability results, original scientific implementations and portable execution wrappers. It does not include every original GPU training restart or server environment; reproducing the complete historical multi-start GPU training campaign from scratch is distinct from reproducing these fitted rules and figures. It also excludes obsolete manuscripts, review PDFs, unused figure variants, logs, caches, and full-pool intermediate simulation products.

Original code is under `provenance/original_code/` for audit. It may retain historical machine paths and old campaign labels. Use the root commands above for portable execution. `provenance/adaptations.md` explains changes made for portability.

The package is a private collaborator handoff; it does not establish that the new data have already been uploaded publicly. Please confirm that the current full-precision rule and probability datasets are included in the eventual GitHub/Zenodo release before finalizing Code and data availability.

## 7. Hardware records and reproduction

`hardware/` contains the 84-circuit results, the frozen execution plan (graphs and angles), raw measurement counts, circuit-resource tables, transpiled circuits (`.qpy`), calibration information and job metadata for job `dbgjqlgte5oc738iauug`. The redundant shot-by-shot bitstring array is omitted because the full counts are retained. The source submission script is retained for provenance; the reproduction commands do not submit hardware jobs.

`python reproduce.py figures` also runs `code/render_hardware.py`, which recomputes hardware success probabilities and ratios from counts, checks all 84 angle assignments against the frozen rules, and checks ideal values against archived PennyLane results. It generates the paired and resource supplementary figures. `code/render_main_hardware.py` generates the two-panel main hardware figure. No IBM account, credentials or new execution is needed.

The archived raw hardware results are used without readout mitigation. Internal mitigation trials are not part of the manuscript or this package. Re-executing on hardware requires separate access and will not reproduce the same noisy counts exactly. Reading the optional QPY artifact requires a compatible Qiskit installation; it is not required for figure regeneration.

## 8. Current figure numbering

Use `RESULTS_INDEX.md` for the current numbering. Some figure filenames and original scripts retain earlier numbering; this is a filename convention, not a different dataset. Do not use historical figure numbers from source-code comments as manuscript references.
