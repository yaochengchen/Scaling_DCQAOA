# Current figure / evidence / code map

All commands run from the extracted package root. `python reproduce.py figures` regenerates the following 15 PDFs; `python reproduce.py build` builds the manuscript with those outputs.

|Current figure|PDF filename|Data and renderer|
|---|---|---|
|1|fig01_powerlaw_overview.pdf|U1 schematic; code/figure1/main.tex|
|2|fig02_powerlaw_scaling_p1.pdf|Training CSVs in inputs/; code/render_shared_column_titles.py|
|3|fig03_single_column.pdf|1,050-graph paired probability table; figure3_depth_average_review.py and render_single_column_trial.py|
|4|fig04_hardware_probability.pdf|hardware/tqe_hw_results.csv; code/render_main_hardware.py|
|5|fig04_single_column.pdf|720 benchmark trajectories in inputs/optimizer_benchmark/runs/ and current formula references in results/benchmark_12graphs.json; reproduce.py and render_single_column_trial.py|
|6|fig05_single_column.pdf|252 refinement chains in results/figure5_ajoint_c128_cpu/runs/; render_single_column_trial.py|
|7|fig06_powerlaw_probability_family_depth_averaged_size_1050.pdf|1,050-graph paired table; figure6_depth_average_review.py|
|8|fig07_single_column.pdf|1,050-graph paired table and weight bootstrap; probability_statistics.py and render_single_column_trial.py|
|S1|figS1_powerlaw_scaling_p4.pdf|Training angles; render_shared_column_titles.py|
|S2|figS2_powerlaw_exponents.pdf|inputs/original_effective_exponents.csv; render_exponents.py; refit_parameters.py --bootstrap for refitting|
|S3|figS6_hardware_paired.pdf|Raw hardware counts and execution plan; render_hardware.py|
|S4|figS7_hardware_resources.pdf|Hardware circuit-resource records and ideal probabilities; render_hardware.py|
|S5|figS3_low_degree_probability.pdf|results/pennylane_probability_low_degree_p6_600/low_degree_probability_55.csv; render_probability_supp.py|
|S6|figS4_p6_probability_600.pdf|results/pennylane_probability_low_degree_p6_600/paired_probability_600_five_depths.csv; render_probability_supp.py|
|S7|figS5_probability_size_by_depth.pdf|1,050-graph paired table; figure3_depth_average_review.py|

Frozen coefficients: models/U1_power_law.json. `refit_parameters.py` checks the 64 fitted cost/CD amplitudes and exponents against the training angles. QAOA comparator schedules and graph definitions are embedded in the probability manifests. Optional `rerun_probabilities.py` uses PennyLane; `rerun_optimization.py` reruns the energy-based experiments. Root commands use package-relative paths; historical source paths in provenance do not need to exist.

Supplementary section numbers differ from figure numbers. Follow the latest TeX/PDF for table and section numbering.
