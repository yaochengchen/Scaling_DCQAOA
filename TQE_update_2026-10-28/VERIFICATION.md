# Verification — 28 October 2026

- Approved LR01–LR09, LD01–LD04 and LC01–LC02 were merged; no language revision macros remain in current manuscript text.
- Main and supplement compiled twice successfully: 14 and 12 pages. Final logs contain no LaTeX Warning or Overfull messages.
- `python reproduce.py all` succeeded using this package's inputs and portable scripts: graph counts and means verified, all 15 publication figures regenerated, both documents rebuilt using those figures.
- All 15 generated figure PDFs were rasterized with Poppler at a maximum dimension of 1200 pixels and compared with the supplied publication figures: identical pixel arrays for all 15.
- Hardware analysis recomputed all 84 success probabilities and approximation ratios from raw counts, checked all assigned angles, and matched ideal values to the frozen PennyLane table.
- `python refit_parameters.py` succeeded, checking the 64 fitted amplitudes/exponents against the supplied model.
- Full 9,700-circuit simulation and optimizer reruns were not repeated for packaging. Optional portable runners are included. Hardware jobs are not submitted by the reproduction commands.
- Windows with Python 3.13 and the pinned numerical/plotting dependencies was used. Other operating systems were not tested in this release.
- ZIP excludes logs, caches, intermediate regeneration products, historical review PDFs, unused plots, and internal readout-mitigation trials. Full raw counts are retained; the redundant per-shot bitstring array is omitted.
- SHA256.json checks every packaged payload file except itself. The archive also passed ZIP CRC validation and an extracted-package integrity check.
