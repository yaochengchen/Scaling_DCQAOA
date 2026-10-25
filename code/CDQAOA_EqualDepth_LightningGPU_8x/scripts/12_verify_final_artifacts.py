"""Fail-fast integrity audit for the final research deliverable."""
from __future__ import annotations

import hashlib
from pathlib import Path

from cdtransfer.analysis import paired_method_summary
from cdtransfer.instances import dump_json, load_json


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "results" / "codex_cpu10h_confirmatory_locked"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    manifest = load_json(RUN / "manifest.json")
    records = load_json(RUN / "offline_optima.json")
    model = load_json(RUN / "FROZEN_MODEL.json")
    heldout = load_json(RUN / "heldout_results.json")
    stats = load_json(RUN / "analysis" / "final_statistics.json")
    assert model["manifest_sha256"] == manifest["config_sha256"]
    assert [len(manifest["splits"][key]) for key in ("train", "val", "test")] == [81, 54, 81]
    assert len(records) == 1458
    record_keys = {(r["id"], r["p"], r["cd"], r["restart"]) for r in records}
    assert len(record_keys) == len(records)
    assert len(heldout) == 162
    heldout_keys = {(r["id"], r["p_cd"]) for r in heldout}
    assert len(heldout_keys) == len(heldout)
    assert all(r["costs_equal_budget"] for r in heldout)
    recalculated = paired_method_summary(
        heldout, "energy_qaoa_best_budget", "energy_cd_proposed")
    recorded = stats["methods_vs_qaoa_best_budget"]["CD proposed"]
    assert abs(recalculated["mean_gain"] - recorded["mean_gain"]) < 1e-15
    assert recalculated["win_rate"] == recorded["win_rate"]

    figures = sorted((ROOT / "paper" / "figures").glob("fig*.png"))
    tables = sorted((ROOT / "paper" / "tables").glob("table*.csv"))
    assert len(figures) == 6 and all(path.stat().st_size > 10_000 for path in figures)
    assert len(tables) == 4 and all(path.stat().st_size > 100 for path in tables)
    assert (ROOT / "assets" / "cd_transfer_table_v1.csv").read_bytes() == \
        (RUN / "cd_transfer_table_v1.csv").read_bytes()
    assert (ROOT / "assets" / "cd_transfer_table_v1.json").read_bytes() == \
        (RUN / "cd_transfer_table_v1.json").read_bytes()
    for relative in ("paper/manuscript_draft.md", "reports/FINAL_RESEARCH_REPORT.md",
                     "reports/REPRODUCIBILITY.md"):
        text = (ROOT / relative).read_text(encoding="utf-8")
        for marker in ("[Fill", "TBD", "XX%", "[measure", "[actual"):
            assert marker not in text, f"Unresolved marker {marker!r} in {relative}"

    key_files = [
        RUN / "manifest.json", RUN / "offline_optima.json", RUN / "FROZEN_MODEL.json",
        RUN / "heldout_results.json", RUN / "analysis" / "final_statistics.json",
        ROOT / "paper" / "manuscript_draft.md", ROOT / "reports" / "FINAL_RESEARCH_REPORT.md",
    ]
    audit = {
        "status": "PASS",
        "main_counts": {"train": 81, "validation": 54, "test": 81,
                        "optimization_records": 1458, "heldout_pairs": 162},
        "figures": [str(path.relative_to(ROOT)) for path in figures],
        "tables": [str(path.relative_to(ROOT)) for path in tables],
        "sha256": {str(path.relative_to(ROOT)): sha256(path) for path in key_files},
    }
    dump_json(ROOT / "reports" / "FINAL_ARTIFACT_AUDIT.json", audit)
    print("FINAL ARTIFACT AUDIT: PASS")
    print(audit["main_counts"])


if __name__ == "__main__":
    main()
