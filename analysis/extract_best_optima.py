"""Compress a run's offline_optima.jsonl (all restarts, hundreds of MB) into two small tables.

    python analysis/extract_best_optima.py results/gpu_eqdepth/eqdepth_n18p6_N20 [more run dirs]

Writes, next to offline_optima.jsonl (or offline_optima.jsonl.gz):
  best_optima.csv.gz     one row per (instance, p, method): best restart's energy/ratio/angles
  restart_summary.csv.gz one row per restart (no angles): energy, init_style, success, calls
Standard library only (no pandas needed), streams the file line by line, so it also runs
directly on the server, e.g. from the GPU package root:
    python extract_best_optima.py results/eqdepth_small_N20 results/eqdepth_large_N20
"""
import csv
import gzip
import json
import sys
from pathlib import Path

BEST_COLS = ["id", "n", "family", "weight_kind", "p", "cd", "restart", "init_style", "energy", "ratio",
             "sum_abs_weights", "success", "message", "eval_calls", "wall_seconds",
             "degree", "rms", "gamma", "beta", "alpha"]
RESTART_COLS = BEST_COLS[:15]


def main(run_dir):
    run = Path(run_dir)
    src = run / "offline_optima.jsonl"
    opener = open
    if not src.exists():
        src = run / "offline_optima.jsonl.gz"
        opener = gzip.open
    if not src.exists():
        raise SystemExit(f"{run}/offline_optima.jsonl(.gz) not found (raw file; copy it from the server first)")
    best, n = {}, 0
    with opener(src, "rt", encoding="utf-8") as f, \
            gzip.open(run / "restart_summary.csv.gz", "wt", newline="", encoding="utf-8") as out:
        w = csv.DictWriter(out, fieldnames=RESTART_COLS)
        w.writeheader()
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            rec = {"id": r["id"], "n": r["n"], "family": r["family"], "weight_kind": r["weight_kind"],
                   "p": r["p"], "cd": bool(r["cd"]), "restart": r["restart"], "init_style": r.get("init_style"),
                   "energy": r["energy"], "ratio": r.get("ratio"),
                   "sum_abs_weights": r["graph_features"]["sum_abs_weights"],
                   "success": bool(r["optimizer_success"]), "message": (r.get("optimizer_message") or "")[:40],
                   "eval_calls": r["eval_calls"], "wall_seconds": r.get("wall_seconds")}
            w.writerow(rec)
            n += 1
            k = (r["id"], r["p"], bool(r["cd"]))
            if k not in best or r["energy"] < best[k]["energy"]:
                b = dict(rec)
                b.update(degree=r["features"]["degree"], rms=r["features"]["rms"],
                         gamma=json.dumps(r["angles"]["gamma"]), beta=json.dumps(r["angles"]["beta"]),
                         alpha=json.dumps(r["angles"]["alpha"]))
                best[k] = b
    with gzip.open(run / "best_optima.csv.gz", "wt", newline="", encoding="utf-8") as out:
        w = csv.DictWriter(out, fieldnames=BEST_COLS)
        w.writeheader()
        w.writerows(best.values())
    print(f"{run.name}: {n} restarts -> {len(best)} best optima")


if __name__ == "__main__":
    for d in sys.argv[1:]:
        main(d)
