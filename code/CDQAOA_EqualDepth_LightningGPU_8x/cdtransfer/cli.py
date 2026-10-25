"""Usage
  GPU equal-depth study (configs with "comparison": "same_depth"):
    python -m cdtransfer.cli pipeline --config configs/eqdepth_smoke.json --out results/eqdepth_smoke \
        --gpus 0,1,2,3,4,5,6,7 [--slots-per-gpu 4] [--task-weights 18:1,20:1,22:2,24:4] [--reuse-from DIR]
  Original CPU pipeline (other configs): python -m cdtransfer.cli pipeline --config configs/pilot.json --out results/pilot
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
from .instances import load_json


def main():
    parser = argparse.ArgumentParser(description="Training-free CD-QAOA reproducible experiment pipeline")
    parser.add_argument("stage", choices=["generate", "train", "fit", "evaluate", "report", "pipeline"])
    parser.add_argument("--config", default="configs/pilot.json")
    parser.add_argument("--out", default="results/pilot")
    parser.add_argument("--gpus", help="GPU IDs, e.g. 0,1,2,3,4,5,6,7 (default: config execution.gpus)")
    parser.add_argument("--slots-per-gpu", type=int,
                        help="concurrent worker processes per GPU (default: config execution.slots_per_gpu)")
    parser.add_argument("--task-weights",
                        help="slots a task of size n occupies, e.g. 18:1,20:1,22:2,24:4 "
                             "(default: config execution.task_weight_by_n, else automatic)")
    parser.add_argument("--reuse-from", help="import finished optimizations from an older output directory "
                                             "(same instances/maxiter/backend/dtype), e.g. the 20-per-cell run")
    args = parser.parse_args()
    from .gpu_pool import set_process_name
    set_process_name("cdqaoa-launch")      # all our processes: ps -e | grep cdqaoa ; stop: pkill cdqaoa
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(key, "1")
    cfg = load_json(args.config)
    out = Path(args.out)
    from .eqdepth import is_eqdepth, run as run_eqdepth
    if is_eqdepth(cfg):
        run_eqdepth(args.stage, cfg, out, reuse_from=args.reuse_from,
                    runtime={"gpus": args.gpus, "slots_per_gpu": args.slots_per_gpu,
                             "task_weights": args.task_weights})
        return
    if args.gpus or args.slots_per_gpu or args.task_weights or args.reuse_from:
        raise SystemExit("GPU options apply only to configs with \"comparison\": \"same_depth\"")
    from .experiment import create_manifest, train_offline, fit_on_validation, evaluate_heldout, report
    manifest = create_manifest(cfg, out)
    if args.stage == "generate":
        print(f"[generate] Saved {out/'manifest.json'}"); return
    if args.stage in ("train", "pipeline"):
        records = train_offline(manifest, out)
        if args.stage == "train": return
    else:
        trainpath = out/"offline_optima.json"
        if not trainpath.exists(): raise FileNotFoundError(f"Run train first: {trainpath}")
        records = load_json(trainpath)
    if args.stage in ("fit", "pipeline"):
        model = fit_on_validation(manifest, records, out)
        print(f"[fit] frozen model at {out/'FROZEN_MODEL.json'}", flush=True)
        if args.stage == "fit": return
    else:
        modelpath = out/"FROZEN_MODEL.json"
        if not modelpath.exists(): raise FileNotFoundError(f"Run fit first: {modelpath}")
        model = load_json(modelpath)
    if args.stage in ("evaluate", "pipeline"):
        rows = evaluate_heldout(manifest, model, out)
        print(f"[test] {len(rows)} paired rows", flush=True)
        if args.stage == "evaluate": return
    else:
        path = out/"heldout_results.json"
        if not path.exists(): raise FileNotFoundError(f"Run evaluate first: {path}")
        rows = load_json(path)
    result = report(out, manifest, records, model, rows)
    print(json.dumps(result["all"], indent=2), flush=True)
    print(f"[report] {out/'EXPERIMENT_REPORT.md'}")


if __name__ == "__main__":
    main()
