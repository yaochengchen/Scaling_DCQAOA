"""Export robust best-restart concentration diagnostics from saved records."""
from __future__ import annotations
from pathlib import Path
import argparse
from cdtransfer.instances import load_json,dump_json
from cdtransfer.analysis import concentration_by_depth


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--out",default="results/pilot")
    args=ap.parse_args();out=Path(args.out)
    rows=load_json(out/"offline_optima.json")
    summary=concentration_by_depth(rows)
    dump_json(out/"concentration_summary.json",summary)
    print(summary)

if __name__=="__main__":main()
