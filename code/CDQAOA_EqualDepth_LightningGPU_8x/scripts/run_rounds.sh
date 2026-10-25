#!/usr/bin/env bash
# Main equal-depth study, one command per round.
#   bash scripts/run_rounds.sh 20            # 20 training instances per cell, both groups
#   bash scripts/run_rounds.sh 40            # later: +20 per cell (first 20 imported, only new ones computed)
#   bash scripts/run_rounds.sh 40 large      # only one group (small | large | n18p6 | both, default both)
#   bash scripts/run_rounds.sh 20 n18p6      # extra: n=18 up to p=6 (p=1-4 imported from large, only p=6 run)
# Groups: small = n 10,12,14,16, p 1,2,3,4,6 ; large = n 18,20,22, p 1,2,3,4.
# Output: results/eqdepth_small_N<N>, results/eqdepth_large_N<N>. Earlier rounds and the older
# stage20/full50 directories are imported automatically. Rerun the SAME command to resume.
# GPU packing per group can be overridden:  OPTS_SMALL="--slots-per-gpu 8" bash scripts/run_rounds.sh 20
# Run inside tmux, conda env active; for MPS first:  source scripts/mps_start.sh 0,1,2,3,4,5,6,7
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
N=${1:?usage: bash scripts/run_rounds.sh N_TRAIN_PER_CELL [small|large|both]}
WHICH=${2:-both}
OPTS_SMALL=${OPTS_SMALL:---slots-per-gpu 4}
OPTS_LARGE=${OPTS_LARGE:---slots-per-gpu 4 --task-weights 18:1,20:1,22:4}

run_group() {
  local group=$1 opts=$2 info cfg out reuse
  info=$(python scripts/23_round_config.py --group "$group" --n-train "$N")
  read -r cfg out <<< "$(sed -n 1p <<< "$info")"
  reuse=$(sed -n 2p <<< "$info"); reuse=${reuse#REUSE=}
  echo "=== group $group, $N train/cell: $cfg -> $out ; reuse: ${reuse:-none}"
  # shellcheck disable=SC2086
  if [ -n "$reuse" ]; then
    bash scripts/run_gpu_8x.sh "$cfg" "$out" $opts --reuse-from "$reuse"
  else
    bash scripts/run_gpu_8x.sh "$cfg" "$out" $opts
  fi
}

case "$WHICH" in
  small) run_group small "$OPTS_SMALL" ;;
  large) run_group large "$OPTS_LARGE" ;;
  n18p6) run_group n18p6 "$OPTS_LARGE" ;;
  both)  run_group small "$OPTS_SMALL"; run_group large "$OPTS_LARGE" ;;
  *) echo "second argument must be small, large, n18p6 or both" >&2; exit 2 ;;
esac
echo "=== done. Reports: results/eqdepth_*_N$N/EXPERIMENT_REPORT.md"
echo "Pooled scaling analysis:"
echo "  python scripts/21_eqdepth_concentration.py --run results/eqdepth_small_N$N results/eqdepth_large_N$N --out results/concentration_N$N"
