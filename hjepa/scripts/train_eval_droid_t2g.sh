#!/usr/bin/env bash
# Train one T2G per frozen DROID LeWM, then evaluate on the original 16 clips.
set -euo pipefail
cd "$(dirname "$0")/.."
: "${HJEPA_HOME:?Set HJEPA_HOME to the checkpoint/data root}"

SEEDS=${SEEDS:-"1 1000 10000"}
PLAN_SEEDS=${PLAN_SEEDS:-"1 2 3"}
TRAIN=${TRAIN:-1}
EVAL=${EVAL:-1}
CONFIGS=${CONFIGS:-"droid_flat droid_flat_metric droid_flat_metric_lastn"}
RUNS=${RUNS:-"$HJEPA_HOME/ckpts/droid"}

for seed in $SEEDS; do
  run="$RUNS/droid_lewm/seed${seed}"
  ckpt="$run/droid_lewm_object.ckpt"
  [[ -f "$ckpt" ]] || { echo "Missing LeWM checkpoint: $ckpt" >&2; exit 1; }
  if [[ "$TRAIN" == 1 ]]; then
    python train_temporal_metric.py --config-name metric_droid \
      seed="$seed" policy="$ckpt" hydra.run.dir="$run/metric_hydra"
  fi
  if [[ "$EVAL" == 1 ]]; then
    for config in $CONFIGS; do
      # Separate cost variants and planner seeds: eval skips saved actions.pt.
      out="$run/eval_${config#droid_}"
      for ps in $PLAN_SEEDS; do
        python eval.py --config-name "$config" policy="$ckpt" seed="$ps" \
          output.dir="$out/plan_seed${ps}" hydra.run.dir="$out/plan_seed${ps}/hydra"
      done
    done
  fi
done
