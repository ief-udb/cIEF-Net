#!/usr/bin/env bash
# Source-domain benchmark on A. alternata (paper: benchmark, ablation and
# paired image-level comparisons).
#
# For every seed:
#   1. stage 1, base network (DoRA adapters, decoder and head), 30 epochs;
#   2. stage 2, harmonic refiner (K = 24) trained for 15 epochs on the frozen base;
#   3. three evaluations of the SAME stage-2 checkpoint on the test set:
#        base              refiner off, no containment suppression
#        base_containment  refiner off, containment suppression
#        cief_net          refiner on,  containment suppression   (the model)
#      Because the base is frozen during stage 2, "refiner off" reproduces the
#      stage-1 model exactly.
# Finally, paired image-level tests of cIEF-Net against both ablations, with
# per-image metrics averaged over the seeds.
#
# Usage: bash experiments/01_alternaria.sh      (SEEDS="42" for a single seed)
set -euo pipefail
source "$(dirname "$0")/config.sh"
OUT="$RUNS/alternaria"

for seed in $SEEDS; do
  base="$OUT/base_s$seed"
  ref="$OUT/cief_net_s$seed"
  [[ -f "$base/checkpoint.pt" ]] || run $PY "$REPO/scripts/train.py" --mode holdout \
      --images-dir "$SOURCE_IMAGES" --split-csv "$SOURCE_SPLIT" --out "$base" \
      $ARCH $STAGE1 --seed "$seed" --device "$DEVICE" $EXTRA $EXTRA_TRAIN
  [[ -f "$ref/checkpoint.pt" ]] || run $PY "$REPO/scripts/train.py" --mode holdout \
      --images-dir "$SOURCE_IMAGES" --split-csv "$SOURCE_SPLIT" --out "$ref" \
      --init-checkpoint "$base/checkpoint.pt" $ARCH $REFINER $STAGE2 --seed "$seed" \
      --device "$DEVICE" $EXTRA $EXTRA_TRAIN

  common=(--checkpoint "$ref/checkpoint.pt" --images-dir "$SOURCE_IMAGES" --split-csv "$SOURCE_SPLIT"
          --subset test --device "$DEVICE")
  run $PY "$REPO/scripts/evaluate.py" "${common[@]}" --no-refine --out "$ref/eval_base" $EXTRA
  run $PY "$REPO/scripts/evaluate.py" "${common[@]}" --no-refine $POSTPROC --out "$ref/eval_base_containment" $EXTRA
  run $PY "$REPO/scripts/evaluate.py" "${common[@]}" $POSTPROC --out "$ref/eval_cief_net" $EXTRA
done

run $PY "$REPO/scripts/paired_tests.py" \
    --reference "cief_net=$OUT/cief_net_s*/eval_cief_net" \
    --compare "base=$OUT/cief_net_s*/eval_base" \
    --compare "base_containment=$OUT/cief_net_s*/eval_base_containment" \
    --out "$OUT/paired_tests_ablation.csv"
