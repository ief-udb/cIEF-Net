#!/usr/bin/env bash
# Selection of the refinement configuration by four-fold cross-validation on
# the 116 training images of A. alternata (the test set is never used).
#
# Protocol (fixed before running):
#   for every validation fold v = 0..3
#     1. base network trained on the other three folds with the stage-1
#        configuration (seed 42, last epoch);
#     2. evaluation of base + containment on fold v (reference without refiner);
#     3. every refiner variant (affine pose, harmonic K = 2, 5, 24) trained for
#        15 epochs at 3e-4 on that frozen base and evaluated with containment
#        on fold v.
#   Every training image is evaluated exactly once per variant. Rule: highest
#   mean out-of-fold PQ (scripts/select_refiner_cv.py).
#
# For every fold a manifest is written in which fold v is renamed 'test', the
# other three folds keep their label and the real test set becomes 'excluded',
# so that the holdout mode trains on three folds and evaluates on v.
#
# Usage: bash experiments/02_refiner_selection_cv.sh
#        FOLDS="0 1" bash experiments/02_refiner_selection_cv.sh     (split the work)
#        VARIANTS="pose k2 k5 k12 k24" bash ...                       (add K = 12)
# Interrupted runs resume: finished steps are skipped.
set -uo pipefail
source "$(dirname "$0")/config.sh"
OUT="${OUT:-$RUNS/cv_refiner_selection}"
FOLDS="${FOLDS:-0 1 2 3}"
VARIANTS="${VARIANTS:-pose k2 k5 k24}"
SEED=42

variant_args() {
  case "$1" in
    pose) echo "--use-refiner --refiner-type pose" ;;
    k*)   echo "--use-refiner --refiner-type harmonic --refiner-kmax ${1#k}" ;;
    *)    echo "unknown variant: $1" >&2; return 1 ;;
  esac
}

mkdir -p "$OUT/splits"
$PY - "$SOURCE_SPLIT" "$OUT/splits" <<'PYCODE'
import sys
from pathlib import Path
import pandas as pd
df = pd.read_csv(sys.argv[1]).rename(columns={"imagen": "image"})
df["fold"] = df["fold"].astype(str)
for v in "0123":
    fold = df["fold"].where(df["fold"].isin(list("0123")), "excluded").where(df["fold"] != v, "test")
    pd.DataFrame({"image": df["image"], "fold": fold}).to_csv(Path(sys.argv[2]) / f"split_val{v}.csv", index=False)
    print(f"validation fold {v}: {int(fold.isin(list('0123')).sum())} train, "
          f"{int((fold == 'test').sum())} validation, {int((fold == 'excluded').sum())} excluded")
PYCODE

for v in $FOLDS; do
  split="$OUT/splits/split_val$v.csv"
  base="$OUT/fold$v/base"
  echo "=== validation fold $v ==="
  [[ -f "$base/checkpoint.pt" ]] || run $PY "$REPO/scripts/train.py" --mode holdout \
      --images-dir "$SOURCE_IMAGES" --split-csv "$split" --out "$base" \
      $ARCH $STAGE1 --seed $SEED --device "$DEVICE" $EXTRA $EXTRA_TRAIN || { echo "[error] base of fold $v"; continue; }
  [[ -f "$base/eval_containment_val/summary.json" ]] || run $PY "$REPO/scripts/evaluate.py" \
      --checkpoint "$base/checkpoint.pt" --images-dir "$SOURCE_IMAGES" --split-csv "$split" \
      --subset test $POSTPROC --device "$DEVICE" --out "$base/eval_containment_val" $EXTRA

  for var in $VARIANTS; do
    args="$(variant_args "$var")" || continue
    r="$OUT/fold$v/ref_$var"
    [[ -f "$r/eval_refiner_val/summary.json" ]] && { echo "[done] $r"; continue; }
    [[ -f "$r/checkpoint.pt" ]] || run $PY "$REPO/scripts/train.py" --mode holdout \
        --images-dir "$SOURCE_IMAGES" --split-csv "$split" --out "$r" \
        --init-checkpoint "$base/checkpoint.pt" $ARCH $args $STAGE2 --seed $SEED \
        --device "$DEVICE" $EXTRA $EXTRA_TRAIN || { echo "[error] refiner $var of fold $v"; continue; }
    run $PY "$REPO/scripts/evaluate.py" --checkpoint "$r/checkpoint.pt" --images-dir "$SOURCE_IMAGES" \
        --split-csv "$split" --subset test $POSTPROC --device "$DEVICE" --out "$r/eval_refiner_val" $EXTRA
  done
done

run $PY "$REPO/scripts/select_refiner_cv.py" --root "$OUT" --variants no_refiner $VARIANTS
