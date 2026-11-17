#!/usr/bin/env bash
# Error decomposition on the A. alternata test set (paper: spectral analysis of
# the contour error and stratified results; supplementary sensitivity of the
# containment thresholds). Uses the seed-42 stage-2 checkpoint.
set -euo pipefail
source "$(dirname "$0")/config.sh"
CKPT="${CKPT:-$RUNS/alternaria/cief_net_s42/checkpoint.pt}"
OUT="$(dirname "$CKPT")"
common=(--checkpoint "$CKPT" --images-dir "$SOURCE_IMAGES" --split-csv "$SOURCE_SPLIT" --subset test
        --device "$DEVICE")

# base model, with the sensitivity sweep of the suppression thresholds
run $PY "$REPO/scripts/diagnose_errors.py" "${common[@]}" --no-refine --sweep-containment \
    --out "$OUT/diagnosis_base" $EXTRA
# base + containment, and the full model
run $PY "$REPO/scripts/diagnose_errors.py" "${common[@]}" --no-refine $POSTPROC \
    --out "$OUT/diagnosis_base_containment" $EXTRA
run $PY "$REPO/scripts/diagnose_errors.py" "${common[@]}" $POSTPROC --out "$OUT/diagnosis_cief_net" $EXTRA
