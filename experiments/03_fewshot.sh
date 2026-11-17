#!/usr/bin/env bash
# Few-shot adaptation to the four target domains (paper: cross-taxon
# adaptation). For every species, support size N in {5, 10} and support seed,
# three cIEF-Net configurations are fine-tuned for 20 epochs from the seed-42
# source-domain checkpoints and evaluated on the fixed test set:
#
#   cief_net                 all parts adapted, refiner on, containment suppression
#   base                     base network only (no refiner, no containment)
#   cief_net_frozen_refiner  as cief_net, but the source-trained refiner is kept fixed
#
# Scale is calibrated from the support annotations (scripts/train.py --mode
# fewshot). The training seed equals the support seed. Paired image-level
# tests of cIEF-Net against both ablations follow (per-image metrics averaged
# over the support seeds). Baseline methods are compared the same way with
# scripts/paired_tests.py once their per-image metrics are available.
#
# Usage: bash experiments/03_fewshot.sh                 (all species)
#        SPECIES="hortaea_werneckii" bash experiments/03_fewshot.sh
set -euo pipefail
source "$(dirname "$0")/config.sh"
SRC="$RUNS/alternaria"
BASE_CKPT="${BASE_CKPT:-$SRC/base_s42/checkpoint.pt}"
CIEF_CKPT="${CIEF_CKPT:-$SRC/cief_net_s42/checkpoint.pt}"
OUT="$RUNS/fewshot"

declare -A CONFIG=(
  [cief_net]="--init-checkpoint $CIEF_CKPT $REFINER $POSTPROC"
  [base]="--init-checkpoint $BASE_CKPT"
  [cief_net_frozen_refiner]="--init-checkpoint $CIEF_CKPT $REFINER $POSTPROC --freeze-refiner"
)

for sp in $SPECIES; do
  images="$DATA/$sp/images"
  for n in $SHOTS; do
    for seed in $FEWSHOT_SEEDS; do
      manifest="$DATA/splits/fewshot/$sp/${n}shot_s${seed}.csv"
      [[ -f "$manifest" ]] || { echo "[skip] missing manifest $manifest"; continue; }
      for cfg in cief_net base cief_net_frozen_refiner; do
        out="$OUT/$sp/$cfg/${n}shot/s$seed"
        [[ -f "$out/summary.json" ]] && { echo "[done] $out"; continue; }
        run $PY "$REPO/scripts/train.py" --mode fewshot --images-dir "$images" \
            --fewshot-csv "$manifest" --out "$out" $ARCH ${CONFIG[$cfg]} $FEWSHOT \
            --seed "$seed" --device "$DEVICE" $EXTRA $EXTRA_TRAIN
      done
    done
    run $PY "$REPO/scripts/paired_tests.py" \
        --reference "cief_net=$OUT/$sp/cief_net/${n}shot/s*" \
        --compare "base=$OUT/$sp/base/${n}shot/s*" \
        --compare "frozen_refiner=$OUT/$sp/cief_net_frozen_refiner/${n}shot/s*" \
        --out "$OUT/$sp/paired_tests_${n}shot.csv"
  done
done

$PY "$REPO/scripts/aggregate_runs.py" --root "$OUT" --out "$OUT/fewshot_summary.csv"
