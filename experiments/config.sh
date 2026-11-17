# Shared configuration of the reproduction scripts. Every variable can be
# overridden from the environment, e.g.  SEEDS="42" bash experiments/01_alternaria.sh
#
# Expected data layout (see data/README.md):
#   data/alternaria_alternata/images/*.{png,jpg,tif}  + LabelMe *.json
#   data/<target species>/images/...
#   data/splits/alternaria_alternata.csv               image,fold
#   data/splits/fewshot/<species>/<N>shot_s<seed>.csv  image,species,subset

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PY:-python}"
DATA="${DATA:-$REPO/data}"
RUNS="${RUNS:-$REPO/runs}"
DEVICE="${DEVICE:-cuda}"
BACKBONE="${BACKBONE:-facebook/dinov3-vitl16-pretrain-lvd1689m}"

SOURCE_IMAGES="${SOURCE_IMAGES:-$DATA/alternaria_alternata/images}"
SOURCE_SPLIT="${SOURCE_SPLIT:-$DATA/splits/alternaria_alternata.csv}"

# Training seeds of the source-domain benchmark (42 is the principal seed;
# few-shot runs start from the seed-42 checkpoints).
SEEDS="${SEEDS:-42 7 123}"
FEWSHOT_SEEDS="${FEWSHOT_SEEDS:-42 7 123 2024}"
SHOTS="${SHOTS:-5 10}"
SPECIES="${SPECIES:-lasiodiplodia colletotrichum_siamense neopestalotiopsis hortaea_werneckii}"

# Architecture of cIEF-Net (defaults of scripts/train.py, listed for clarity)
ARCH="${ARCH:---backbone $BACKBONE --patch-size 16 --grid-stride 4 --n-rays 48 --dora-r 16}"
REFINER="${REFINER:---use-refiner --refiner-type harmonic --refiner-kmax 24}"
POSTPROC="${POSTPROC:---containment 0.8 --containment-area-ratio 0.7 --protect-merges}"

# Stage 1 (base network) and stage 2 (refiner), source domain
STAGE1="${STAGE1:---epochs 30 --lr-t-max 60 --lr-backbone 1e-5 --lr-new 1e-4 --batch-size 8 --radial-loss log --eval-every 30}"
STAGE2="${STAGE2:---refiner-only --epochs 15 --lr-new 3e-4 --batch-size 8 --eval-every 5}"

# Few-shot fine-tuning (all trainable parts, 20 epochs)
FEWSHOT="${FEWSHOT:---epochs 20 --lr-backbone 2e-6 --lr-new 2e-5 --batch-size 8}"

# Optional overrides, e.g. for a quick smoke test:
#   EXTRA="--tile-size 256 --ref-side 512"   (training and evaluation)
#   EXTRA_TRAIN="--num-workers 0"            (training only)
EXTRA="${EXTRA:-}"
EXTRA_TRAIN="${EXTRA_TRAIN:-}"

run() { echo "+ $*"; "$@"; }
