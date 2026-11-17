# Reproducibility notes

## Experiments and paper sections

| Script | Paper | Main outputs |
|---|---|---|
| `experiments/01_alternaria.sh` | Source-domain benchmark (cIEF-Net row), ablation of containment suppression and contour refinement, paired image-level tests | `runs/alternaria/cief_net_s*/eval_*/summary.json`, `runs/alternaria/paired_tests_ablation.csv` |
| `experiments/02_refiner_selection_cv.sh` | Selection of the refinement configuration (four-fold cross-validation) | `runs/cv_refiner_selection/refiner_selection_cv*.csv` |
| `experiments/03_fewshot.sh` | Few-shot adaptation (cIEF-Net, base model and frozen-refiner ablation) | `runs/fewshot/fewshot_summary.csv`, `runs/fewshot/<species>/paired_tests_<N>shot.csv` |
| `experiments/04_diagnostics.sh` | Spectral analysis of the contour error, stratified results, containment sensitivity | `runs/alternaria/cief_net_s42/diagnosis_*/error_summary.json` |

All hyperparameters are in `experiments/config.sh`:

| Setting | Value |
|---|---|
| Working scale (source) | longest side 1536 px; tiles 512 px, overlap 0.2 |
| Stage 1 | AdamW, weight decay 0.05; learning rates 1e-5 (DoRA) and 1e-4 (decoder, head); cosine schedule with period 60 epochs, 30 epochs; batch 8; geometric and photometric augmentation |
| Stage 2 | refiner only, 15 epochs, learning rate 3e-4 (cosine over the 15 epochs); refiner inputs from the predicted contour, or (p = 0.3) from a perturbed annotation |
| Few-shot | all trainable parts, 20 epochs, learning rates 2e-6 and 2e-5; training seed = support seed |
| Inference | centre probability > 0.5; polygon NMS IoU 0.5; tile merging IoU 0.5; one detection per connected component of the probability map; containment suppression tau = 0.8, rho = 0.7 with merge protection |
| Seeds | source domain 42, 7, 123 (42 principal); few-shot supports 42, 7, 123, 2024 |

## Evaluation conventions

* Every result is computed with the weights of the **last epoch**.
  `--save-best-checkpoint` keeps the best intermediate epoch for inspection
  only.
* Predictions with a score of at least 0.5 are evaluated as individual
  binary instance masks at the working resolution; overlaps between predicted
  instances are not resolved.
* PQ, DQ and SQ use one-to-one matching at IoU > 0.5. AJI uses greedy
  one-to-one matching in annotation order. Boundary IoU and Dice are computed
  on the foreground of each image (union of instance masks); the boundary
  band is obtained with two erosions by the four-connected 3 x 3 element.
* Metrics are computed per image and averaged over images. mAP uses at most
  1000 detections per image.
* Few-shot runs are evaluated at their support-calibrated working side, stored
  as `ref_side_calibrated` in `summary.json` and reused automatically by
  `evaluate.py`, `diagnose_errors.py`, `visualize_predictions.py` and
  `predict.py`.

## One detection per connected component

Inside every tile, at most one polygon is kept per connected component of the
thresholded centre-probability map (the one with the highest score). This rule
prevents elongated conidia from being split into two detections at their
narrowest part. It is part of the inference used for every cIEF-Net result in
the paper and is enabled by default (`--no-component-filter` disables it).

## Determinism

GPU training is not bit-exact across hardware and library versions (cuDNN
kernels are non-deterministic), so retraining reproduces the reported results
up to run-to-run variation, which is why several seeds are reported. Note also
that intermediate evaluations (`--eval-every`) draw from the PyTorch random
number generator when their data loaders are created, so changing
`--eval-every` changes the random stream of later epochs; the values in
`experiments/config.sh` are those used for the paper.

## Correspondence with the original implementation

This repository is a cleaned, English-language version of the code used for
the paper. With identical inputs and seeds, it reproduces the original
implementation exactly (identical weights after training and identical
per-image metrics), which was verified for stage 1, stage 2 (harmonic and
pose refiners), few-shot fine-tuning with scale calibration, evaluation with
and without refinement and containment, error diagnosis, cross-validation
selection and paired tests.

Checkpoints of the original implementation load directly: module names are
converted on the fly (`cief_net/checkpoints.py`). Because such checkpoints
have no `model_config.json`, pass the architecture flags when using them
(e.g. `--use-refiner --refiner-type harmonic --refiner-kmax 24`), or convert
them once with `scripts/convert_legacy_checkpoint.py`.

## Exploratory components

The following options were evaluated during model development, are reported
in the supplementary development history, and are disabled by default:
density head with density-adaptive NMS (`--use-density-head
--adaptive-nms`), sub-grid offset head (`--use-offset-head`), L1 radial loss
(`--radial-loss l1`), unweighted radial loss (`--no-boundary-weighting`),
moving-centre refiner (`--refiner-type harmonic_moving`) and polar-frame
recentring (`--refiner-recenter`).
