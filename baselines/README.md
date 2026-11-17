# Comparison methods

The paper compares cIEF-Net with four methods, each representing a different
strategy for forming instances. All were trained and evaluated with the same
partitions, tiling and metrics as cIEF-Net, and fine-tuned from their
*A. alternata* checkpoints in the few-shot experiments.

| Method | Source domain (*A. alternata*) | Few-shot (target taxa) |
|---|---|---|
| Mask R-CNN | ResNet-50-FPN (torchvision, COCO weights); tiles 512 px; 20 epochs; learning rate 0.005 | 20 epochs; learning rate 0.001 |
| Cellpose-SAM | `cpsam` weights; tiles 256 px; 30 epochs; learning rate 1e-5; diameter estimation and rescaling disabled | 20 epochs; learning rate 2e-6; minimum of one mask per training tile |
| StarDist | U-Net with three levels and 32 base filters; 32 rays; 50 epochs; learning rate 1e-4 | 20 epochs; learning rate 2e-5 |
| YOLO11m-seg | COCO weights; tiles 512 px; 100 epochs; batch 16; AdamW, learning rate 2e-3, 3 warm-up epochs; mosaic except in the last 10 epochs; inference confidence 0.05, internal NMS 0.7 | 20 epochs |

The training and evaluation code of these methods will be added to this
folder. Each method writes a `per_image_metrics.csv` with the columns of
`cief_net.metrics.PER_IMAGE_COLUMNS`, so that it can be compared with
cIEF-Net by `scripts/paired_tests.py`.
