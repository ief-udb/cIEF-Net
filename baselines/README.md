# Comparison methods

This repository contains only cIEF-Net. The four comparison methods were
trained and evaluated with their public implementations, using the same
partitions, working scale, tiling and metrics as cIEF-Net, and were
fine-tuned from their *A. alternata* checkpoints in the few-shot experiments.

| Method | Implementation | Source domain (*A. alternata*) | Few-shot (target taxa) |
|---|---|---|---|
| Mask R-CNN (He et al., 2017) | [pytorch/vision](https://github.com/pytorch/vision) (`maskrcnn_resnet50_fpn`) | ResNet-50-FPN, COCO weights; tiles 512 px; 20 epochs; learning rate 0.005 | 20 epochs; learning rate 0.001 |
| Cellpose-SAM (Pachitariu et al., 2025) | [MouseLand/cellpose](https://github.com/MouseLand/cellpose) | `cpsam` weights; tiles 256 px; 30 epochs; learning rate 1e-5; diameter estimation and rescaling disabled | 20 epochs; learning rate 2e-6; minimum of one mask per training tile |
| StarDist (Schmidt et al., 2018) | [stardist/stardist](https://github.com/stardist/stardist) | U-Net with three levels and 32 base filters; 32 rays; 50 epochs; learning rate 1e-4 | 20 epochs; learning rate 2e-5 |
| YOLO11m-seg (Jocher and Qiu, 2024) | [ultralytics/ultralytics](https://github.com/ultralytics/ultralytics) | COCO weights; tiles 512 px; 100 epochs; batch 16; AdamW, learning rate 2e-3, 3 warm-up epochs; mosaic except in the last 10 epochs; inference confidence 0.05, internal NMS 0.7 | 20 epochs |

To compare another method with cIEF-Net, write its per-image metrics to a
`per_image_metrics.csv` with the columns of `cief_net.metrics.PER_IMAGE_COLUMNS`
(computed with `cief_net.metrics.InstanceEvaluator`) and run
`scripts/paired_tests.py`.
