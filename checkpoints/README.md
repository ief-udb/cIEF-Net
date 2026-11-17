# Pretrained models

Weight files are not versioned. Download them from **[link to be added]** and
place each model in its own folder:

```
checkpoints/
├── cief_net_alternaria/          final model (base + harmonic refiner K = 24), seed 42
│   ├── checkpoint.pt
│   ├── model_config.json
│   ├── diameter_calibration.json
│   └── summary.json
└── base_alternaria/              base network (stage 1), seed 42
    └── ...
```

* `checkpoint.pt` is a PyTorch `state_dict` (weights of the last epoch) of
  the full model, including the frozen encoder. Building the model still
  requires access to the DINOv3 encoder, from the Hugging Face Hub or from a
  local folder given with `--backbone` (or as `"backbone"` in
  `model_config.json`).
* `model_config.json` stores the architecture; the scripts read it
  automatically.
* `diameter_calibration.json` stores the median conidium diameter of the
  training set, used to calibrate the scale of few-shot runs started from
  this checkpoint.
* `summary.json` stores the test metrics and the working side (`ref_side`).

Use with any script through `--checkpoint checkpoints/cief_net_alternaria/checkpoint.pt`.
The DINOv3 weights are distributed under the DINOv3 License by Meta; the
fine-tuned components are released under CC BY-NC 4.0.
