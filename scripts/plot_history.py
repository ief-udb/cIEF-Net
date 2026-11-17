#!/usr/bin/env python3
"""Plot the training history written by ``scripts/train.py`` (``history.json``).

Panel 1 shows the training loss of every epoch and the validation loss of the
evaluated epochs; both use the same formula, so a training loss that keeps
decreasing while the validation loss stalls indicates overfitting. Further
panels show the selected evaluation metrics, grouped by scale (bounded
metrics in [0, 1] versus counting error).

Example::

    python scripts/plot_history.py --history runs/base_s42/history.json \\
        --out runs/base_s42/training_curve.png
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

BOUNDED = {"aji_mean", "pq_mean", "dq_mean", "sq_mean", "boundary_iou_mean", "dice_mean",
           "count_corr", "map_50"}


def plot(history, out_path, metrics, title=""):
    epochs = [h["epoch"] for h in history]
    groups = [g for g in ([m for m in metrics if m in BOUNDED], [m for m in metrics if m not in BOUNDED]) if g]
    fig, axes = plt.subplots(1 + len(groups), 1, figsize=(10, 5.5 * (1 + len(groups))), squeeze=False)
    axes = axes[:, 0]

    ax = axes[0]
    ax.plot(epochs, [h["train_loss"] for h in history], color="#3b6ea5", label="training loss", lw=1.6)
    val = [(h["epoch"], h["val_loss"]) for h in history if h.get("val_loss") is not None]
    if val:
        ax.plot(*zip(*val), color="#c0392b", marker="o", ms=5, label="validation loss", lw=1.6)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(title or "Training and validation loss")
    ax.legend(fontsize=9)

    colours = ["#27ae60", "#8e44ad", "#e67e22", "#16a085", "#c0392b"]
    for ax, group in zip(axes[1:], groups):
        for i, m in enumerate(group):
            pts = [(h["epoch"], h["eval"][m]) for h in history if "eval" in h and m in h["eval"]]
            if pts:
                ax.plot(*zip(*pts), marker="o", ms=4, color=colours[i % len(colours)], label=m, lw=1.6)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Evaluation metric")
        ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--history", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--metrics", default="aji_mean,pq_mean,count_mae")
    ap.add_argument("--title", default="")
    args = ap.parse_args()
    path = Path(args.history)
    if not path.exists():
        sys.exit(f"--history does not exist: {path}")
    with open(path) as f:
        history = json.load(f)
    if not any(h.get("val_loss") is not None for h in history):
        print("note: no intermediate evaluation in this history (--eval-every 0); only the training "
              "loss is plotted")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    plot(history, out, [m.strip() for m in args.metrics.split(",") if m.strip()], args.title)
    print(f"Saved to {out.resolve()}")


if __name__ == "__main__":
    main()
