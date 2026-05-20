"""Evaluate a trained checkpoint on the Sen1Floods11 test split.

Loads a checkpoint, runs inference on the 105-sample all-hand test split, and
reports overall and per-region water-class IoU/F1. Also saves a qualitative
figure comparing VV input, ground truth, and prediction on a few test scenes.

Run from repo root:
    uv run python scripts/evaluate.py
    uv run python scripts/evaluate.py --checkpoint checkpoints/unet_resnet34_hand_best.pt
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

# data.py imports datasets (pyarrow) before torch; keep that order by importing
# the local modules before torch/matplotlib. See data.py for why.
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from data import Sen1FloodsDataset  # noqa: E402
from model import build_model  # noqa: E402
from metrics import confusion_counts, iou_score, f1_score  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

FIG_DIR = REPO_ROOT / "outputs" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_PATH = REPO_ROOT / "outputs" / "test_results.json"


def evaluate(model, loader, device):
    """Run inference. Return overall [tp, fp, fn] and per-region counts."""
    model.eval()
    overall = [0, 0, 0]
    per_region = defaultdict(lambda: [0, 0, 0])

    with torch.no_grad():
        for batch in loader:
            image = batch["image"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)
            regions = batch["region"]

            with torch.autocast(device_type=device, enabled=(device == "cuda")):
                logits = model(image)
            preds = logits.argmax(dim=1)

            for i in range(preds.size(0)):
                tp, fp, fn, _ = confusion_counts(preds[i:i + 1], mask[i:i + 1])
                overall[0] += tp
                overall[1] += fp
                overall[2] += fn
                r = regions[i]
                per_region[r][0] += tp
                per_region[r][1] += fp
                per_region[r][2] += fn

    return overall, per_region


def visualize_predictions(model, ds, device, n=6, seed=0):
    rng = np.random.default_rng(seed)
    idxs = rng.choice(len(ds), size=min(n, len(ds)), replace=False)

    fig, axes = plt.subplots(n, 3, figsize=(13, 4 * n))
    if n == 1:
        axes = axes[np.newaxis, :]

    model.eval()
    for row, idx in enumerate(idxs):
        sample = ds[int(idx)]
        image = sample["image"].unsqueeze(0).to(device)
        with torch.no_grad():
            with torch.autocast(device_type=device, enabled=(device == "cuda")):
                pred = model(image).argmax(dim=1)[0].cpu().numpy()

        vv = sample["image"][0].numpy()
        gt = sample["mask"].numpy()
        region = sample["region"]
        vlo, vhi = np.percentile(vv, [2, 98])

        axes[row, 0].imshow(vv, cmap="gray", vmin=vlo, vmax=vhi)
        axes[row, 0].set_title(f"VV  |  {region}")
        axes[row, 0].axis("off")

        gt_rgba = np.zeros((*gt.shape, 4))
        gt_rgba[gt == -1] = [0.5, 0.5, 0.5, 0.35]
        gt_rgba[gt == 1] = [0.1, 0.4, 1.0, 0.6]
        axes[row, 1].imshow(vv, cmap="gray", vmin=vlo, vmax=vhi)
        axes[row, 1].imshow(gt_rgba)
        axes[row, 1].set_title("ground truth (blue = water)")
        axes[row, 1].axis("off")

        pred_rgba = np.zeros((*pred.shape, 4))
        pred_rgba[pred == 1] = [1.0, 0.2, 0.1, 0.6]
        axes[row, 2].imshow(vv, cmap="gray", vmin=vlo, vmax=vhi)
        axes[row, 2].imshow(pred_rgba)
        axes[row, 2].set_title("prediction (red = water)")
        axes[row, 2].axis("off")

    fig.suptitle("Test-set predictions", fontsize=14, y=1.0)
    fig.tight_layout()
    out = FIG_DIR / "test_predictions.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Prediction figure saved to {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/unet_resnet34_hand_best.pt")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--n-viz", type=int, default=6)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    ckpt_path = REPO_ROOT / args.checkpoint
    # weights_only=False: this is our own trusted checkpoint with metadata.
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    print(f"Loaded {ckpt_path.name}: epoch {ckpt.get('epoch')}, val IoU {ckpt.get('val_iou', float('nan')):.4f}")

    model = build_model(in_channels=2, classes=2, encoder_weights=None).to(device)
    model.load_state_dict(ckpt["model_state"])

    test_ds = Sen1FloodsDataset(split="test")  # all 105 hand-labeled
    print(f"test samples: {len(test_ds)}")
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=0, pin_memory=(device == "cuda"))

    overall, per_region = evaluate(model, test_loader, device)
    tp, fp, fn = overall
    overall_iou = iou_score(tp, fp, fn)
    overall_f1 = f1_score(tp, fp, fn)

    print("\n=== Overall test (water class) ===")
    print(f"  IoU: {overall_iou:.4f}   F1: {overall_f1:.4f}")
    print(f"  TP {tp:,}  FP {fp:,}  FN {fn:,}")

    print("\n=== Per-region test IoU ===")
    region_results = {}
    for region in sorted(per_region):
        rtp, rfp, rfn = per_region[region]
        riou = iou_score(rtp, rfp, rfn)
        rf1 = f1_score(rtp, rfp, rfn)
        region_results[region] = {"iou": riou, "f1": rf1, "tp": rtp, "fp": rfp, "fn": rfn}
        print(f"  {region:12s}  IoU {riou:.4f}  F1 {rf1:.4f}")

    results = {
        "checkpoint": args.checkpoint,
        "split": "test",
        "overall": {"iou": overall_iou, "f1": overall_f1, "tp": tp, "fp": fp, "fn": fn},
        "per_region": region_results,
    }
    RESULTS_PATH.write_text(json.dumps(results, indent=2))
    print(f"\nResults written to {RESULTS_PATH}")

    visualize_predictions(model, test_ds, device, n=args.n_viz)


if __name__ == "__main__":
    main()
