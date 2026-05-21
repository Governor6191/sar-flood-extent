"""Run the trained SAR flood model on an input and save a flood mask.

Two input modes:
  - A Sen1Floods11 sample by split + index (default; uses the local dataset).
  - A NumPy file of shape (2, H, W) holding standardized VV/VH SAR, via --input.

Outputs a predicted binary flood mask (.npy) and an overlay PNG to
outputs/predictions/.

Note on raw scenes: this expects input already standardized the way the
Sen1Floods11 preprocessed-dl variant is (per-channel global z-score, mean 0
std 1). Running on a raw Sentinel-1 GRD scene needs the matching preprocessing
applied first. That step (recovering and applying the dataset's standardization
constants) is the FEMA inference stage and is not implemented yet.

Run from repo root:
    uv run python scripts/predict.py --checkpoint checkpoints/unet_resnet34_all_best.pt --split test --index 0
    uv run python scripts/predict.py --checkpoint checkpoints/unet_resnet34_all_best.pt --input scene.npy
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# data.py imports datasets (pyarrow) before torch; keep that order.
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from data import Sen1FloodsDataset  # noqa: E402
from model import build_model  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

OUT_DIR = REPO_ROOT / "outputs" / "predictions"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_model(checkpoint_path, device):
    # weights_only=False: our own trusted checkpoint with metadata.
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = build_model(in_channels=2, classes=2, encoder_weights=None).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt


def predict(model, image, device):
    """image: (2, H, W) float32 standardized SAR. Returns (H, W) int mask."""
    x = torch.from_numpy(np.ascontiguousarray(image, dtype=np.float32))
    x = x.unsqueeze(0).to(device)
    with torch.no_grad():
        with torch.autocast(device_type=device, enabled=(device == "cuda")):
            logits = model(x)
    return logits.argmax(dim=1)[0].cpu().numpy()


def save_outputs(image, mask, name, gt=None):
    np.save(OUT_DIR / f"{name}_mask.npy", mask)

    vv = image[0]
    vlo, vhi = np.percentile(vv, [2, 98])
    ncols = 3 if gt is not None else 2
    fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols, 5))

    axes[0].imshow(vv, cmap="gray", vmin=vlo, vmax=vhi)
    axes[0].set_title("VV input")
    axes[0].axis("off")

    pred_rgba = np.zeros((*mask.shape, 4))
    pred_rgba[mask == 1] = [1.0, 0.2, 0.1, 0.6]
    axes[1].imshow(vv, cmap="gray", vmin=vlo, vmax=vhi)
    axes[1].imshow(pred_rgba)
    axes[1].set_title("prediction (red = water)")
    axes[1].axis("off")

    if gt is not None:
        gt_rgba = np.zeros((*gt.shape, 4))
        gt_rgba[gt == -1] = [0.5, 0.5, 0.5, 0.35]
        gt_rgba[gt == 1] = [0.1, 0.4, 1.0, 0.6]
        axes[2].imshow(vv, cmap="gray", vmin=vlo, vmax=vhi)
        axes[2].imshow(gt_rgba)
        axes[2].set_title("ground truth (blue = water)")
        axes[2].axis("off")

    fig.tight_layout()
    out_png = OUT_DIR / f"{name}_overlay.png"
    fig.savefig(out_png, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out_png


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/unet_resnet34_hand_best.pt")
    parser.add_argument("--input", default=None, help="path to a (2, H, W) .npy of standardized VV/VH")
    parser.add_argument("--split", default="test", choices=["train", "validation", "test"])
    parser.add_argument("--index", type=int, default=0)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    model, ckpt = load_model(REPO_ROOT / args.checkpoint, device)
    print(f"Loaded {args.checkpoint}: epoch {ckpt.get('epoch')}, val IoU {ckpt.get('val_iou', float('nan')):.4f}")

    gt = None
    if args.input is not None:
        image = np.load(args.input)
        if image.shape[0] != 2:
            raise ValueError(f"expected input shape (2, H, W), got {image.shape}")
        name = Path(args.input).stem
    else:
        ds = Sen1FloodsDataset(split=args.split)
        sample = ds[args.index]
        image = sample["image"].numpy()
        gt = sample["mask"].numpy()
        name = f"{args.split}_{args.index}_{sample['region']}"

    mask = predict(model, image, device)
    water_frac = float((mask == 1).mean())
    print(f"predicted water fraction: {water_frac:.3f}")
    out_png = save_outputs(image, mask, name, gt=gt)
    print(f"mask saved:    {OUT_DIR / (name + '_mask.npy')}")
    print(f"overlay saved: {out_png}")


if __name__ == "__main__":
    main()
