"""Diagnose NaN-input pixels and the dataset's standardization scheme.

Answers two questions raised by the normalization statistics:

1. Do NaN SAR pixels correspond to -1 (unlabeled) mask pixels? This decides
   whether filling NaN with 0 and relying on ignore_index=-1 in the loss is
   enough, or whether we need a separate validity mask.
2. Is the dataset standardized per-image or globally? Per-image means each
   sample has mean ~0 and std ~1 on its own; global means individual samples
   deviate. This matters for inference preprocessing on raw Sentinel-1 scenes.

Run from repo root:
    uv run python scripts/diagnose_nan_and_scaling.py
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from data import Sen1FloodsDataset  # noqa: E402

N_SAMPLE = 300
rng = np.random.default_rng(0)


def main() -> None:
    train = Sen1FloodsDataset(split="train")
    idxs = rng.choice(len(train), size=min(N_SAMPLE, len(train)), replace=False)

    nan_label_counts = Counter()
    total_nan = 0
    images_with_nan = 0

    per_image_mean = []
    per_image_std = []

    for idx in idxs:
        sample = train[int(idx)]
        image = sample["image"].numpy()   # (2, H, W)
        mask = sample["mask"].numpy()      # (H, W)
        vv = image[0]

        nan_mask = ~np.isfinite(vv)        # nodata pixels (identical for VV/VH)
        if nan_mask.any():
            images_with_nan += 1
            labels_at_nan = mask[nan_mask]
            for lbl in (-1, 0, 1):
                nan_label_counts[lbl] += int((labels_at_nan == lbl).sum())
            total_nan += int(nan_mask.sum())

        finite = vv[np.isfinite(vv)]
        if finite.size:
            per_image_mean.append(float(finite.mean()))
            per_image_std.append(float(finite.std()))

    print(f"Sampled {len(idxs)} train images\n")

    print("=== Q1: label values at NaN-input pixels ===")
    print(f"  images containing NaN pixels: {images_with_nan}/{len(idxs)}")
    if total_nan == 0:
        print("  No NaN input pixels in this sample.")
    else:
        for lbl in (-1, 0, 1):
            pct = 100 * nan_label_counts[lbl] / total_nan
            name = {-1: "unlabeled", 0: "dry", 1: "water"}[lbl]
            print(f"  label {lbl:2d} ({name:9s}): {nan_label_counts[lbl]:>12,}  ({pct:5.1f}%)")
        print(f"  total NaN pixels: {total_nan:,}")

    print("\n=== Q2: per-image standardization check (VV) ===")
    pm = np.array(per_image_mean)
    ps = np.array(per_image_std)
    print(f"  per-image mean: avg={pm.mean():.3f}  std={pm.std():.3f}  range=[{pm.min():.3f}, {pm.max():.3f}]")
    print(f"  per-image std:  avg={ps.mean():.3f}  std={ps.std():.3f}  range=[{ps.min():.3f}, {ps.max():.3f}]")
    if abs(pm.mean()) < 0.15 and abs(ps.mean() - 1.0) < 0.15 and pm.std() < 0.15 and ps.std() < 0.15:
        print("  -> PER-IMAGE standardized (each image ~mean 0, ~std 1 on its own).")
    else:
        print("  -> GLOBALLY standardized (individual images deviate from 0/1).")


if __name__ == "__main__":
    main()
