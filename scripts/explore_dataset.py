"""Sen1Floods11 dataset exploration and sanity check.

Loads all three splits, prints schema + region + label-source distributions,
verifies a single sample's tensor shapes and value ranges, and saves both
per-sample PNGs and a combined grid PNG of the train hand-labeled subset to
outputs/figures/.

Run from repo root:
    uv run python scripts/explore_dataset.py
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from data import Sen1FloodsDataset  # noqa: E402

OUT_DIR = REPO_ROOT / "outputs" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def summarize_split(ds: Sen1FloodsDataset, name: str) -> None:
    print(f"\n=== {name} ===")
    print(f"  total samples: {len(ds)}")
    regions = Counter(s["region"] for s in ds.ds)
    sources = Counter(s["label_source"] for s in ds.ds)
    print(f"  regions ({len(regions)}):")
    for region, count in regions.most_common():
        print(f"    {region:20s} {count:5d}")
    print(f"  label_source:")
    for source, count in sources.most_common():
        print(f"    {source:20s} {count:5d}")


def _plot_sample_row(axes, vv, vh, mask, region, patch_id, label_source):
    """Render one sample as three panels: VV, VH, VV+label-overlay.

    Uses per-image 2nd-98th-percentile clipping so each panel auto-scales
    to its own dynamic range regardless of upstream normalization.
    """
    vv_lo, vv_hi = np.percentile(vv, [2, 98])
    vh_lo, vh_hi = np.percentile(vh, [2, 98])

    axes[0].imshow(vv, cmap="gray", vmin=vv_lo, vmax=vv_hi)
    axes[0].set_title(f"VV  |  {region}  |  {label_source}")
    axes[0].axis("off")

    axes[1].imshow(vh, cmap="gray", vmin=vh_lo, vmax=vh_hi)
    axes[1].set_title(f"VH  |  patch_id: {patch_id[:28]}")
    axes[1].axis("off")

    rgba = np.zeros((*mask.shape, 4))
    rgba[mask == -1] = [0.5, 0.5, 0.5, 0.35]
    rgba[mask == 1] = [1.0, 0.1, 0.1, 0.55]

    axes[2].imshow(vv, cmap="gray", vmin=vv_lo, vmax=vv_hi)
    axes[2].imshow(rgba)
    axes[2].set_title("VV + label overlay  (red=water, grey=unlabeled)")
    axes[2].axis("off")


def visualize(ds: Sen1FloodsDataset, n: int = 4, seed: int = 42) -> tuple[Path, list[Path]]:
    """Generate per-sample PNGs plus one combined grid PNG.

    Returns ``(grid_path, list_of_individual_paths)``.
    """
    rng = np.random.default_rng(seed)
    idxs = rng.choice(len(ds), size=n, replace=False)

    samples = []
    for idx in idxs:
        sample = ds[int(idx)]
        samples.append({
            "vv": sample["image"][0].numpy(),
            "vh": sample["image"][1].numpy(),
            "mask": sample["mask"].numpy(),
            "region": sample["region"],
            "patch_id": sample["patch_id"],
            "label_source": sample["label_source"],
        })

    # Per-sample PNGs (one figure per sample, high resolution)
    individual_paths: list[Path] = []
    for s in samples:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        _plot_sample_row(
            axes, s["vv"], s["vh"], s["mask"],
            s["region"], s["patch_id"], s["label_source"],
        )
        fig.suptitle(f"Sen1Floods11 — {s['patch_id']}", fontsize=12)
        fig.tight_layout()

        # Sanitize patch_id for the filesystem
        safe_id = s["patch_id"].replace("/", "_").replace("\\", "_")
        out = OUT_DIR / f"sample_{s['region']}_{safe_id}.png"
        fig.savefig(out, dpi=120, bbox_inches="tight")
        plt.close(fig)
        individual_paths.append(out)

    # Combined grid PNG (all samples on one canvas, for contact-sheet view)
    fig, axes = plt.subplots(n, 3, figsize=(15, 5 * n))
    if n == 1:
        axes = axes[np.newaxis, :]
    for row, s in enumerate(samples):
        _plot_sample_row(
            axes[row], s["vv"], s["vh"], s["mask"],
            s["region"], s["patch_id"], s["label_source"],
        )
    fig.suptitle("Sen1Floods11 — train sanity check (hand-labeled)", fontsize=14, y=1.0)
    fig.tight_layout()

    grid_path = OUT_DIR / "data_exploration_train_grid.png"
    fig.savefig(grid_path, dpi=120, bbox_inches="tight")
    plt.close(fig)

    return grid_path, individual_paths


def main() -> None:
    print("Loading Sen1Floods11 splits (this re-uses the local 80GB cache)...")
    train = Sen1FloodsDataset(split="train")
    val = Sen1FloodsDataset(split="validation")
    test = Sen1FloodsDataset(split="test")

    summarize_split(train, "TRAIN")
    summarize_split(val, "VAL")
    summarize_split(test, "TEST")

    print("\n=== Sample 0 (train) — tensor sanity check ===")
    s = train[0]
    print(f"  image.shape: {tuple(s['image'].shape)}  dtype: {s['image'].dtype}")
    print(f"  mask.shape:  {tuple(s['mask'].shape)}  dtype: {s['mask'].dtype}")
    print(f"  VV range:    [{s['image'][0].min():.2f}, {s['image'][0].max():.2f}]")
    print(f"  VH range:    [{s['image'][1].min():.2f}, {s['image'][1].max():.2f}]")
    unique_labels = sorted(set(s["mask"].numpy().flatten().tolist()))
    print(f"  mask unique: {unique_labels}")
    print(f"  region:      {s['region']}")
    print(f"  source:      {s['label_source']}")

    print("\nGenerating visualizations (hand-labeled samples only)...")
    train_hand = Sen1FloodsDataset(split="train", label_source="hand")
    print(f"  hand-labeled train samples: {len(train_hand)}")
    grid_path, individual_paths = visualize(train_hand, n=4)
    print(f"  grid saved to:    {grid_path}")
    for p in individual_paths:
        print(f"  per-sample saved: {p}")

    print("\nSanity check complete.")


if __name__ == "__main__":
    main()

