"""Compute per-channel SAR normalization statistics on the Sen1Floods11 train split.

Streams the full train split accumulating exact per-channel mean, std, min,
and max from running sums, and subsamples pixels for robust percentile
estimates. Writes results to src/normalization_stats.json for the training
pipeline to load, and a histogram figure to outputs/figures/.

Statistics are computed on the full train split (both hand and weak labels).
The input SAR distribution does not depend on label provenance, so these
stats are valid for any training subset (hand-only, weak-only, or curriculum).
Stats are never computed on val/test, which would leak information.

Run from repo root:
    uv run python scripts/compute_normalization_stats.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from data import Sen1FloodsDataset  # noqa: E402

CHANNELS = ["VV", "VH"]
PERCENTILES = [1, 2, 5, 25, 50, 75, 95, 98, 99]
SUBSAMPLE_PER_IMAGE = 2000  # pixels per image kept for percentile estimation
STATS_PATH = REPO_ROOT / "src" / "normalization_stats.json"
FIG_DIR = REPO_ROOT / "outputs" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    print("Loading Sen1Floods11 train split...")
    train = Sen1FloodsDataset(split="train")
    n = len(train)
    n_channels = len(CHANNELS)
    print(f"  {n} training samples, {n_channels} channels")

    # Streaming accumulators for exact mean/std
    total_sum = np.zeros(n_channels, dtype=np.float64)
    total_sumsq = np.zeros(n_channels, dtype=np.float64)
    total_count = np.zeros(n_channels, dtype=np.int64)
    ch_min = np.full(n_channels, np.inf, dtype=np.float64)
    ch_max = np.full(n_channels, -np.inf, dtype=np.float64)

    # Per-channel subsample buffers for percentile estimation
    subsamples: list[list[np.ndarray]] = [[] for _ in range(n_channels)]
    rng = np.random.default_rng(0)

    nan_count = np.zeros(n_channels, dtype=np.int64)
    inf_count = np.zeros(n_channels, dtype=np.int64)

    print("Streaming through train split (this takes a few minutes)...")
    for i in range(n):
        image = train[i]["image"].numpy()  # (2, H, W) float32
        for c in range(n_channels):
            band = image[c].ravel()

            nan_count[c] += int(np.isnan(band).sum())
            inf_count[c] += int(np.isinf(band).sum())
            band = band[np.isfinite(band)]
            if band.size == 0:
                continue

            total_sum[c] += band.sum(dtype=np.float64)
            total_sumsq[c] += np.square(band.astype(np.float64)).sum()
            total_count[c] += band.size
            ch_min[c] = min(ch_min[c], float(band.min()))
            ch_max[c] = max(ch_max[c], float(band.max()))

            k = min(SUBSAMPLE_PER_IMAGE, band.size)
            idx = rng.choice(band.size, size=k, replace=False)
            subsamples[c].append(band[idx])

        if (i + 1) % 500 == 0:
            print(f"  processed {i + 1}/{n}")

    mean = total_sum / total_count
    var = total_sumsq / total_count - mean**2
    std = np.sqrt(np.clip(var, 0.0, None))

    stats = {
        "dataset": "pdosquet/sen1floods11-preprocessed-dl",
        "split": "train",
        "label_source": "all",
        "n_samples": n,
        "channel_order": CHANNELS,
        "notes": (
            "Per-channel statistics on the full Sen1Floods11 train split. "
            "Use mean/std for z-score standardization: (x - mean) / std. "
            "Percentiles estimated from a per-image random subsample."
        ),
        "channels": {},
    }

    for c, name in enumerate(CHANNELS):
        sub = np.concatenate(subsamples[c])
        pcts = np.percentile(sub, PERCENTILES)
        stats["channels"][name] = {
            "mean": float(mean[c]),
            "std": float(std[c]),
            "min": float(ch_min[c]),
            "max": float(ch_max[c]),
            "n_pixels": int(total_count[c]),
            "n_nan": int(nan_count[c]),
            "n_inf": int(inf_count[c]),
            "percentiles": {str(p): float(v) for p, v in zip(PERCENTILES, pcts)},
        }

    STATS_PATH.write_text(json.dumps(stats, indent=2))
    print(f"\nStats written to {STATS_PATH}")

    for name in CHANNELS:
        ch = stats["channels"][name]
        print(f"\n=== {name} ===")
        print(f"  mean: {ch['mean']:.4f}   std: {ch['std']:.4f}")
        print(f"  min:  {ch['min']:.4f}   max: {ch['max']:.4f}")
        print(f"  pixels: {ch['n_pixels']:,}   NaN: {ch['n_nan']:,}   inf: {ch['n_inf']:,}")
        p = ch["percentiles"]
        print(f"  p2: {p['2']:.4f}   p50: {p['50']:.4f}   p98: {p['98']:.4f}")

    # Histogram figure (uses the subsample)
    fig, axes = plt.subplots(1, n_channels, figsize=(13, 5))
    for c, name in enumerate(CHANNELS):
        sub = np.concatenate(subsamples[c])
        axes[c].hist(sub, bins=200, color="steelblue", alpha=0.85)
        axes[c].axvline(mean[c], color="red", linestyle="-", label=f"mean = {mean[c]:.2f}")
        axes[c].axvline(mean[c] - std[c], color="orange", linestyle="--", label="+/- 1 std")
        axes[c].axvline(mean[c] + std[c], color="orange", linestyle="--")
        axes[c].set_title(f"{name} backscatter distribution (train)")
        axes[c].set_xlabel("value")
        axes[c].set_ylabel("pixel count (subsample)")
        axes[c].legend()
    fig.tight_layout()
    fig_path = FIG_DIR / "normalization_histograms.png"
    fig.savefig(fig_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"\nHistogram saved to {fig_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
