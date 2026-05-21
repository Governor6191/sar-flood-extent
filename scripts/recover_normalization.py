"""Recover the Sen1Floods11 preprocessed-dl standardization constants.

The model trained on the `pdosquet/sen1floods11-preprocessed-dl` variant, which
standardized each SAR channel to ~mean 0 / std 1. To run the model on a new raw
Sentinel-1 scene (Hurricane Harvey 2017), we must apply the SAME transform. Its
constants are not publicly documented, so we recover them empirically: fetch a
few ORIGINAL Sen1Floods11 dB tiles (public GCS bucket, read over HTTPS), match
them to our local standardized tiles by patch_id, and fit the per-channel affine
    standardized = (dB - mean) / std.

Writes src/sen1floods11_norm_constants.json and prints diagnostics (R^2,
residuals, raw dB stats) so we can confirm the transform is a clean global
affine. R^2 near 1.0 with a tiny residual means the recovered mean/std are the
constants to apply to Harvey. R^2 well below 1.0 means clipping or grid
misalignment, which we then diagnose before trusting anything.

Run from repo root:
    uv run python scripts/recover_normalization.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# GDAL config for anonymous remote (/vsicurl) reads of the public COG bucket.
os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
os.environ.setdefault("CPL_VSIL_CURL_ALLOWED_EXTENSIONS", ".tif")

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from data import Sen1FloodsDataset  # noqa: E402  (datasets before torch)

import numpy as np  # noqa: E402
import rasterio  # noqa: E402

OUT = REPO_ROOT / "src" / "sen1floods11_norm_constants.json"
N_TILES = 8
CHANNELS = ["VV", "VH"]

# Candidate remote paths for the original S1 (dB) hand-labeled tiles.
URL_TEMPLATES = [
    "/vsicurl/https://storage.googleapis.com/sen1floods11/v1.1/data/flood_events/HandLabeled/S1Hand/{pid}_S1Hand.tif",
    "/vsicurl/https://storage.googleapis.com/sen1floods11/data/flood_events/HandLabeled/S1Hand/{pid}_S1Hand.tif",
]


def fetch_db_tile(pid: str):
    """Return (2, H, W) dB array for a patch_id, or (None, error-string)."""
    last = "no template tried"
    for tmpl in URL_TEMPLATES:
        url = tmpl.format(pid=pid)
        try:
            with rasterio.open(url) as src:
                return src.read().astype(np.float64), url
        except Exception as e:  # noqa: BLE001
            last = f"{type(e).__name__}: {e}"
    return None, last


def main() -> None:
    test = Sen1FloodsDataset(split="test")  # all hand-labeled -> exist as S1Hand
    print(f"test split: {len(test)} samples; fetching up to {N_TILES} original dB tiles\n")

    db_vals: list[list[np.ndarray]] = [[], []]
    std_vals: list[list[np.ndarray]] = [[], []]
    used = 0
    seen: dict[str, int] = {}  # region -> count, for region-diverse sampling

    for i in range(len(test)):
        if used >= N_TILES:
            break
        sample = test.ds[i]
        region = sample["region"]
        if seen.get(region, 0) >= 1:  # at most one tile per region, for diversity
            continue
        pid = sample["patch_id"]
        db, info = fetch_db_tile(pid)
        if db is None:
            print(f"  [skip] {pid}: {info[:90]}")
            continue
        std = np.asarray(sample["s1"], dtype=np.float64)  # (2, H, W) standardized
        if db.shape != std.shape:
            print(f"  [skip] {pid}: shape mismatch {db.shape} vs {std.shape}")
            continue
        for c in range(2):
            d = db[c].ravel()
            s = std[c].ravel()
            # finite in both, and physically plausible sigma0 dB (drop fills)
            m = np.isfinite(d) & np.isfinite(s) & (d > -50.0) & (d < 20.0)
            db_vals[c].append(d[m])
            std_vals[c].append(s[m])
        seen[region] = seen.get(region, 0) + 1
        used += 1
        print(f"  [ok]   {region:12s} {pid}")

    if used == 0:
        sys.exit("\nNo original tiles fetched. Check the bucket path or network and retry.")

    print(f"\nFitted on {used} tiles.")
    constants = {
        "source": "empirical recovery from original Sen1Floods11 S1Hand dB tiles",
        "transform": "standardized = (dB - mean) / std, per channel",
        "n_tiles": used,
        "channels": {},
    }
    for c, name in enumerate(CHANNELS):
        d = np.concatenate(db_vals[c])
        s = np.concatenate(std_vals[c])
        A = np.vstack([d, np.ones_like(d)]).T
        (a, b), *_ = np.linalg.lstsq(A, s, rcond=None)
        pred = a * d + b
        ss_res = float(np.sum((s - pred) ** 2))
        ss_tot = float(np.sum((s - s.mean()) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        resid = float(np.max(np.abs(s - pred)))
        std_c = 1.0 / a
        mean_c = -b / a

        print(f"\n=== {name} ===")
        print(f"  fit: standardized = {a:.6f} * dB + {b:.6f}")
        print(f"  R^2 = {r2:.6f}   max|residual| = {resid:.4f}")
        print(f"  recovered constants: mean = {mean_c:.4f} dB,  std = {std_c:.4f} dB")
        print(f"  raw dB global (cross-check): mean = {d.mean():.4f}  std = {d.std():.4f}")

        constants["channels"][name] = {
            "mean": mean_c,
            "std": std_c,
            "fit_slope": float(a),
            "fit_intercept": float(b),
            "r2": r2,
            "max_abs_residual": resid,
            "raw_db_mean": float(d.mean()),
            "raw_db_std": float(d.std()),
        }

    OUT.write_text(json.dumps(constants, indent=2))
    print(f"\nSaved {OUT}")
    print("\nRead the R^2 values:")
    print("  ~1.000 with tiny residual  -> clean global affine; mean/std are the Harvey constants.")
    print("  < 0.99 or large residual   -> clipping or grid misalignment; do not trust, we diagnose next.")


if __name__ == "__main__":
    main()
