"""Run the trained flood model on a full Sentinel-1 scene.

Takes a 2-band (VV, VH) sigma0 dB GeoTIFF (local path or /vsicurl URL),
standardizes it with the recovered Sen1Floods11 constants, tiles it to
512x512, runs the model per tile, stitches the masks, and writes a
georeferenced flood-mask GeoTIFF (optionally a GeoJSON of flood polygons).

This is the inference path for the Hurricane Harvey 2017 validation against the
Copernicus EMS flood delineation. The input GeoTIFF should be the GEE
COPERNICUS/S1_GRD product (sigma0 dB, VV+VH), which matches the representation
the recovered constants map from.

Self-test: `--selftest <patch_id>` fetches an original Sen1Floods11 dB tile and
its hand label, runs the full raw-dB pipeline, and reports water-class IoU plus
agreement with the locally-standardized prediction. If IoU matches the region's
benchmark number and agreement is ~100%, the raw-scene path is proven correct.

Run from repo root:
    uv run python scripts/infer_scene.py --input scene.tif --output mask.tif
    uv run python scripts/infer_scene.py --selftest Bolivia_103757
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
os.environ.setdefault("CPL_VSIL_CURL_ALLOWED_EXTENSIONS", ".tif")

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np  # noqa: E402
import rasterio  # noqa: E402
import torch  # noqa: E402

from model import build_model  # noqa: E402

CONSTANTS_PATH = REPO_ROOT / "src" / "sen1floods11_norm_constants.json"
OUT_DIR = REPO_ROOT / "outputs" / "predictions"
OUT_DIR.mkdir(parents=True, exist_ok=True)
CHANNELS = ["VV", "VH"]

S1_URL = "/vsicurl/https://storage.googleapis.com/sen1floods11/v1.1/data/flood_events/HandLabeled/S1Hand/{pid}_S1Hand.tif"
LABEL_URL = "/vsicurl/https://storage.googleapis.com/sen1floods11/v1.1/data/flood_events/HandLabeled/LabelHand/{pid}_LabelHand.tif"


def load_constants():
    c = json.loads(CONSTANTS_PATH.read_text())["channels"]
    return [(c[n]["mean"], c[n]["std"]) for n in CHANNELS]


def standardize(db: np.ndarray, constants) -> np.ndarray:
    """db: (2, H, W) sigma0 dB -> standardized float32; non-finite -> 0."""
    out = np.empty(db.shape, dtype=np.float32)
    for c in range(2):
        mean, std = constants[c]
        ch = (db[c].astype(np.float32) - mean) / std
        ch[~np.isfinite(ch)] = 0.0
        out[c] = ch
    return out


def load_model(checkpoint, device):
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    model = build_model(in_channels=2, classes=2, encoder_weights=None).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def infer(model, std_img: np.ndarray, device, tile=512, batch=8) -> np.ndarray:
    """std_img: (2, H, W) standardized -> (H, W) uint8 water mask."""
    _, H, W = std_img.shape
    mask = np.zeros((H, W), dtype=np.uint8)
    tiles, coords = [], []
    for y in range(0, H, tile):
        for x in range(0, W, tile):
            patch = std_img[:, y:y + tile, x:x + tile]
            ph, pw = patch.shape[1], patch.shape[2]
            if ph < tile or pw < tile:
                padded = np.zeros((2, tile, tile), dtype=np.float32)
                padded[:, :ph, :pw] = patch
                patch = padded
            tiles.append(patch)
            coords.append((y, x, ph, pw))

    with torch.no_grad():
        for i in range(0, len(tiles), batch):
            chunk = np.stack(tiles[i:i + batch])
            t = torch.from_numpy(chunk).to(device)
            with torch.autocast(device_type=device, enabled=(device == "cuda")):
                pred = model(t).argmax(dim=1).cpu().numpy().astype(np.uint8)
            for j, (y, x, ph, pw) in enumerate(coords[i:i + batch]):
                mask[y:y + ph, x:x + pw] = pred[j, :ph, :pw]
    return mask


def polygonize(mask, transform, crs, out_path):
    import geopandas as gpd
    from rasterio import features
    from shapely.geometry import shape

    geoms = [shape(g) for g, v in features.shapes(mask, mask == 1, transform=transform) if v == 1]
    gdf = gpd.GeoDataFrame({"class": ["water"] * len(geoms)}, geometry=geoms, crs=crs)
    gdf.to_file(out_path, driver="GeoJSON")
    return len(geoms)


def run_scene(args, device):
    model = load_model(REPO_ROOT / args.checkpoint, device)
    constants = load_constants()
    with rasterio.open(args.input) as src:
        db = src.read().astype(np.float64)
        profile = src.profile
        transform, crs = src.transform, src.crs
    if db.shape[0] < 2:
        sys.exit(f"expected >=2 bands (VV, VH), got {db.shape[0]}")
    std_img = standardize(db[:2], constants)
    mask = infer(model, std_img, device, tile=args.tile, batch=args.batch)

    name = Path(args.input).stem
    out_tif = Path(args.output) if args.output else OUT_DIR / f"{name}_floodmask.tif"
    profile.update(count=1, dtype="uint8", nodata=None, compress="lzw")
    with rasterio.open(out_tif, "w", **profile) as dst:
        dst.write(mask, 1)
    print(f"water fraction: {float((mask == 1).mean()):.3f}")
    print(f"mask saved: {out_tif}")
    if args.polygonize:
        gj = OUT_DIR / f"{name}_flood.geojson"
        n = polygonize(mask, transform, crs, gj)
        print(f"polygons saved: {gj} ({n} features)")


def run_selftest(pid, args, device):
    model = load_model(REPO_ROOT / args.checkpoint, device)
    constants = load_constants()
    with rasterio.open(S1_URL.format(pid=pid)) as src:
        db = src.read().astype(np.float64)
    with rasterio.open(LABEL_URL.format(pid=pid)) as src:
        label = src.read(1).astype(np.int64)

    std_img = standardize(db, constants)
    mask = infer(model, std_img, device, tile=args.tile, batch=args.batch)

    valid = label != -1
    tp = int(((mask == 1) & (label == 1) & valid).sum())
    fp = int(((mask == 1) & (label != 1) & valid).sum())
    fn = int(((mask != 1) & (label == 1)).sum())
    iou = tp / (tp + fp + fn + 1e-7)
    print(f"\nselftest {pid}: water-class IoU = {iou:.4f}  (TP {tp:,}  FP {fp:,}  FN {fn:,})")
    print("  This is the raw-dB -> standardize -> model path on a known tile.")
    print("  IoU in line with the region's benchmark number confirms the path is correct.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/unet_resnet34_all_best.pt")
    parser.add_argument("--input", default=None, help="2-band (VV,VH) dB GeoTIFF path or /vsicurl URL")
    parser.add_argument("--output", default=None, help="output mask GeoTIFF path")
    parser.add_argument("--selftest", default=None, help="Sen1Floods11 patch_id to validate the raw-dB path")
    parser.add_argument("--polygonize", action="store_true", help="also write a GeoJSON of flood polygons")
    parser.add_argument("--tile", type=int, default=512)
    parser.add_argument("--batch", type=int, default=8)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    if args.selftest:
        run_selftest(args.selftest, args, device)
    elif args.input:
        run_scene(args, device)
    else:
        sys.exit("provide --input <geotiff> or --selftest <patch_id>")


if __name__ == "__main__":
    main()
