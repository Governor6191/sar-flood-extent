"""Score the Harvey flood prediction against the Copernicus EMS observed extent.

Compares the model's flood mask for the 2017-08-30 Houston Sentinel-1 scene to
the Copernicus EMS EMSR229 Houston delineation (observed flood extent from radar,
2017-08-28/29). It reprojects the Copernicus polygons onto the prediction grid,
restricts scoring to the area Copernicus actually mapped, strips JRC permanent
water from both sides so we score flood against flood rather than penalizing the
model for finding the bayous and Galveston Bay, and reports water-class IoU, F1,
precision, and recall. It also writes a four-panel comparison figure.

Reports two numbers: a raw score over the mapped AOI, and a flood-only score with
permanent water removed. The flood-only number is the fair one.

Run from repo root, after unzipping the Copernicus vector package into
data/harvey/copernicus/:
    uv run python scripts/validate_harvey.py
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.warp import Resampling, reproject

REPO_ROOT = Path(__file__).resolve().parents[1]
PROJECT = "sar-flood-ee"

DEF_PRED = REPO_ROOT / "outputs" / "predictions" / "harvey_s1_houston_2017-08-30_floodmask.tif"
DEF_SCENE = REPO_ROOT / "data" / "harvey" / "harvey_s1_houston_2017-08-30.tif"
DEF_TRUTH = REPO_ROOT / "data" / "harvey" / "copernicus"
DEF_FIG = REPO_ROOT / "docs" / "figures" / "harvey_validation.png"
DEF_METRICS = REPO_ROOT / "outputs" / "harvey_validation_metrics.json"


# ---- Copernicus vector loading -------------------------------------------------

def load_truth(truth_dir: Path, dst_crs):
    """Return (flood_gdf, aoi_gdf_or_None, flood_layer_name), reprojected to dst_crs."""
    import geopandas as gpd

    shps = sorted(truth_dir.rglob("*.shp"))
    if not shps:
        sys.exit(f"No .shp found under {truth_dir}. Unzip the Copernicus vector package there.")

    print("Copernicus layers found:")
    layers = []
    for shp in shps:
        gdf = gpd.read_file(shp)
        layers.append((shp, gdf))
        geom = sorted(set(gdf.geom_type))
        print(f"  {shp.name:55s} {str(geom):28s} n={len(gdf)} crs={gdf.crs}")

    def is_poly(g):
        return any("Polygon" in t for t in set(g.geom_type))

    def norm(name):
        return name.lower().replace("_", "").replace(" ", "")

    # Flood extent: the observed-event / crisis-information polygon layer. Copernicus
    # EMS has used both "observedEvent" (older schema) and "crisis_information_poly"
    # (current schema) for this. Match on the underscore-stripped name so both work.
    flood = None
    flood_name = None
    for key in ("crisisinformation", "observedevent", "crisisinfo", "flood", "obsrvd"):
        for shp, gdf in layers:
            if key in norm(shp.name) and is_poly(gdf):
                flood, flood_name = gdf, shp.name
                print(f"  -> flood extent: {shp.name}")
                break
        if flood is not None:
            break
    if flood is None:
        # last resort: the largest polygon layer that is not the AOI
        polys = [(s, g) for s, g in layers if is_poly(g) and "areaofinterest" not in norm(s.name)]
        if polys:
            shp, flood = max(polys, key=lambda sg: len(sg[1]))
            flood_name = shp.name
            print(f"  -> flood extent (fallback, largest polygon layer): {shp.name}")
    if flood is None:
        sys.exit("Could not identify a flood-extent polygon layer. Paste the layer list above.")

    # Keep only flooded polygons if the layer carries a class field. EMSR229's
    # crisis_information layer tags each polygon (interpret "Flooded Area",
    # subtype "EM009 - Flood"); a mixed layer would otherwise inflate the truth.
    for field in ("interpret", "subtype", "obj_type", "notation"):
        if field in flood.columns:
            mask = flood[field].astype(str).str.contains("flood", case=False, na=False)
            if mask.any():
                kept, total = int(mask.sum()), len(flood)
                if kept < total:
                    print(f"  -> filtered flood layer on {field}: kept {kept}/{total} flooded polygons")
                flood = flood[mask]
                break

    aoi = None
    for shp, gdf in layers:
        if "areaofinterest" in norm(shp.name) and is_poly(gdf):
            aoi = gdf
            print(f"  -> mapped AOI:    {shp.name}")
            break
    if aoi is None:
        print("  -> no areaOfInterest layer; scoring domain falls back to flood-extent convex hull")

    flood = flood.to_crs(dst_crs)
    if aoi is not None:
        aoi = aoi.to_crs(dst_crs)
    return flood, aoi, flood_name


# ---- JRC permanent water via Earth Engine -------------------------------------

def fetch_permanent_water(bounds, dst_shape, dst_transform, dst_crs, occurrence_thresh):
    """Return a bool array on the prediction grid: True where JRC water occurrence >= thresh."""
    import ee

    try:
        ee.Initialize(project=PROJECT)
    except Exception as e:  # noqa: BLE001
        print(f"  [warn] Earth Engine init failed ({e}); skipping permanent-water mask")
        return np.zeros(dst_shape, dtype=bool)

    minx, miny, maxx, maxy = bounds
    region = ee.Geometry.Rectangle([minx, miny, maxx, maxy])
    occ = ee.Image("JRC/GSW1_4/GlobalSurfaceWater").select("occurrence").unmask(0)
    url = occ.getDownloadURL(
        {"scale": 30, "crs": "EPSG:4326", "region": region, "format": "GEO_TIFF", "bands": ["occurrence"]}
    )
    tmp = Path(tempfile.mkdtemp(prefix="jrc_")) / "occ.tif"
    urlretrieve(url, tmp)

    with rasterio.open(tmp) as src:
        dst = np.zeros(dst_shape, dtype=np.float32)
        reproject(
            source=rasterio.band(src, 1),
            destination=dst,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            resampling=Resampling.nearest,
        )
    return dst >= occurrence_thresh


# ---- metrics ------------------------------------------------------------------

def score(pred, gt, dom):
    p = pred & dom
    g = gt & dom
    tp = int((p & g).sum())
    fp = int((p & ~g).sum())
    fn = int((~p & g).sum())
    tn = int((~p & ~g).sum())
    iou = tp / (tp + fp + fn + 1e-9)
    prec = tp / (tp + fp + 1e-9)
    rec = tp / (tp + fn + 1e-9)
    f1 = 2 * prec * rec / (prec + rec + 1e-9)
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "iou": iou,
            "precision": prec, "recall": rec, "f1": f1}


# ---- figure -------------------------------------------------------------------

def make_figure(vv, pred, gt, dom, perm, out_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    H, W = pred.shape
    ds = max(1, max(H, W) // 1500)
    vv_d = vv[::ds, ::ds]
    pred_d = pred[::ds, ::ds]
    gt_d = gt[::ds, ::ds]
    dom_d = dom[::ds, ::ds]
    eval_d = dom_d & ~perm[::ds, ::ds]

    g = vv_d.astype(np.float32)
    g[~dom_d] = np.nan
    lo, hi = np.nanpercentile(g, 2), np.nanpercentile(g, 98)
    gn = np.clip((g - lo) / (hi - lo + 1e-9), 0, 1)
    base = np.dstack([gn, gn, gn])
    base[np.isnan(gn)] = 0.12

    def overlay(color, mask):
        img = base.copy()
        img[mask] = color
        return img

    pred_e = pred_d & eval_d
    gt_e = gt_d & eval_d
    agree = base.copy()
    agree[gt_e & ~pred_e] = [0.15, 0.45, 1.0]   # FN blue
    agree[pred_e & ~gt_e] = [1.0, 0.25, 0.15]   # FP red
    agree[pred_e & gt_e] = [0.20, 0.85, 0.30]   # TP green

    fig, ax = plt.subplots(2, 2, figsize=(13, 11))
    ax[0, 0].imshow(base); ax[0, 0].set_title("Sentinel-1 VV (dB), 2017-08-30")
    ax[0, 1].imshow(overlay([1.0, 0.25, 0.15], pred_d & dom_d)); ax[0, 1].set_title("Model prediction (red)")
    ax[1, 0].imshow(overlay([0.15, 0.45, 1.0], gt_d & dom_d)); ax[1, 0].set_title("Copernicus EMS observed flood (blue)")
    ax[1, 1].imshow(agree); ax[1, 1].set_title("Agreement, permanent water removed")
    ax[1, 1].legend(handles=[
        Patch(color=[0.20, 0.85, 0.30], label="TP (both)"),
        Patch(color=[1.0, 0.25, 0.15], label="FP (model only)"),
        Patch(color=[0.15, 0.45, 1.0], label="FN (Copernicus only)"),
    ], loc="lower right", fontsize=8)
    for a in ax.ravel():
        a.set_xticks([]); a.set_yticks([])
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    print(f"figure saved: {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pred", default=str(DEF_PRED))
    p.add_argument("--scene", default=str(DEF_SCENE))
    p.add_argument("--truth-dir", default=str(DEF_TRUTH))
    p.add_argument("--perm-occurrence", type=float, default=50.0,
                   help="JRC occurrence %% at/above which a pixel counts as permanent water")
    p.add_argument("--out-fig", default=str(DEF_FIG))
    p.add_argument("--metrics", default=str(DEF_METRICS))
    args = p.parse_args()

    with rasterio.open(args.pred) as src:
        pred = src.read(1).astype(bool)
        transform, crs = src.transform, src.crs
        bounds = tuple(src.bounds)
        H, W = pred.shape
    with rasterio.open(args.scene) as src:
        vv = src.read(1).astype(np.float32)

    dst_crs = crs.to_wkt()
    flood, aoi, flood_name = load_truth(Path(args.truth_dir), dst_crs)

    src_dates = []
    date_col = next((c for c in flood.columns if c.lower() in ("src_date", "src_dat", "srcdate")), None)
    if date_col is not None:
        src_dates = sorted({str(v)[:10] for v in flood[date_col].dropna()})

    gt = rasterize([(g, 1) for g in flood.geometry if g is not None],
                   out_shape=(H, W), transform=transform, fill=0, dtype="uint8").astype(bool)

    if aoi is not None:
        dom = rasterize([(g, 1) for g in aoi.geometry if g is not None],
                        out_shape=(H, W), transform=transform, fill=0, dtype="uint8").astype(bool)
    else:
        hull = flood.unary_union.convex_hull
        dom = rasterize([(hull, 1)], out_shape=(H, W), transform=transform, fill=0, dtype="uint8").astype(bool)

    print("\nfetching JRC permanent-water mask from Earth Engine ...")
    perm = fetch_permanent_water(bounds, (H, W), transform, dst_crs, args.perm_occurrence)

    raw = score(pred, gt, dom)
    floodonly = score(pred & ~perm, gt & ~perm, dom & ~perm)

    print(f"\nscoring domain: {int(dom.sum()):,} px mapped by Copernicus "
          f"({dom.mean()*100:.1f}% of the scene)")
    print(f"permanent water in domain: {int((dom & perm).sum()):,} px\n")

    def show(name, m):
        print(f"  {name:18s} IoU {m['iou']:.3f}  F1 {m['f1']:.3f}  "
              f"P {m['precision']:.3f}  R {m['recall']:.3f}  "
              f"(TP {m['tp']:,} FP {m['fp']:,} FN {m['fn']:,})")

    print("water-class scores vs Copernicus EMS:")
    show("raw (all water)", raw)
    show("flood-only", floodonly)

    make_figure(vv, pred, gt, dom, perm, Path(args.out_fig))

    out = {
        "scene": "harvey_s1_houston_2017-08-30",
        "ground_truth": "Copernicus EMS EMSR229 Houston delineation",
        "ground_truth_layer": flood_name,
        "ground_truth_src_dates": src_dates,
        "perm_occurrence_thresh": args.perm_occurrence,
        "domain_px": int(dom.sum()),
        "raw": raw,
        "flood_only": floodonly,
    }
    Path(args.metrics).parent.mkdir(parents=True, exist_ok=True)
    Path(args.metrics).write_text(json.dumps(out, indent=2))
    print(f"metrics saved: {args.metrics}")


if __name__ == "__main__":
    main()
