"""Break down the Harvey flood validation by land cover.

Stratifies the Copernicus EMS flood (permanent water removed) by ESA WorldCover
class and reports the model's recall per class. Recall is highest on open water
and bare ground, where floodwater darkens the radar return, and collapses on
built-up and vegetated land, where flooding keeps or raises backscatter (double
bounce, volume scattering) and a single-image open-water detector can't see it.
Writes a recall-by-land-cover figure to docs/figures/.

Run from repo root:
    uv run python scripts/stratify_harvey.py
"""
from __future__ import annotations

import tempfile
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.warp import Resampling, reproject
import geopandas as gpd
import ee

REPO = Path(__file__).resolve().parents[1]
PRED = REPO / "outputs" / "predictions" / "harvey_s1_houston_2017-08-30_floodmask.tif"
COP = REPO / "data" / "harvey" / "copernicus"
OUT_FIG = REPO / "docs" / "figures" / "harvey_landcover_stratification.png"
PROJECT = "sar-flood-ee"
PERM_OCC = 50.0

WC_NAMES = {10: "Tree cover", 20: "Shrubland", 30: "Grassland", 40: "Cropland",
            50: "Built-up", 60: "Bare/sparse", 70: "Snow/ice", 80: "Permanent water",
            90: "Herb. wetland", 95: "Mangroves", 100: "Moss/lichen"}
# Grouped by whether floodwater is physically visible to SAR open-water detection.
OPEN = {30, 40, 60, 80, 90, 100}      # smooth/dark when flooded, or already water
VEGETATED = {10, 20, 95}              # volume scattering keeps it bright
BUILTUP = {50}                        # double-bounce keeps it bright


def ee_grid_layer(ee_img, band, bounds, shape, transform, crs_wkt, scale=30):
    """Download an EE single-band image over bounds and reproject (nearest) to the grid."""
    minx, miny, maxx, maxy = bounds
    region = ee.Geometry.Rectangle([minx, miny, maxx, maxy])
    url = ee_img.getDownloadURL({"scale": scale, "crs": "EPSG:4326", "region": region,
                                 "format": "GEO_TIFF", "bands": [band]})
    tmp = Path(tempfile.mkdtemp(prefix="ee_")) / "layer.tif"
    urlretrieve(url, tmp)
    out = np.zeros(shape, dtype=np.float32)
    with rasterio.open(tmp) as src:
        reproject(rasterio.band(src, 1), out, src_transform=src.transform, src_crs=src.crs,
                  dst_transform=transform, dst_crs=crs_wkt, resampling=Resampling.nearest)
    return out


def main():
    with rasterio.open(PRED) as src:
        pred = src.read(1).astype(bool)
        transform, crs, bounds = src.transform, src.crs, tuple(src.bounds)
        H, W = pred.shape
    crs_wkt = crs.to_wkt()

    flood = gpd.read_file(next(COP.rglob("*crisis_information_poly.shp"))).to_crs(crs_wkt)
    aoi = gpd.read_file(next(COP.rglob("*area_of_interest.shp"))).to_crs(crs_wkt)
    gt = rasterize([(g, 1) for g in flood.geometry if g], (H, W), transform=transform,
                   fill=0, dtype="uint8").astype(bool)
    dom = rasterize([(g, 1) for g in aoi.geometry if g], (H, W), transform=transform,
                    fill=0, dtype="uint8").astype(bool)

    ee.Initialize(project=PROJECT)
    print("fetching JRC permanent water ...")
    occ = ee_grid_layer(ee.Image("JRC/GSW1_4/GlobalSurfaceWater").select("occurrence").unmask(0),
                        "occurrence", bounds, (H, W), transform, crs_wkt)
    perm = occ >= PERM_OCC
    print("fetching ESA WorldCover (2020) ...")
    wc = ee_grid_layer(ee.ImageCollection("ESA/WorldCover/v100").first().select("Map").unmask(0),
                       "Map", bounds, (H, W), transform, crs_wkt).astype(np.int16)

    # flood-only frame: inside mapped AOI, permanent water removed
    keep = dom & ~perm
    gt_f = gt & keep
    pred_f = pred & keep

    print(f"\nflood-only domain: {keep.sum():,} px   Copernicus flood (perm removed): {gt_f.sum():,} px"
          f"   model water: {pred_f.sum():,} px")
    print("\nper land-cover class, within flood-only frame:")
    print(f"  {'class':16s} {'flood px':>11s} {'%flood':>7s} {'recall':>7s} {'precision':>9s}")

    present = sorted(set(np.unique(wc[keep])) & set(WC_NAMES))
    total_flood = int(gt_f.sum())
    rows = []
    for c in present:
        m = wc == c
        gtc = gt_f & m
        predc = pred_f & m
        nflood = int(gtc.sum())
        tp = int((pred_f & gtc).sum())
        recall = tp / (nflood + 1e-9)
        prec = int((predc & gt).sum()) / (int(predc.sum()) + 1e-9)
        share = 100 * nflood / (total_flood + 1e-9)
        rows.append((c, nflood, share, recall, prec))
        print(f"  {WC_NAMES[c]:16s} {nflood:11,} {share:6.1f}% {recall:7.3f} {prec:9.3f}")

    def group_recall(group):
        gmask = np.isin(wc, list(group))
        gtc = gt_f & gmask
        tp = int((pred_f & gtc).sum())
        n = int(gtc.sum())
        return n, tp / (n + 1e-9), 100 * n / (total_flood + 1e-9)

    print("\ngrouped by SAR flood visibility:")
    for label, grp in [("OPEN (detectable)", OPEN), ("VEGETATED (bright)", VEGETATED),
                       ("BUILT-UP (bright)", BUILTUP)]:
        n, rec, share = group_recall(grp)
        print(f"  {label:22s} flood {n:11,} ({share:5.1f}%)   recall {rec:.3f}")

    # IoU restricted to open-water-compatible terrain vs the full flood-only IoU
    def iou(p, g, d):
        p, g = p & d, g & d
        inter = int((p & g).sum()); union = int((p | g).sum())
        return inter / (union + 1e-9)

    open_mask = np.isin(wc, list(OPEN))
    iou_all = iou(pred, gt, keep)
    iou_open = iou(pred, gt, keep & open_mask)
    print(f"\nflood-only IoU, all terrain:              {iou_all:.3f}")
    print(f"flood-only IoU, open-water terrain only:  {iou_open:.3f}")
    fn_hard = group_recall(VEGETATED)[0] + group_recall(BUILTUP)[0]
    print(f"\nshare of Copernicus flood on bright (vegetated+built-up) terrain: "
          f"{100*fn_hard/(total_flood+1e-9):.1f}%")

    # figure: flood area by class + recall by class
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [WC_NAMES[c] for c, *_ in rows]
    shares = [r[2] for r in rows]
    recalls = [r[3] for r in rows]
    colors = ["#2c7fb8" if c in OPEN else "#d95f0e" for c, *_ in rows]
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    ax[0].barh(labels, shares, color=colors)
    ax[0].set_xlabel("% of Copernicus flood (permanent water removed)")
    ax[0].set_title("Where the flood is, by land cover")
    ax[0].invert_yaxis()
    ax[1].barh(labels, recalls, color=colors)
    ax[1].set_xlabel("model recall")
    ax[1].set_title("Can the model see it? (recall by land cover)")
    ax[1].set_xlim(0, 1)
    ax[1].invert_yaxis()
    fig.suptitle("Hurricane Harvey: model flood recall by land cover "
                 "(blue = open/bare, orange = vegetated/built-up)", fontsize=12)
    fig.tight_layout()
    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_FIG, dpi=130, bbox_inches="tight")
    print(f"\nfigure saved: {OUT_FIG}")


if __name__ == "__main__":
    main()
