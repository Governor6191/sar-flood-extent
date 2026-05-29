"""Pull a Sentinel-1 scene over Houston during Hurricane Harvey (2017) from GEE.

Queries COPERNICUS/S1_GRD (sigma0 dB, IW mode, VV+VH) over a Houston AOI for the
peak-flood window in late August 2017, lists what's available, picks the
date+orbit pass that best covers the AOI, mosaics that pass, and downloads a
2-band (VV, VH) dB GeoTIFF. The output drops straight into scripts/infer_scene.py.

The GEE S1_GRD bands are already terrain-corrected sigma0 in decibels, which is
the same representation the recovered Sen1Floods11 constants map from, so the
file needs no extra preprocessing before inference.

Download is tiled locally by default (the full file lands in data/harvey/). For a
very large AOI use --drive to fall back to a Google Drive export task.

Run from repo root:
    uv run python scripts/pull_harvey_scene.py
    uv run python scripts/pull_harvey_scene.py --start 2017-08-29 --end 2017-08-31
    uv run python scripts/pull_harvey_scene.py --bbox -95.75 29.60 -95.10 30.10
"""
from __future__ import annotations

import argparse
import sys
import tempfile
import urllib.request
from pathlib import Path

import ee
import rasterio
from rasterio.merge import merge as rio_merge

PROJECT = "sar-flood-ee"
REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "data" / "harvey"

# West-southwest Houston, where the Addicks/Barker releases and Buffalo Bayou
# drove the worst Harvey flooding. Tight enough to download fast, big enough to
# score a meaningful IoU against the Copernicus EMS flood extent.
DEFAULT_BBOX = (-95.75, 29.60, -95.10, 30.10)
DEFAULT_START = "2017-08-25"
DEFAULT_END = "2017-09-02"

# Keep each getDownloadURL request well under the request-size cap. A 2048x2048
# two-band float32 tile is ~33 MB; the cap is ~48 MB.
MAX_TILE_PX = 2048
SCALE_M = 10  # native Sentinel-1 GRD resolution, matches the training data


def init_ee() -> None:
    try:
        ee.Initialize(project=PROJECT)
    except Exception as e:  # noqa: BLE001
        sys.exit(
            f"ee.Initialize failed: {e}\n"
            "Run `uv run earthengine authenticate --force` and pick the account "
            f"that owns the {PROJECT} project."
        )


def build_collection(aoi, start, end):
    return (
        ee.ImageCollection("COPERNICUS/S1_GRD")
        .filterBounds(aoi)
        .filterDate(start, end)
        .filter(ee.Filter.eq("instrumentMode", "IW"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
        .select(["VV", "VH"])
    )


def inventory(col, aoi):
    """Return per-scene rows (date, pass, orbit, AOI coverage) without pulling geometry."""
    aoi_area = aoi.area(30)

    def to_feat(img):
        cov = img.geometry().intersection(aoi, 30).area(30).divide(aoi_area)
        return ee.Feature(
            None,
            {
                "date": ee.Date(img.get("system:time_start")).format("YYYY-MM-dd"),
                "pass": img.get("orbitProperties_pass"),
                "relorbit": img.get("relativeOrbitNumber_start"),
                "cov": cov,
            },
        )

    feats = ee.FeatureCollection(col.map(to_feat)).getInfo()["features"]
    return [f["properties"] for f in feats]


def pick_best(rows, target="2017-08-30"):
    """Group by (date, pass), sum coverage, pick max coverage; tiebreak nearest target."""
    from datetime import date as _date

    def to_d(s):
        y, m, d = (int(x) for x in s.split("-"))
        return _date(y, m, d)

    groups: dict[tuple[str, str], float] = {}
    for r in rows:
        key = (r["date"], r["pass"])
        groups[key] = groups.get(key, 0.0) + float(r["cov"])

    tgt = to_d(target)
    best = sorted(
        groups.items(),
        key=lambda kv: (-min(kv[1], 1.0), abs((to_d(kv[0][0]) - tgt).days)),
    )
    return best[0][0], min(best[0][1], 1.0)  # (date, pass), coverage


def build_image(col, date, pass_, aoi):
    day = ee.Date(date)
    sel = col.filterDate(day, day.advance(1, "day")).filter(
        ee.Filter.eq("orbitProperties_pass", pass_)
    )
    return sel.mosaic().select(["VV", "VH"]).clip(aoi).toFloat()


def _tile_edges(lo, hi, step):
    edges = []
    x = lo
    while x < hi:
        edges.append((x, min(x + step, hi)))
        x += step
    return edges


def download_local(image, bbox, out_path, scale=SCALE_M):
    """Tile the AOI, fetch each tile via getDownloadURL, mosaic to one GeoTIFF."""
    minx, miny, maxx, maxy = bbox
    # Degrees per MAX_TILE_PX block. Use the smaller cos(lat) factor so lon tiles
    # stay within the pixel budget.
    import math

    deg_per_px_lat = scale / 111320.0
    deg_per_px_lon = scale / (111320.0 * math.cos(math.radians((miny + maxy) / 2)))
    step_lat = MAX_TILE_PX * deg_per_px_lat
    step_lon = MAX_TILE_PX * deg_per_px_lon

    xedges = _tile_edges(minx, maxx, step_lon)
    yedges = _tile_edges(miny, maxy, step_lat)
    n_tiles = len(xedges) * len(yedges)
    print(f"downloading {n_tiles} tile(s) at {scale} m ...")

    tmpdir = Path(tempfile.mkdtemp(prefix="harvey_tiles_"))
    tile_paths = []
    k = 0
    for (y0, y1) in yedges:
        for (x0, x1) in xedges:
            k += 1
            region = ee.Geometry.Rectangle([x0, y0, x1, y1])
            url = image.getDownloadURL(
                {
                    "scale": scale,
                    "crs": "EPSG:4326",
                    "region": region,
                    "format": "GEO_TIFF",
                    "bands": ["VV", "VH"],
                }
            )
            tp = tmpdir / f"tile_{k:03d}.tif"
            urllib.request.urlretrieve(url, tp)
            tile_paths.append(tp)
            print(f"  tile {k}/{n_tiles} ok")

    srcs = [rasterio.open(p) for p in tile_paths]
    mosaic, transform = rio_merge(srcs)
    profile = srcs[0].profile
    profile.update(
        height=mosaic.shape[1],
        width=mosaic.shape[2],
        transform=transform,
        count=2,
        dtype="float32",
        compress="lzw",
    )
    for s in srcs:
        s.close()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(mosaic.astype("float32"))
        dst.set_band_description(1, "VV")
        dst.set_band_description(2, "VH")
    print(f"\nsaved: {out_path}  ({mosaic.shape[2]} x {mosaic.shape[1]} px)")


def export_drive(image, aoi, name, scale=SCALE_M):
    task = ee.batch.Export.image.toDrive(
        image=image,
        description=name,
        folder="ee_harvey",
        fileNamePrefix=name,
        region=aoi,
        scale=scale,
        crs="EPSG:4326",
        maxPixels=int(1e10),
    )
    task.start()
    print(
        f"started Drive export task '{name}' (folder ee_harvey). "
        "Monitor at https://code.earthengine.google.com/tasks and download the "
        f"GeoTIFF into {OUT_DIR} when it finishes."
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bbox", type=float, nargs=4, default=list(DEFAULT_BBOX),
                   metavar=("MINLON", "MINLAT", "MAXLON", "MAXLAT"))
    p.add_argument("--start", default=DEFAULT_START)
    p.add_argument("--end", default=DEFAULT_END)
    p.add_argument("--date", default=None, help="force a specific scene date (YYYY-MM-DD)")
    p.add_argument("--pass", dest="orbit_pass", default=None,
                   choices=["ASCENDING", "DESCENDING"], help="force orbit pass")
    p.add_argument("--scale", type=int, default=SCALE_M)
    p.add_argument("--drive", action="store_true", help="export to Google Drive instead of local download")
    p.add_argument("--name", default="harvey_s1_houston")
    args = p.parse_args()

    init_ee()
    aoi = ee.Geometry.Rectangle(args.bbox)
    col = build_collection(aoi, args.start, args.end)

    n = col.size().getInfo()
    if n == 0:
        sys.exit("no S1 IW VV+VH scenes in this window/AOI; widen --start/--end or --bbox")
    print(f"{n} scene(s) in window {args.start}..{args.end}\n")

    rows = inventory(col, aoi)
    rows.sort(key=lambda r: (r["date"], str(r["pass"])))
    print(f"{'date':12s} {'pass':11s} {'orbit':>6s} {'AOI cover':>10s}")
    for r in rows:
        print(f"{r['date']:12s} {str(r['pass']):11s} {int(r['relorbit']):6d} {float(r['cov'])*100:9.1f}%")

    if args.date and args.orbit_pass:
        date, opass = args.date, args.orbit_pass
        cov = next((float(r["cov"]) for r in rows if r["date"] == date and r["pass"] == opass), 0.0)
    else:
        (date, opass), cov = pick_best(rows)
    print(f"\nselected: {date}  {opass}  (AOI coverage {cov*100:.1f}%)")
    if cov < 0.98:
        print("  note: coverage < 100%. Edges outside the swath will read as 0 in the mask.")

    image = build_image(col, date, opass, aoi)

    if args.drive:
        export_drive(image, aoi, args.name, scale=args.scale)
        return

    out_path = OUT_DIR / f"{args.name}_{date}.tif"
    download_local(image, tuple(args.bbox), out_path, scale=args.scale)
    print("\nnext:")
    print(f"  uv run python scripts/infer_scene.py --input {out_path} \\")
    print(f"      --output outputs/predictions/{args.name}_{date}_floodmask.tif --polygonize")


if __name__ == "__main__":
    main()
