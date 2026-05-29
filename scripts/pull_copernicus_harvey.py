"""Download the Copernicus EMS EMSR229 (Hurricane Harvey) Houston flood extent.

Fetches the public Copernicus EMS activation archive page for EMSR229, finds the
Houston (AOI 01) delineation vector packages, downloads the candidates, reads the
flood-layer source dates, picks the product whose radar acquisition is closest to
our 2017-08-30 Sentinel-1 scene, and unzips that one into data/harvey/copernicus/.
Then it describes the shapefiles so we can confirm the flood and AOI layers.

The archive at mapping.emergency.copernicus.eu is public, no login. The newer
dashboard API (rapidmapping.emergency.copernicus.eu) gates pre-2018 activations
behind a Copernicus account; the activation archive does not, so we use it.

Run from repo root:
    uv run python scripts/pull_copernicus_harvey.py
    uv run python scripts/pull_copernicus_harvey.py --aoi houston --keep-all
"""
from __future__ import annotations

import argparse
import re
import shutil
import sys
import tempfile
import zipfile
from datetime import datetime
from pathlib import Path
from urllib.request import Request, urlopen

REPO_ROOT = Path(__file__).resolve().parents[1]
COP_DIR = REPO_ROOT / "data" / "harvey" / "copernicus"

CODE = "EMSR229"
ACTIVATION_PAGE = "https://mapping.emergency.copernicus.eu/activations/{code}/"
SCENE_DATE = datetime(2017, 8, 30)

# The S1 scene AOI we ran inference on, for an overlap sanity check.
S1_BBOX = (-95.75, 29.60, -95.10, 30.10)

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36")


def get(url: str, timeout: int = 300) -> bytes:
    req = Request(url, headers={"User-Agent": UA, "Accept-Language": "en-US,en;q=0.9"})
    with urlopen(req, timeout=timeout) as r:
        return r.read()


def list_vector_zips(code: str) -> list[str]:
    html = get(ACTIVATION_PAGE.format(code=code), timeout=120).decode("utf-8", "replace")
    urls = re.findall(r'href="([^"]+_vector\.zip)"', html)
    # absolute already, but guard against relative
    out = []
    for u in sorted(set(urls)):
        out.append(u if u.startswith("http") else "https://mapping.emergency.copernicus.eu" + u)
    return out


def flood_src_dates(extract_dir: Path) -> list[datetime]:
    """Per-polygon source acquisition dates from the flood (crisis_information) layer.

    Returns one date per polygon (with repeats), so a Counter over the result gives
    the dominant acquisition date weighted by how much of the layer it mapped.
    """
    import geopandas as gpd

    shp = next(
        (p for p in extract_dir.rglob("*.shp")
         if "crisis_information_poly" in p.name.lower()
         or "observedevent" in p.name.lower().replace("_", "")),
        None,
    )
    if shp is None:
        return []
    gdf = gpd.read_file(shp)
    col = next((c for c in gdf.columns if c.lower() in ("src_date", "src_dat", "srcdate")), None)
    if col is None:
        return []
    dates = []
    for v in gdf[col].dropna():
        try:
            dates.append(v.to_pydatetime() if hasattr(v, "to_pydatetime") else datetime.fromisoformat(str(v)[:10]))
        except (ValueError, TypeError):
            pass
    return dates


def date_score(extract_dir: Path):
    """(distance of dominant src_date to scene, distance of nearest src_date) in days."""
    from collections import Counter

    dates = flood_src_dates(extract_dir)
    if not dates:
        return (10_000, 10_000)
    mode_day = Counter(d.date() for d in dates).most_common(1)[0][0]
    mode_dist = abs((datetime(mode_day.year, mode_day.month, mode_day.day) - SCENE_DATE).days)
    near_dist = min(abs((d - SCENE_DATE).days) for d in dates)
    return (mode_dist, near_dist)


def describe_vectors(root: Path):
    import geopandas as gpd

    shps = sorted(root.rglob("*.shp"))
    if not shps:
        print("\nNo .shp extracted. Contents:")
        for p in sorted(root.rglob("*")):
            print(f"  {p.relative_to(root)}")
        return
    print("\nshapefiles extracted:")
    for shp in shps:
        try:
            gdf = gpd.read_file(shp)
            geom = sorted(set(gdf.geom_type))
            cols = [c for c in gdf.columns if c != gdf.geometry.name]
            print(f"  {shp.name:62s} {str(geom):26s} n={len(gdf):5d} crs={gdf.crs}")
            print(f"      fields: {cols}")
        except Exception as e:  # noqa: BLE001
            print(f"  {shp.name}: read error {e}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--aoi", default="houston", help="AOI name substring to target")
    p.add_argument("--keep-all", action="store_true",
                   help="extract every delineation product for the AOI, not just the best match")
    args = p.parse_args()

    print(f"querying Copernicus EMS activation archive for {CODE} ...")
    zips = list_vector_zips(CODE)
    if not zips:
        sys.exit(f"no vector packages listed for {CODE}; the archive page layout may have changed")
    print(f"{len(zips)} vector package(s) on the activation page")

    cands = [u for u in zips
             if args.aoi.lower() in u.rsplit("/", 1)[-1].lower()
             and "delineation" in u.lower()]
    if not cands:
        sys.exit(f"no delineation vector package matching AOI '{args.aoi}'. Available:\n  "
                 + "\n  ".join(u.rsplit("/", 1)[-1] for u in zips))

    print(f"\n{len(cands)} '{args.aoi}' delineation product(s):")
    for u in cands:
        print(f"  {u.rsplit('/', 1)[-1]}")

    COP_DIR.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix="emsr229_"))
    scored = []
    for u in cands:
        name = u.rsplit("/", 1)[-1]
        print(f"\ndownloading {name} ...")
        data = get(u)
        print(f"  {len(data)/1e6:.1f} MB")
        zpath = staging / name
        zpath.write_bytes(data)
        edir = staging / name[:-4]
        with zipfile.ZipFile(zpath) as z:
            z.extractall(edir)
        dist = date_score(edir)
        dates = sorted({d.date().isoformat() for d in flood_src_dates(edir)})
        print(f"  flood src_dates {dates}  -> distance-to-{SCENE_DATE.date()} (mode, nearest) = {dist} days")
        scored.append((dist, name, zpath, edir))

    scored.sort(key=lambda t: t[0])
    best = scored[0]
    print(f"\nbest match to our {SCENE_DATE.date()} scene: {best[1]}")

    chosen = scored if args.keep_all else [best]
    for _dist, name, zpath, edir in chosen:
        dest = COP_DIR / name[:-4]
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(edir, dest)
        shutil.copy2(zpath, COP_DIR / name)
        print(f"  -> {dest}")

    shutil.rmtree(staging, ignore_errors=True)
    describe_vectors(COP_DIR / best[1][:-4])
    print(f"\nour S1 scene bbox (WGS84): {S1_BBOX}")
    print("Next: uv run python scripts/validate_harvey.py")


if __name__ == "__main__":
    main()
