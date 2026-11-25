"""Vector tiles of the asset map for the map viewer (docs/viewer/).

    python tools/viewer/build_tiles.py --assets data/assets.parquet --out streetnet-nyc.pmtiles

Reads the published GeoParquet (data/assets.parquet of the Hugging Face dataset) and writes one PMTiles archive with a
point layer "assets" with compact integer properties (k class index in CLASSES, c confidence in percent,
s support, r position source 1 = range and 0 = ground plane, b borough index in BOROUGHS) to keep tiles and browser memory small. Needs tippecanoe
(2.17 or later, conda-forge: `conda install -c conda-forge tippecanoe`).

Zoom 15 and above hold every asset. Below it, tippecanoe keeps an even sample of at most --max-features points per
tile (--drop-densest-as-needed), so that a citywide view holds a few hundred thousand points at most.
"""
import argparse
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import pyarrow.parquet as pq

CLASSES = ["street_light", "sign_pole", "traffic_signal", "traffic_sign", "bus_stop_sign", "manhole", "storm_drain",
           "curb_ramp", "fire_hydrant", "bollard", "barrier", "fire_alarm_box", "trash_can", "tree", "planter", "bus_shelter",
           "bike_rack", "bike_share_dock", "parking_meter", "bench", "mailbox", "kiosk"]      # order used by docs/viewer
BOROUGHS = ["Manhattan", "Bronx", "Brooklyn", "Queens", "Staten Island"]
COLS = ["asset_id", "class", "group", "confidence", "support", "position_source", "borough", "lon", "lat"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--assets", required=True, help="data/assets.parquet of the dataset")
    ap.add_argument("--out", required=True, help="output .pmtiles")
    ap.add_argument("--tippecanoe", default=shutil.which("tippecanoe") or "tippecanoe")
    ap.add_argument("--max-zoom", type=int, default=15)
    ap.add_argument("--max-features", type=int, default=8000, help="points per tile below the full-detail zoom")
    a = ap.parse_args()
    t = pq.read_table(a.assets, columns=COLS).to_pandas()
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "assets.ndjson"
        with open(src, "w") as f:
            for r in t.itertuples(index=False):
                f.write(json.dumps({"type": "Feature", "id": int(r.asset_id),
                                    "geometry": {"type": "Point", "coordinates": [round(r.lon, 7), round(r.lat, 7)]},
                                    "properties": {"k": CLASSES.index(r[1]), "c": int(round(100 * float(r.confidence))),
                                                   "s": int(r.support), "r": int(r.position_source == "range"),
                                                   "b": BOROUGHS.index(r.borough) if r.borough in BOROUGHS else -1}}) + "\n")
        cmd = [a.tippecanoe, "-o", a.out, "--force", "-l", "assets", "-n", "StreetNet asset map",
               "-A", "Asset map: StreetNet. Basemap attribution is shown by the viewer.",
               "-Z", "10", "-z", str(a.max_zoom), "-r1", "--drop-densest-as-needed", "--extend-zooms-if-still-dropping",
               "--maximum-tile-bytes=400000", f"--maximum-tile-features={a.max_features}", "-P", "-q", str(src)]
        subprocess.run(cmd, check=True)
    print(f"[tiles] {len(t):,} assets -> {a.out} ({Path(a.out).stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
