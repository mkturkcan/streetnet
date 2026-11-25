"""Vector tiles of the asset map for the map viewer (docs/viewer/).

    python tools/viewer/build_tiles.py --assets data/assets.parquet --out streetnet-nyc.pmtiles

Reads the published GeoParquet (data/assets.parquet of the Hugging Face dataset) and writes one PMTiles archive with a
point layer "assets" (properties: id, class, group, confidence, support, position_source, borough). Needs tippecanoe
(2.17 or later, conda-forge: `conda install -c conda-forge tippecanoe`).

Zoom 15 and above hold every asset. Below it, tippecanoe keeps an even sample of points per tile (--drop-densest-as-needed)
so that each tile stays small; the viewer shows the full map from street level.
"""
import argparse
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import pyarrow.parquet as pq

COLS = ["asset_id", "class", "group", "confidence", "support", "position_source", "borough", "lon", "lat"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--assets", required=True, help="data/assets.parquet of the dataset")
    ap.add_argument("--out", required=True, help="output .pmtiles")
    ap.add_argument("--tippecanoe", default=shutil.which("tippecanoe") or "tippecanoe")
    ap.add_argument("--max-zoom", type=int, default=15)
    a = ap.parse_args()
    t = pq.read_table(a.assets, columns=COLS).to_pandas()
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "assets.ndjson"
        with open(src, "w") as f:
            for r in t.itertuples(index=False):
                f.write(json.dumps({"type": "Feature", "id": int(r.asset_id),
                                    "geometry": {"type": "Point", "coordinates": [round(r.lon, 7), round(r.lat, 7)]},
                                    "properties": {"class": r[1], "group": r[2], "confidence": round(float(r.confidence), 3),
                                                   "support": int(r.support), "position_source": r.position_source,
                                                   "borough": r.borough}}) + "\n")
        cmd = [a.tippecanoe, "-o", a.out, "--force", "-l", "assets", "-n", "StreetNet asset map",
               "-A", "Asset map: StreetNet. Basemap attribution is shown by the viewer.",
               "-Z", "10", "-z", str(a.max_zoom), "-r1", "--drop-densest-as-needed", "--extend-zooms-if-still-dropping",
               "--maximum-tile-bytes=1500000", "-P", "-q", str(src)]
        subprocess.run(cmd, check=True)
    print(f"[tiles] {len(t):,} assets -> {a.out} ({Path(a.out).stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
