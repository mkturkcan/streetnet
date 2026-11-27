#!/usr/bin/env python
"""Build the Hugging Face dataset folder of the New York City asset map from a `streetnet map` output.

    python tools/export_nyc_dataset.py --index nyc_index.parquet --map run/map/<name>_assets.parquet \
        --summary run/map/<name>_summary.json --gt ground_truth.parquet --region-centers region_centers.parquet \
        --region-note "..." --selection "..." --version v1.0 --figure docs/assets/figures/nyc_map.jpg --out hf/

Needs <name>_panos.txt next to --map, and an index with the columns borough and date (YYYY-MM) besides those of
streetnet.index.

Writes
    data/assets.parquet          one row per asset (GeoParquet, WKB points in OGC:CRS84)
    eval/ground_truth.parquet    the NYC Open Data points used for the accuracy table
    eval/region_centers.parquet  camera positions whose 15 m disks form the evaluation region
    eval/metrics.json            streetnet.evaluate output for the published table
    figures/map.jpg              the overview figure (when --figure is given)
    summary.json                 counts and settings of the version (read by tools/readme_tables.py)
    README.md                    the dataset card
The table keeps the classes of streetnet/configs/classes.json and drops panorama identifiers.
"""
from __future__ import annotations

import argparse
import json
import shutil
import struct
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from streetnet import __version__  # noqa: E402
from streetnet.evaluate import evaluate  # noqa: E402
from streetnet.index import ll_to_xy, read_index, utm_epsg  # noqa: E402

CLASSES = json.loads((Path(__file__).resolve().parents[1] / "streetnet" / "configs" / "classes.json").read_text())
REPO = "https://github.com/mkturkcan/streetnet"
DATASET = "mehmetkeremturkcan/streetnet-nyc"
LICENSES = {"cc-by-nc-4.0": "CC BY-NC 4.0", "cc-by-4.0": "CC BY 4.0", "odbl": "ODbL 1.0"}
GT_SOURCES = [   # class, NYC Open Data dataset, agency, id, note
    ("tree", "2015 Street Tree Census - Tree Data", "Parks", "uvpi-gqnh", "census of 2015"),
    ("curb_ramp", "Pedestrian Ramp Locations", "DOT", "ufzp-rrqu", "survey of 2019"),
    ("fire_hydrant", "Hydrants", "DEP", "5bgh-vtsn", ""),
    ("storm_drain", "NYCDEP Citywide Catch Basins", "DEP", "2w2g-fk3i", ""),
    ("traffic_sign", "Street Sign Work Orders", "DOT", "qt6m-xctn", "current records; work orders, partial"),
    ("sign_pole", "Street Sign Work Orders", "DOT", "qt6m-xctn", "sign positions, partial"),
    ("street_light", "Street Sign Work Orders; Mobile Telecommunications Franchise Pole Reservation Locations",
     "DOT; OTI", "qt6m-xctn; tbgj-tdd6", "partial"),
    ("trash_can", "DSNY Litter Basket Inventory", "DSNY", "8znf-7b2c", "public litter baskets"),
    ("bike_rack", "Bicycle Parking", "DOT", "592z-n7dk", ""),
    ("bench", "Seating Locations", "DOT", "esmy-s8q5", ""),
    ("bus_shelter", "Bus Stop Shelters", "DOT", "t4f2-8md7", ""),
    ("parking_meter", "Parking Meters Locations and Status", "DOT", "693u-uax6", ""),
    ("kiosk", "LinkNYC Kiosk Locations", "OTI", "s4kf-3yrf", ""),
    ("fire_alarm_box", "In-Service Alarm Box Locations", "FDNY", "v57i-gtxb", "2019"),
]


def write_geoparquet(df: pd.DataFrame, path: Path):
    """Parquet with a WKB point column `geometry` and GeoParquet 1.1 metadata (CRS OGC:CRS84)."""
    wkb = [struct.pack("<BIdd", 1, 1, x, y) for x, y in zip(df["lon"].to_numpy(float), df["lat"].to_numpy(float))]
    t = pa.Table.from_pandas(df, preserve_index=False).append_column("geometry", pa.array(wkb, pa.binary()))
    geo = {"version": "1.1.0", "primary_column": "geometry",
           "columns": {"geometry": {"encoding": "WKB", "geometry_types": ["Point"],
                                    "bbox": [float(df.lon.min()), float(df.lat.min()), float(df.lon.max()), float(df.lat.max())]}}}
    t = t.replace_schema_metadata({**(t.schema.metadata or {}), b"geo": json.dumps(geo).encode()})
    pq.write_table(t, path, compression="zstd")


def fmt(x, nd=3):
    return "n/a" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{nd}f}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--index", required=True, help="panorama index with borough and date columns")
    ap.add_argument("--map", required=True)
    ap.add_argument("--summary", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--region-centers", required=True)
    ap.add_argument("--region-note", required=True, help="one sentence describing the evaluation region")
    ap.add_argument("--selection", required=True, help="one sentence describing the panoramas of this version")
    ap.add_argument("--version", required=True)
    ap.add_argument("--license", default="cc-by-nc-4.0")
    ap.add_argument("--figure")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    for sub in ("data", "eval", "figures"):
        (out / sub).mkdir(parents=True, exist_ok=True)

    index = read_index(a.index)
    summary = json.loads(Path(a.summary).read_text())
    used = index.loc[[p for p in Path(a.map.replace("_assets.parquet", "_panos.txt")).read_text().split() if p]]
    A = pd.read_parquet(a.map)
    A = A[A.label.isin(CLASSES["classes"])].copy()
    cls = CLASSES["classes"]
    pub = pd.DataFrame({
        "asset_id": np.arange(len(A), dtype=np.int64),
        "class": A.label.to_numpy(),
        "group": [CLASSES["groups"][cls[c]["group"]] for c in A.label],
        "experimental": [bool(cls[c].get("experimental", False)) for c in A.label],
        "lon": A.lon.to_numpy(float).round(7), "lat": A.lat.to_numpy(float).round(7),
        "support": A.support.to_numpy(np.int32),
        "confidence": A.conf.to_numpy(float).round(4),
        "position_source": np.where(A.estimator.to_numpy() == "depth", "range", "ground"),
        "spread_m": A.spread.to_numpy(float).round(2),
        "range_min_m": A.rng_min.to_numpy(float).round(2),
        "borough": index.loc[A.best_pano, "borough"].to_numpy() if "borough" in index.columns else None,
    })
    write_geoparquet(pub, out / "data" / "assets.parquet")

    gt = pd.read_parquet(a.gt)
    gt_pub = gt[["gt_key", "lon", "lat", "source"]].copy()
    gt_pub.to_parquet(out / "eval" / "ground_truth.parquet", index=False)
    rc = pd.read_parquet(a.region_centers)[["lon", "lat"]]
    rc.to_parquet(out / "eval" / "region_centers.parquet", index=False)
    # the accuracy table, computed from the published files exactly as `streetnet eval` does
    epsg = utm_epsg(float(rc.lon.median()), float(rc.lat.median()))
    ev_a = pub.rename(columns={"class": "label"})
    ev_a["x"], ev_a["y"] = ll_to_xy(ev_a.lon.to_numpy(), ev_a.lat.to_numpy(), epsg)
    gt_pub["x"], gt_pub["y"] = ll_to_xy(gt_pub.lon.to_numpy(), gt_pub.lat.to_numpy(), epsg)
    cx, cy = ll_to_xy(rc.lon.to_numpy(), rc.lat.to_numpy(), epsg)
    res = evaluate(ev_a, gt_pub, np.c_[cx, cy], region_r=15.0)
    res["n_region_centers"] = int(len(rc))
    (out / "eval" / "metrics.json").write_text(json.dumps(res, indent=1, default=float))
    if a.figure:
        shutil.copy(a.figure, out / "figures" / "map.jpg")

    # ------------------------------------------------------------------ card
    counts = pub["class"].value_counts()
    years = pd.to_numeric(used["date"].astype(str).str.slice(0, 4), errors="coerce") if "date" in used.columns else None
    y0, y1 = (int(years.min()), int(years.max())) if years is not None and years.notna().any() else (None, None)
    yr = f"{y0} to {y1}"
    R3 = res["R3"]
    class_rows = "\n".join(
        f"| `{c}` | {cls[c]['name']} | {CLASSES['groups'][cls[c]['group']]} | {int(counts.get(c, 0)):,} | "
        f"{'yes' if cls[c].get('experimental') else ''} |" for c in cls)
    acc_rows = "\n".join(
        f"| {cls[c]['name']} | {R3[c]['n_gt']:,} | {R3[c]['n_pred']:,} | {fmt(R3[c]['precision'])} | {fmt(R3[c]['recall'])} | "
        f"{fmt(R3[c]['f1'])} | {fmt(res['err'].get(c, {}).get('median'), 2)} |"
        for c in sorted(R3, key=lambda k: -R3[k]["n_gt"] if not k.startswith("_") else 0) if not c.startswith("_"))
    allr = R3["_all"]
    src_rows = "\n".join(f"| {cls[c]['name']} | {n} | {ag} | `{i}` | {note} |" for c, n, ag, i, note in GT_SOURCES
                         if c in R3)
    n_assets = len(pub)
    share1 = float((pub["support"] == 1).mean())
    size_cat = "1M<n<10M" if n_assets >= 1_000_000 else "100K<n<1M"
    by_borough = pub["borough"].value_counts().to_dict() if pub["borough"].notna().any() else {}
    borough_line = ", ".join(f"{b} {n:,}" for b, n in sorted(by_borough.items()))
    card = f"""---
license: {a.license}
pretty_name: StreetNet New York City Asset Map
tags:
- geospatial
- street-assets
- asset-inventory
- urban-computing
- new-york-city
size_categories:
- {size_cat}
configs:
- config_name: default
  data_files:
  - split: train
    path: data/assets.parquet
---

# StreetNet: New York City Asset Map

![Street asset density per 100 m across all five NYC boroughs and a close-up in Harlem colored by asset group](figures/map.jpg)

This dataset provides the citywide street asset inventory generated by [StreetNet]({REPO}) across all five boroughs of New York City. Each record corresponds to a single fused physical asset with its geographic coordinates, class label, multi-view panorama support count, and aggregated confidence score. Released to document city-scale system output, the inventory contains localization, classification, and coverage errors quantified in the accuracy evaluation below. No street-level imagery or panorama identifiers are included.

| Field | Details |
|---|---|
| Assets | {n_assets:,} across {pub['class'].nunique()} classes ({borough_line}) |
| Panoramas | {summary['n_panos']:,} street-level panoramas (Google Street View), captured {yr} |
| Coverage | {a.selection} |
| Accuracy | {fmt(allr['f1'])} micro-F1 at 3 m over {allr['n_gt']:,} NYC Open Data reference assets (precision {fmt(allr['precision'])}, recall {fmt(allr['recall'])}) |
| Release | {a.version}, built {date.today().isoformat()} with `streetnet` {__version__} |

## Files

| Path | Contents |
|---|---|
| `data/assets.parquet` | Fused asset inventory: one row per asset, GeoParquet 1.1 with WKB point geometry (`OGC:CRS84`) |
| `eval/ground_truth.parquet` | Municipal NYC Open Data reference points used for evaluation (`gt_key`, `lon`, `lat`, `source`) |
| `eval/region_centers.parquet` | Camera locations (`lon`, `lat`) whose 15 m disks define the evaluation region |
| `eval/metrics.json` | Evaluation metrics at 2, 3, and 5 m matching radii |
| `summary.json` | Asset counts, evaluation summary, and release metadata |

## Schema (`data/assets.parquet`)

| Column | Type | Description |
|---|---|---|
| `asset_id` | int64 | Zero-based integer asset identifier |
| `class` | string | Asset class key (see table below) |
| `group` | string | Functional asset group |
| `experimental` | bool | True if visual audit precision fell below 0.67 in the pilot study (up to 12 samples/class) or class was unaudited |
| `lon`, `lat` | float64 | Fused geographic coordinates (WGS 84) |
| `support` | int32 | Number of distinct panoramas contributing to the fused cluster |
| `confidence` | float64 | Aggregated confidence score: `1 - prod(1 - best_score_i)` across supporting panoramas |
| `position_source` | string | Majority localization estimator in the cluster: `range` (MoGe-2 metric range along the contact ray) or `ground` (leveled ground-plane intersection at 2.5 m camera height) |
| `spread_m` | float64 | Median distance of contributing observations from the fused asset position (meters) |
| `range_min_m` | float64 | Minimum horizontal distance among contributing observations (meters) |
| `borough` | string | Borough of the closest observing camera |
| `geometry` | binary | WKB Point geometry (`OGC:CRS84`) |

## Asset classes

| Key | Name | Group | Assets | Experimental |
|---|---|---|---:|---|
{class_rows}

## Construction pipeline

For each panorama, StreetNet projects four perspective views (90° FOV, −10° pitch, 1008 × 1008 px), segments 20 classes via SAM 3 from 25 natural-language prompts, segments curb ramps and catch basins via Mask2Former trained on Mapillary Vistas, and estimates metric 3D point maps via MoGe-2. Each instance is localized independently from a single panorama along its ground-contact ray: street trees, street lights, signs, signals, benches, and bike racks use MoGe-2 metric range when available, while remaining ground and road-surface classes intersect a leveled ground plane 2.5 m below the camera. Pole-mounted signs and signals inherit the contact ray and range of their supporting pole. Same-class observations within a class-specific radius (1–4 m) are fused into unique asset records filtered by minimum support and confidence. Fusion parameters for the 14 referenced classes were tuned on the 478-panorama Upper Manhattan pilot development split; the 8 unreferenced classes use a shared default configuration. See [docs/method.md]({REPO}/blob/main/docs/method.md) for complete pseudocode and constants.

## Evaluation

{a.region_note} Reference layers originate from NYC Open Data (listed below). Within each class, predicted assets and reference points within 3 m are matched one-to-one via Hungarian assignment, maximizing match cardinality and then minimizing total distance. Matched pairs whose reference lies inside the region count as true positives, unmatched predictions inside the region as false positives, and unmatched reference points inside the region as false negatives. Classes without municipal reference layers are omitted.

| Class | Ground truth | Predicted | Precision | Recall | F1 | Median error (m) |
|---|---:|---:|---:|---:|---:|---:|
{acc_rows}
| All classes | {allr['n_gt']:,} | {allr['n_pred']:,} | {fmt(allr['precision'])} | {fmt(allr['recall'])} | {fmt(allr['f1'])} | |

Median positional error is computed over pairs matched within 5 m. Several reference layers are partial or dated (work-order sign records, the 2015 tree census, and the 2019 pedestrian ramp survey), so valid detections absent from a municipal layer count as false positives. All numbers can be reproduced directly from this dataset:

```bash
pip install "streetnet @ git+{REPO}"
hf download {DATASET} --repo-type dataset --local-dir streetnet-nyc
cd streetnet-nyc
streetnet eval --map data/assets.parquet --gt eval/ground_truth.parquet \\
    --region-centers eval/region_centers.parquet --out metrics.json
```

| Class | NYC Open Data dataset | Agency | ID | Note |
|---|---|---|---|---|
{src_rows}

## Limitations

- Source panoramas span {y0}–{y1}; physical assets installed or removed outside their capture date may be missing or stale.
- Under sparse intermediate sampling, most assets are seen from a single viewpoint ({share1:.1%} have `support = 1` in this release). Capping minimum support at 1 increases recall at the expense of precision.
- Ground-plane localization incurs bias on sloped roadways and raised curbs, while monocular range errors grow with viewing distance.
- Classes marked `experimental` exhibited low precision or lacked sufficient reference data in the pilot study.

## Usage

```python
import pandas as pd
assets = pd.read_parquet("hf://datasets/{DATASET}/data/assets.parquet")
hydrants = assets[(assets["class"] == "fire_hydrant") & (assets["support"] >= 2)]
```

Reading `hf://` URIs requires `huggingface_hub`. For spatial operations, load directly as a GeoDataFrame via `geopandas.read_parquet`.

## License and attribution

This dataset is released under {LICENSES.get(a.license, a.license)}. Reference layers in `eval/` originate from NYC Open Data under the City of New York's terms of use. The StreetNet codebase is licensed under the GNU Affero General Public License v3.0 or later.

## Citation

```bibtex
@software{{turkcan2026streetnet,
  author  = {{Turkcan, Mehmet Kerem}},
  title   = {{{{StreetNet}}: Open-Vocabulary Street Asset Inventories from Panoramas at City Scale}},
  year    = {{2026}},
  version = {{{__version__}}},
  url     = {{{REPO}}}
}}
```
"""
    (out / "README.md").write_text(card)
    (out / "summary.json").write_text(json.dumps(dict(
        version=a.version, built=date.today().isoformat(), streetnet=__version__, selection=a.selection,
        n_assets=int(n_assets), n_classes=int(pub["class"].nunique()), n_panos=int(summary["n_panos"]),
        capture_years=[y0, y1], assets_by_class={c: int(counts.get(c, 0)) for c in cls}, assets_by_borough=by_borough,
        region_note=a.region_note, n_region_centers=int(len(rc)), accuracy_R3=R3, median_error_m=res["err"]), indent=1,
        default=float))
    print(f"[export] {n_assets} assets, {len(gt_pub)} reference points, {len(rc)} region centers -> {out}")
    print(f"[export] F1 at 3 m {allr['f1']:.3f} (P {allr['precision']:.3f}, R {allr['recall']:.3f}, {allr['n_gt']} ground truth)")


if __name__ == "__main__":
    main()
