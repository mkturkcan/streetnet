"""Command line: streetnet detect | map | eval | status."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

DETECT_HELP = """Detect assets in the panoramas of the index (resumable). With --gpus, one process per GPU."""
MAP_HELP = """Fuse the detections of a run folder into an asset map (CPU)."""
EVAL_HELP = """Compare an asset map with ground-truth points inside an evaluation region."""
STATUS_HELP = """Progress of a detection run."""


def _detect(a, argv):
    from .detect import launch, run_shard
    from .index import read_index, read_list
    if a.gpus:
        rest, skip = [], False
        for t in argv:
            if skip:
                skip = False
                continue
            if t in ("--gpus", "--per-gpu"):
                skip = True
                continue
            if t.startswith(("--gpus=", "--per-gpu=")):
                continue
            rest.append(t)
        from .detect import run_config
        from .vocabulary import DEFAULT_VOCABULARY
        run_config(Path(a.out), a.views, Path(a.vocabulary or DEFAULT_VOCABULARY), a.threshold)   # before the shards start
        return launch(rest, a.gpus.split(","), Path(a.out), a.per_gpu)
    index = read_index(a.index)
    order = read_list(a.order) if a.order else None
    return run_shard(index, a.images, a.out, order=order, shard=a.shard, n_views=a.views, vocabulary=a.vocabulary,
                     threshold=a.threshold, compile=a.compile, workers=a.workers, keep_raw=a.keep_raw,
                     limit=a.limit, log=lambda s: print(s, flush=True))


def _map(a):
    from .index import read_index, read_list
    from .mapping import build_map, write_map
    from .vocabulary import DEFAULT_PARAMS
    index = read_index(a.index)
    if a.panos:
        index = index[index.index.isin(set(read_list(a.panos)))]
    params = json.loads(Path(a.params or DEFAULT_PARAMS).read_text())
    dense = read_list(a.dense_panos) if a.dense_panos else None
    assets, obs, summary = build_map(index, Path(a.run) / "det", params, block=a.block, halo=a.halo,
                                     min_support_cap=a.min_support_cap, dense_panos=dense, workers=a.workers,
                                     log=lambda s: print(s, flush=True))
    summary.update(name=a.name, params=str(a.params or DEFAULT_PARAMS), panos_list=a.panos, dense_panos=a.dense_panos)
    out = Path(a.out or Path(a.run) / "map")
    done = {f.stem for f in (Path(a.run) / "det").glob("*.parquet")}
    write_map(out, a.name, assets, obs, summary, [p for p in index.index if p in done])
    print(f"[map] {len(assets)} assets from {summary['n_panos']} panoramas in {summary['seconds']}s -> "
          f"{out / (a.name + '_assets.parquet')}", flush=True)
    return 0


def _table(path):
    import pandas as pd
    return pd.read_parquet(path) if str(path).endswith(".parquet") else pd.read_csv(path)


def _eval(a):
    from .evaluate import evaluate, format_table
    from .index import ll_to_xy, read_index, read_list, utm_epsg
    if not (a.index or a.region_centers):
        raise SystemExit("eval needs --index (region from its panoramas) or --region-centers")
    index = read_index(a.index) if a.index else None
    if a.region_centers:
        rc = _table(a.region_centers)
        epsg = a.epsg or (index.attrs["epsg"] if index is not None else utm_epsg(rc.lon.median(), rc.lat.median()))
        cx, cy = ll_to_xy(rc["lon"].to_numpy(), rc["lat"].to_numpy(), epsg)
        centers = np.c_[cx, cy]
    else:
        epsg = a.epsg or index.attrs["epsg"]
        region = set(read_list(a.region)) if a.region else set(index.index)
        centers = index.loc[index.index.isin(region), ["x", "y"]].to_numpy(float)
    assets, gt = _table(a.map), _table(a.gt)
    if "label" not in assets.columns:
        assets = assets.rename(columns={"class": "label"})
    for t in (assets, gt):
        if not {"x", "y"} <= set(t.columns) or a.epsg:
            t["x"], t["y"] = ll_to_xy(t["lon"].to_numpy(), t["lat"].to_numpy(), epsg)
    res = evaluate(assets, gt, centers, region_r=a.region_radius)
    res["n_region_centers"] = int(len(centers))
    res["region_radius_m"] = a.region_radius
    print(f"[eval] region: {len(centers)} disks of {a.region_radius:g} m; matching radius 3 m")
    print(format_table(res, "R3"))
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1, default=float))
    return 0


def _status(a):
    from .index import read_index
    index = read_index(a.index)
    det = Path(a.run) / "det"
    files = list(det.glob("*.parquet"))
    done = len({f.stem for f in files} & set(index.index))
    now = time.time()
    recent = sum(1 for f in files if now - f.stat().st_mtime < 3600)
    print(f"{done} of {len(index)} panoramas done ({100 * done / max(1, len(index)):.1f}%); {recent} in the last hour")
    if recent:
        print(f"at that rate the rest takes {(len(index) - done) / recent:.1f} h")
    return 0


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    ap = argparse.ArgumentParser(prog="streetnet", description="Street-asset maps from street-level panoramas.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("detect", help=DETECT_HELP, description=DETECT_HELP)
    d.add_argument("--index", required=True, help="panorama index (Parquet or CSV), see docs/data.md")
    d.add_argument("--images", help="folder of <pano_id>.jpg (unless the index has an image column)")
    d.add_argument("--out", required=True, help="run folder")
    d.add_argument("--order", help="text file of pano ids: process in this order (default: index order)")
    d.add_argument("--gpus", help="comma-separated GPU ids: one process per GPU")
    d.add_argument("--per-gpu", type=int, default=1, help="processes per GPU with --gpus")
    d.add_argument("--shard", default="0/1", help="i/n: this process takes panoramas i, i+n, i+2n, ...")
    d.add_argument("--views", type=int, default=4, help="detection views per panorama (4 or 8)")
    d.add_argument("--vocabulary", help="prompt file (default: streetnet/configs/prompts.json)")
    d.add_argument("--threshold", type=float, default=0.25, help="SAM 3 score threshold")
    d.add_argument("--compile", action="store_true", help="torch.compile SAM 3 (compilation when the process starts)")
    d.add_argument("--workers", type=int, default=8, help="CPU worker processes per GPU process")
    d.add_argument("--keep-raw", action="store_true", help="also write SAM 3 and Mask2Former records and label maps")
    d.add_argument("--limit", type=int, default=0, help="process at most N panoramas")

    m = sub.add_parser("map", help=MAP_HELP, description=MAP_HELP)
    m.add_argument("--index", required=True)
    m.add_argument("--run", required=True, help="run folder of streetnet detect")
    m.add_argument("--name", required=True, help="map name: files <out>/<name>_assets.parquet etc.")
    m.add_argument("--out", help="output folder (default: <run>/map)")
    m.add_argument("--panos", help="text file of pano ids: map these panoramas")
    m.add_argument("--params", help="fusion parameters (default: streetnet/configs/fusion_params.json)")
    m.add_argument("--block", type=float, default=1000.0, help="block side in metres")
    m.add_argument("--halo", type=float, default=60.0, help="halo around each block in metres")
    m.add_argument("--min-support-cap", type=int, default=0, help="cap each class's minimum support (0 = no cap)")
    m.add_argument("--dense-panos", help="text file of pano ids; blocks at least half made of them keep the uncapped minimum support")
    m.add_argument("--workers", type=int, default=32)

    e = sub.add_parser("eval", help=EVAL_HELP, description=EVAL_HELP)
    e.add_argument("--map", required=True, help="assets (Parquet or CSV): label or class, and x, y or lon, lat")
    e.add_argument("--gt", required=True, help="ground truth (Parquet or CSV): gt_key, and x, y or lon, lat")
    e.add_argument("--index", help="panorama index; the region is made of disks around its panoramas")
    e.add_argument("--region", help="text file of pano ids of the index that form the region (default: all)")
    e.add_argument("--region-centers", help="table with lon, lat of the disk centres (instead of --region)")
    e.add_argument("--region-radius", type=float, default=15.0)
    e.add_argument("--epsg", type=int, help="metric CRS for lon/lat inputs (default: the index's UTM zone)")
    e.add_argument("--out", help="write the metrics as JSON")

    s = sub.add_parser("status", help=STATUS_HELP, description=STATUS_HELP)
    s.add_argument("--index", required=True)
    s.add_argument("--run", required=True)

    a = ap.parse_args(argv)
    if a.cmd == "detect":
        return _detect(a, argv[1:])
    return {"map": _map, "eval": _eval, "status": _status}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
