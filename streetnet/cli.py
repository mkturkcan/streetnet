"""Command line: streetnet detect | map | verify | eval | status."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

DETECT_HELP = """Run per-panorama asset detection over an index (resumable; spawns one process per GPU with --gpus)."""
MAP_HELP = """Fuse per-panorama detection records into a georeferenced asset map on CPUs."""
VERIFY_HELP = """Score and filter fused asset candidates with zero-shot multi-distractor SigLIP 2."""
EVAL_HELP = """Evaluate an asset map against reference points inside an evaluation region."""
STATUS_HELP = """Report completion progress and throughput for an active detection run."""


def _detect(a, argv):
    from .detect import launch, run_config, run_shard
    from .index import read_index, read_list
    from .vocabulary import DEFAULT_VOCABULARY
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
        run_config(Path(a.out), a.views, Path(a.vocabulary or DEFAULT_VOCABULARY), a.threshold,
                   sam_checkpoint=a.sam_checkpoint, depth_model=a.depth_model)
        return launch(rest, a.gpus.split(","), Path(a.out), a.per_gpu)
    index = read_index(a.index)
    order = read_list(a.order) if a.order else None
    return run_shard(
        index,
        a.images,
        a.out,
        order=order,
        shard=a.shard,
        n_views=a.views,
        vocabulary=a.vocabulary,
        threshold=a.threshold,
        compile=a.compile,
        workers=a.workers,
        keep_raw=a.keep_raw,
        limit=a.limit,
        sam_checkpoint=a.sam_checkpoint,
        depth_model=a.depth_model,
        log=lambda s: print(s, flush=True),
    )


def _map(a):
    from .index import read_index, read_list
    from .mapping import build_map, write_map
    from .vocabulary import DEFAULT_PARAMS
    index = read_index(a.index)
    if a.panos:
        index = index[index.index.isin(set(read_list(a.panos)))]
    params = json.loads(Path(a.params or DEFAULT_PARAMS).read_text())
    dense = read_list(a.dense_panos) if a.dense_panos else None
    assets, obs, summary = build_map(
        index,
        Path(a.run) / "det",
        params,
        block=a.block,
        halo=a.halo,
        min_support_cap=a.min_support_cap,
        dense_panos=dense,
        defer_verify=a.defer_verify,
        workers=a.workers,
        log=lambda s: print(s, flush=True),
    )
    summary.update(name=a.name, params=str(a.params or DEFAULT_PARAMS), panos_list=a.panos, dense_panos=a.dense_panos)
    out = Path(a.out or Path(a.run) / "map")
    done = {f.stem for f in (Path(a.run) / "det").glob("*.parquet")}
    write_map(out, a.name, assets, obs, summary, [p for p in index.index if p in done])
    print(f"[map] {len(assets)} assets from {summary['n_panos']} panoramas in {summary['seconds']}s -> "
          f"{out / (a.name + '_assets.parquet')}", flush=True)
    return 0


def _verify(a, argv):
    import pandas as pd

    from .detect import views_of
    from .fusion import apply_candidate_filters, class_params
    from .geometry import detection_views
    from .index import read_index
    from .mapping import cap_min_support, write_geojson
    from .models.siglip2 import launch_verify, score_candidates
    from .vocabulary import DEFAULT_PARAMS

    map_path = Path(a.map)
    out_path = Path(a.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    params = cap_min_support(json.loads(Path(a.params or DEFAULT_PARAMS).read_text()), a.min_support_cap)

    if a.gpus:
        rest, skip = [], False
        for t in argv:
            if skip:
                skip = False
                continue
            if t == "--gpus":
                skip = True
                continue
            if t.startswith("--gpus="):
                continue
            rest.append(t)
        gpus = a.gpus.split(",")
        work = out_path.parent / f".verify_{out_path.stem}"
        rc = launch_verify(rest, gpus, work)
        if rc != 0:
            return rc
        shards = [pd.read_parquet(work / f"siglip_shard_{i}.parquet") for i in range(len(gpus))
                  if (work / f"siglip_shard_{i}.parquet").exists()]
        scores = pd.concat(shards, ignore_index=True) if shards else pd.DataFrame(
            columns=["asset_id", "sig2_tight", "sig2_wide", "sig2_comb"]
        )
    else:
        index = read_index(a.index)
        assets_in = _table(map_path)
        if a.run and (Path(a.run) / "run.json").exists():
            views = views_of(json.loads((Path(a.run) / "run.json").read_text()))
        else:
            views = detection_views(a.views)
        active = {str(c) for c in assets_in["label"].dropna().unique()
                  if float(class_params(params, str(c)).get("w_sig", 0.0)) > 0.0}
        scores = score_candidates(assets_in, index, a.images, views, batch_size=a.batch_size,
                                  shard=a.shard, active_classes=active, log=lambda s: print(s, flush=True))
        if a.scores_out:
            scores.to_parquet(a.scores_out, index=False)
            return 0

    assets = _table(map_path).copy()
    if "asset_id" not in assets.columns:
        assets["asset_id"] = np.arange(len(assets), dtype=np.int64)
    assets = assets.drop(columns=[c for c in ("sig2_tight", "sig2_wide", "sig2_comb") if c in assets.columns])
    assets = assets.merge(scores.drop_duplicates("asset_id"), on="asset_id", how="left")
    for c in ("sig2_tight", "sig2_wide", "sig2_comb"):
        assets[c] = assets[c].fillna(0.0).astype(np.float32)
    verified = apply_candidate_filters(assets, params, use_priors=True, use_siglip=True, defer_verify=False)
    verified.to_parquet(out_path, index=False)
    if {"lon", "lat"} <= set(verified.columns):
        write_geojson(verified, out_path.with_suffix(".geojson"))
    print(f"[verify] {len(verified)} / {len(assets)} candidates retained -> {out_path}", flush=True)
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
    if not {"x", "y"} <= set(assets.columns) or a.epsg:
        assets["x"], assets["y"] = ll_to_xy(assets["lon"].to_numpy(), assets["lat"].to_numpy(), epsg)
    if not {"x", "y"} <= set(gt.columns) or a.epsg or a.gt_datum == "nad83":
        if {"lon", "lat"} <= set(gt.columns):
            gt["x"], gt["y"] = ll_to_xy(gt["lon"].to_numpy(), gt["lat"].to_numpy(), epsg, datum=a.gt_datum)
    res = evaluate(assets, gt, centers, region_r=a.region_radius)
    res["n_region_centers"] = int(len(centers))
    res["region_radius_m"] = a.region_radius
    res["gt_datum"] = a.gt_datum
    print(f"[eval] region: {len(centers)} disks of {a.region_radius:g} m; matching radius 3 m (gt_datum={a.gt_datum})")
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
    ap = argparse.ArgumentParser(prog="streetnet", description="Open-vocabulary street asset inventories from street-level panoramas.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("detect", help=DETECT_HELP, description=DETECT_HELP)
    d.add_argument("--index", required=True, help="panorama index (Parquet or CSV); see docs/data.md")
    d.add_argument("--images", help="directory of <pano_id>.jpg files (unless the index has an image column)")
    d.add_argument("--out", required=True, help="output run directory")
    d.add_argument("--order", help="text file of panorama IDs specifying processing order (default: index order)")
    d.add_argument("--gpus", help="comma-separated GPU IDs (launches one worker process per GPU)")
    d.add_argument("--per-gpu", type=int, default=1, help="worker processes per GPU with --gpus")
    d.add_argument("--shard", default="0/1", help="i/n: process panoramas i, i+n, i+2n, ...")
    d.add_argument("--views", type=int, default=4, help="perspective views per panorama (4 or 8)")
    d.add_argument("--vocabulary", help="prompt vocabulary JSON (default: streetnet/configs/prompts.json)")
    d.add_argument("--threshold", type=float, default=0.22, help="SAM 3.1 score threshold")
    d.add_argument("--sam-checkpoint", default="sam3.1", help="'sam3.1' (default), 'sam3', or path to checkpoint")
    d.add_argument("--depth-model", default="moge3_g", choices=("moge3_g", "moge3", "moge2"),
                   help="monocular metric geometry model (default: moge3_g)")
    d.add_argument("--compile", action="store_true", help="compile SAM 3.1 with torch.compile at startup")
    d.add_argument("--workers", type=int, default=8, help="CPU post-processing workers per GPU process")
    d.add_argument("--keep-raw", action="store_true", help="persist raw SAM 3.1 / Mask2Former records and label maps")
    d.add_argument("--limit", type=int, default=0, help="process at most N panoramas")

    m = sub.add_parser("map", help=MAP_HELP, description=MAP_HELP)
    m.add_argument("--index", required=True)
    m.add_argument("--run", required=True, help="run directory produced by streetnet detect")
    m.add_argument("--name", required=True, help="map output prefix (<out>/<name>_assets.parquet, etc.)")
    m.add_argument("--out", help="output directory (default: <run>/map)")
    m.add_argument("--panos", help="text file of panorama IDs to include")
    m.add_argument("--params", help="fusion parameter JSON (default: streetnet/configs/fusion_params.json)")
    m.add_argument("--block", type=float, default=1000.0, help="tile side length in meters")
    m.add_argument("--halo", type=float, default=60.0, help="tile halo width in meters")
    m.add_argument("--min-support-cap", type=int, default=0, help="cap per-class minimum support (0 = uncapped)")
    m.add_argument("--dense-panos", help="text file of panorama IDs; tiles with >=50%% dense panoramas keep uncapped support")
    m.add_argument("--defer-verify", action="store_true",
                   help="retain fused candidates prior to SigLIP 2 thresholding for downstream `streetnet verify`")
    m.add_argument("--workers", type=int, default=32)

    v = sub.add_parser("verify", help=VERIFY_HELP, description=VERIFY_HELP)
    v.add_argument("--index", required=True, help="panorama index (Parquet or CSV)")
    v.add_argument("--images", help="directory of <pano_id>.jpg files (unless the index has an image column)")
    v.add_argument("--map", required=True, help="candidate asset map Parquet produced by `streetnet map --defer-verify`")
    v.add_argument("--out", required=True, help="output verified asset map Parquet path")
    v.add_argument("--run", help="optional detect run directory (reads view geometry from run.json)")
    v.add_argument("--views", type=int, default=4, help="perspective view count if --run is omitted")
    v.add_argument("--params", help="fusion parameter JSON (default: streetnet/configs/fusion_params.json)")
    v.add_argument("--gpus", help="comma-separated GPU IDs for sharded verification")
    v.add_argument("--shard", default="0/1", help="i/n shard when running single-process")
    v.add_argument("--min-support-cap", type=int, default=0,
                   help="cap per-class minimum support during final filtering (match `streetnet map --min-support-cap`)")
    v.add_argument("--batch-size", type=int, default=128, help="SigLIP 2 image batch size")
    v.add_argument("--scores-out", help=argparse.SUPPRESS)

    e = sub.add_parser("eval", help=EVAL_HELP, description=EVAL_HELP)
    e.add_argument("--map", required=True, help="asset map (Parquet or CSV): label/class and x, y or lon, lat")
    e.add_argument("--gt", required=True, help="reference table (Parquet or CSV): gt_key and x, y or lon, lat")
    e.add_argument("--index", help="panorama index defining evaluation region disks")
    e.add_argument("--region", help="text file of panorama IDs forming the evaluation region (default: all)")
    e.add_argument("--region-centers", help="table with lon, lat disk centers (alternative to --region)")
    e.add_argument("--region-radius", type=float, default=15.0)
    e.add_argument("--epsg", type=int, help="metric CRS for lon/lat inputs (default: index UTM zone)")
    e.add_argument("--gt-datum", default="wgs84", choices=("wgs84", "nad83"),
                   help="geodetic datum of lon/lat reference points (use 'nad83' for NAD83(2011) municipal GIS)")
    e.add_argument("--out", help="write evaluation metrics JSON to this path")

    s = sub.add_parser("status", help=STATUS_HELP, description=STATUS_HELP)
    s.add_argument("--index", required=True)
    s.add_argument("--run", required=True)

    a = ap.parse_args(argv)
    if a.cmd == "detect":
        return _detect(a, argv[1:])
    if a.cmd == "verify":
        return _verify(a, argv[1:])
    return {"map": _map, "eval": _eval, "status": _status}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
