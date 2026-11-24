"""City-scale asset map from per-panorama detections, in parallel square blocks.

For each block of side `block` metres (UTM) that contains at least one panorama:
    1. read the detections of the panoramas inside the block or within `halo` metres of it;
    2. relabel (pedestrian_signal -> traffic_signal), associate_poles, select_obs, localize (hybrid);
    3. fuse_with_params per class;
    4. keep the assets whose fused position lies inside the block.
Each asset is kept by exactly one block. The halo gives an asset near a block edge its observations from the
neighbouring block: R_max + r_c is at most 29 m for the shipped parameters, below the 60 m halo. Clusters of the
connected-component method can chain beyond r_c; such a cluster near a block edge can lose observations.
Classes without an entry in the parameter file are left out; their names are logged and stored in the summary.

Two parameter sets can be used at once. `min_support_cap` caps each class's minimum support (for sparsely
sampled areas, where an asset is rarely seen from two panoramas); blocks whose own panoramas are at least half
in `dense_panos` keep the parameter file's minimum support.
"""
from __future__ import annotations

import copy
import json
import multiprocessing as mp
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from .fusion import associate_poles, fuse_with_params, localize, select_obs
from .index import leveling, xy_to_ll

OBS_COLUMNS = ["pano_id", "row", "model", "label", "raw_label", "score", "view", "x0", "y0", "x1", "y1",
               "X", "Y", "rng", "bearing", "asset", "loc_src", "loc_est", "cut_bottom", "cut_side"]
_G: dict = {}


def cap_min_support(params: dict, cap: int) -> dict:
    p = copy.deepcopy(params)
    if cap:
        for v in p.get("per_class", {}).values():
            if "min_support" in v:
                v["min_support"] = min(int(v["min_support"]), cap)
        if "min_support" in p.get("default", {}):
            p["default"]["min_support"] = min(int(p["default"]["min_support"]), cap)
    return p


def read_detections(det_dir: Path, pano_ids) -> pd.DataFrame | None:
    frames = []
    for pid in pano_ids:
        try:
            d = pd.read_parquet(det_dir / f"{pid}.parquet")
        except Exception:
            continue
        if len(d):
            d["row"] = np.arange(len(d), dtype=np.int32)
            frames.append(d)
    return pd.concat(frames, ignore_index=True) if frames else None


def _run_block(key):
    P, block, halo = _G["P"], _G["block"], _G["halo"]
    bx, by = key
    x0, y0 = bx * block, by * block
    inside = (P.x >= x0) & (P.x < x0 + block) & (P.y >= y0) & (P.y < y0 + block)
    dense = bool(_G["dense"]) and P.index[inside].isin(_G["dense"]).mean() >= 0.5
    params = _G["params_dense"] if dense else _G["params"]
    near = (P.x >= x0 - halo) & (P.x < x0 + block + halo) & (P.y >= y0 - halo) & (P.y < y0 + block + halo)
    ids = P.index[near]
    dets = read_detections(_G["det_dir"], ids)
    if dets is None:
        return None
    for c in ("cut_bottom", "cut_side", "owner"):
        dets[c] = dets[c].fillna(False).astype(bool)
    dets["score"] = dets["score"].astype(float)
    dets["obs_id"] = np.arange(len(dets))
    dets["label"] = dets["label"].replace(params.get("relabel", {}))
    R = leveling(P, dets.pano_id.unique())
    obs = localize(select_obs(associate_poles(dets)), P, R, estimator=_G["estimator"])
    obs = obs[obs["loc_ok"]]
    classes = sorted(set(params["per_class"]) & set(obs["label"]))
    missing = set(obs["label"]) - set(params["per_class"])
    o, a = fuse_with_params(obs, params, classes)
    if not len(a):
        return None if not missing else (None, None, int(len(ids)), int(len(dets)), missing)
    a = a[(a.x >= x0) & (a.x < x0 + block) & (a.y >= y0) & (a.y < y0 + block)].copy()
    o = o[o["asset"].isin(a["asset_id"])]
    best = dets.set_index("obs_id").loc[a["best_obs"].to_numpy(), ["pano_id", "row"]]
    a["best_pano"], a["best_row"] = best["pano_id"].to_numpy(), best["row"].to_numpy()
    a["estimator"] = a["asset_id"].map(o.groupby("asset")["loc_est"].agg(lambda s: s.mode().iat[0]))
    a["block"] = f"{bx}_{by}"
    a["dense"] = dense
    o = o.assign(block=f"{bx}_{by}")
    return a, o[[c for c in OBS_COLUMNS if c in o.columns] + ["block"]], int(len(ids)), int(len(dets)), missing


def build_map(index: pd.DataFrame, det_dir, params: dict, block: float = 1000.0, halo: float = 60.0,
              min_support_cap: int = 0, dense_panos=None, estimator: str = "hybrid", workers: int = 32,
              log=print):
    """Returns (assets, observations, summary). `index` holds the panoramas to map (see index.read_index)."""
    t0 = time.time()
    det_dir = Path(det_dir)
    done = {f.stem for f in det_dir.glob("*.parquet")}
    P = index[index.index.isin(done)]
    _G.update(P=P, det_dir=det_dir, block=float(block), halo=float(halo), estimator=estimator,
              params=cap_min_support(params, min_support_cap), params_dense=copy.deepcopy(params),
              dense=set(dense_panos or ()))
    keys = sorted(set(zip(np.floor(P.x / block).astype(int), np.floor(P.y / block).astype(int))))
    log(f"[map] {len(P)} panoramas in {len(keys)} blocks of {block:g} m")
    A, O, n_read, missing = [], [], 0, set()
    ctx = mp.get_context("fork")
    with ProcessPoolExecutor(workers, mp_context=ctx) as ex:
        for k, r in enumerate(ex.map(_run_block, keys, chunksize=1)):
            if r is not None:
                if r[0] is not None:
                    A.append(r[0]), O.append(r[1])
                n_read += r[3]
                missing |= r[4]
            if (k + 1) % 200 == 0:
                log(f"[map] {k + 1}/{len(keys)} blocks ({time.time() - t0:.0f}s)")
    if missing:
        log(f"[map] classes without fusion parameters, left out: {sorted(missing)}")
    if not A:
        raise RuntimeError("no assets: check the detection folder and the index")
    assets = pd.concat(A, ignore_index=True)
    obs = pd.concat(O, ignore_index=True)
    gid = pd.Series(np.arange(len(assets)), index=(assets["block"] + ":" + assets["asset_id"].astype(str)).to_numpy())
    assets["asset_id"] = np.arange(len(assets))
    obs["asset"] = gid.loc[(obs["block"] + ":" + obs["asset"].astype(str)).to_numpy()].to_numpy()
    assets["lon"], assets["lat"] = xy_to_ll(assets["x"].to_numpy(), assets["y"].to_numpy(), index.attrs["epsg"])
    assets["source"] = assets["label"].map(lambda c: params["per_class"].get(c, {}).get("source", "sam3"))
    summary = dict(n_panos=int(len(P)), n_blocks=len(keys), n_detections_read=int(n_read),
                   n_assets=int(len(assets)), assets_by_class=assets["label"].value_counts().to_dict(),
                   block_m=float(block), halo_m=float(halo), min_support_cap=int(min_support_cap),
                   n_dense_blocks=int(assets.loc[assets["dense"], "block"].nunique()),
                   epsg=int(index.attrs["epsg"]), estimator=estimator, classes_without_params=sorted(missing),
                   seconds=round(time.time() - t0, 1))
    return assets, obs, summary


def write_map(out_dir, name: str, assets: pd.DataFrame, obs: pd.DataFrame, summary: dict, pano_ids):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    assets.to_parquet(out / f"{name}_assets.parquet")
    obs.to_parquet(out / f"{name}_obs.parquet")
    (out / f"{name}_panos.txt").write_text("\n".join(pano_ids) + "\n")
    feats = [{"type": "Feature", "geometry": {"type": "Point", "coordinates": [round(r.lon, 7), round(r.lat, 7)]},
              "properties": {"id": int(r.asset_id), "class": r.label, "support": int(r.support),
                             "confidence": round(float(r.conf), 3)}} for r in assets.itertuples()]
    (out / f"{name}_assets.geojson").write_text(json.dumps({"type": "FeatureCollection", "features": feats}))
    (out / f"{name}_summary.json").write_text(json.dumps(summary, indent=1, default=float))
