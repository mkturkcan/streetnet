"""City-scale asset map from per-panorama detections, in parallel square blocks.

For each tile of side `block` meters (UTM) that contains at least one panorama:
    1. read the detections of the panoramas inside the tile or within `halo` meters of it;
    2. relabel (pedestrian_signal -> traffic_signal), associate_poles, select_obs, localize (hybrid with
       automatic car-base occlusion switching);
    3. fuse observations per class with bounded average-linkage / cc / greedy clustering, close-view tail
       trimming, and anisotropic refinement;
    4. attach zero-GIS crosswalk distances (`d_cw` from `<run>/cw/`) and 3D occlusion-aware clear-view
       counts (`n_clear` from `<run>/occ/`) when present, then apply priors, support, and confidence filters;
    5. retain assets whose fused position lies inside the tile interior.
"""
from __future__ import annotations

import copy
import json
import math
import multiprocessing as mp
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from .fusion import apply_candidate_filters, associate_poles, fuse_with_params, localize, select_obs
from .geometry import ASSET_SIGHTLINE_Z, ray_to_equirect_pixel, unpack_occlusion_mask
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


def _attach_crosswalk_dist(assets: pd.DataFrame, cw_dir: Path, pano_ids) -> None:
    if not len(assets) or not cw_dir.is_dir():
        return
    pts = []
    for pid in pano_ids:
        p = cw_dir / f"{pid}.npy"
        if p.exists():
            try:
                arr = np.load(p)
                if len(arr):
                    pts.append(arr)
            except Exception:
                continue
    if not pts:
        return
    tree = cKDTree(np.concatenate(pts, axis=0))
    d_cw, _ = tree.query(assets[["x", "y"]].to_numpy(float))
    assets["d_cw"] = d_cw.astype(np.float32)


def _attach_clear_views(assets: pd.DataFrame, P_near: pd.DataFrame, R_lev: dict, occ_dir: Path,
                        radius: float = 15.0) -> None:
    """Count unobstructed sightlines (`n_clear`) within `radius` m from compressed per-pano occlusion masks."""
    if not len(assets) or not occ_dir.is_dir():
        return
    cxy = assets[["x", "y"]].to_numpy(float)
    clbls = assets["label"].to_numpy()
    pxy = P_near[["x", "y"]].to_numpy(float)
    pids = P_near.index.to_numpy()
    nbrs = cKDTree(pxy).query_ball_point(cxy, radius)
    by_pano: dict[str, list[int]] = {}
    for ci, js in enumerate(nbrs):
        for j in js:
            by_pano.setdefault(str(pids[j]), []).append(ci)

    n_clear = np.zeros(len(assets), dtype=np.int32)
    any_occ = False
    for pid, cis in by_pano.items():
        fp = occ_dir / f"{pid}.bin"
        if not fp.exists():
            continue
        try:
            occ = unpack_occlusion_mask(fp.read_bytes())
        except Exception:
            continue
        any_occ = True
        Ho, Wo = occ.shape
        row = P_near.loc[pid]
        px, py, heading, h_cam = float(row["x"]), float(row["y"]), float(row["heading"]), float(row["h"])
        hr = math.radians(heading)
        ch, sh = math.cos(hr), math.sin(hr)
        Rt = R_lev.get(pid, np.eye(3)).T
        for ci in cis:
            dx = cxy[ci, 0] - px
            dy = cxy[ci, 1] - py
            if math.hypot(dx, dy) < 1.5:
                continue
            z_mid = float(ASSET_SIGHTLINE_Z.get(str(clbls[ci]), 0.8))
            vc = Rt @ np.array([dx * ch - dy * sh, dx * sh + dy * ch, z_mid - h_cam], dtype=float)
            nm = float(np.linalg.norm(vc))
            if nm < 1e-6:
                continue
            u, v = ray_to_equirect_pixel((vc / nm)[None], Wo, Ho)
            ui = int(round(float(u[0]))) % Wo
            vi = int(np.clip(round(float(v[0])), 0, Ho - 1))
            u_idx = (np.arange(ui - 1, ui + 2)) % Wo
            v_idx = np.clip(np.arange(vi - 1, vi + 2), 0, Ho - 1)
            if float(occ[np.ix_(v_idx, u_idx)].mean()) < 0.5:
                n_clear[ci] += 1
    if any_occ:
        assets["n_clear"] = n_clear


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
    # Cluster first, attach tile-level zero-GIS crosswalk & occlusion context, then filter.
    o, a = fuse_with_params(obs, params, classes, apply_filters=False)
    if not len(a):
        return None if not missing else (None, None, int(len(ids)), int(len(dets)), missing)
    run_dir = _G["det_dir"].parent
    _attach_crosswalk_dist(a, run_dir / "cw", ids)
    _attach_clear_views(a, P.loc[ids], R, run_dir / "occ")
    a = apply_candidate_filters(a, params, use_priors=True, use_siglip=True, defer_verify=_G["defer_verify"])
    if not len(a):
        return None if not missing else (None, None, int(len(ids)), int(len(dets)), missing)
    a = a[(a.x >= x0) & (a.x < x0 + block) & (a.y >= y0) & (a.y < y0 + block)].copy()
    o = o[o["asset"].isin(a["asset_id"])]
    if not len(a):
        return None if not missing else (None, None, int(len(ids)), int(len(dets)), missing)
    best = dets.set_index("obs_id").loc[a["best_obs"].to_numpy(), ["pano_id", "row"]]
    a["best_pano"], a["best_row"] = best["pano_id"].to_numpy(), best["row"].to_numpy()
    a["estimator"] = a["asset_id"].map(o.groupby("asset")["loc_est"].agg(lambda s: s.mode().iat[0]))
    a["block"] = f"{bx}_{by}"
    a["dense"] = dense
    o = o.assign(block=f"{bx}_{by}")
    return a, o[[c for c in OBS_COLUMNS if c in o.columns] + ["block"]], int(len(ids)), int(len(dets)), missing


def build_map(
    index: pd.DataFrame,
    det_dir,
    params: dict,
    block: float = 1000.0,
    halo: float = 60.0,
    min_support_cap: int = 0,
    dense_panos=None,
    estimator: str = "hybrid",
    defer_verify: bool = False,
    workers: int = 32,
    log=print,
):
    """Returns (assets, observations, summary). `index` holds the panoramas to map (see index.read_index)."""
    t0 = time.time()
    det_dir = Path(det_dir)
    done = {f.stem for f in det_dir.glob("*.parquet")}
    P = index[index.index.isin(done)]
    _G.update(
        P=P,
        det_dir=det_dir,
        block=float(block),
        halo=float(halo),
        estimator=estimator,
        defer_verify=bool(defer_verify),
        params=cap_min_support(params, min_support_cap),
        params_dense=copy.deepcopy(params),
        dense=set(dense_panos or ()),
    )
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
    assets["source"] = assets["label"].map(lambda c: params["per_class"].get(c, {}).get("source", "sam31"))
    summary = dict(
        n_panos=int(len(P)),
        n_blocks=len(keys),
        n_detections_read=int(n_read),
        n_assets=int(len(assets)),
        assets_by_class=assets["label"].value_counts().to_dict(),
        block_m=float(block),
        halo_m=float(halo),
        min_support_cap=int(min_support_cap),
        defer_verify=bool(defer_verify),
        n_dense_blocks=int(assets.loc[assets["dense"], "block"].nunique()),
        epsg=int(index.attrs["epsg"]),
        estimator=estimator,
        classes_without_params=sorted(missing),
        seconds=round(time.time() - t0, 1),
    )
    return assets, obs, summary


def write_geojson(assets: pd.DataFrame, path: Path):
    feats = [{
        "type": "Feature",
        "geometry": {"type": "Point", "coordinates": [round(float(r.lon), 7), round(float(r.lat), 7)]},
        "properties": {"id": int(r.asset_id), "class": r.label, "support": int(r.support),
                       "confidence": round(float(r.conf), 3)},
    } for r in assets.itertuples()]
    path.write_text(json.dumps({"type": "FeatureCollection", "features": feats}))


def write_map(out_dir, name: str, assets: pd.DataFrame, obs: pd.DataFrame, summary: dict, pano_ids):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    assets.to_parquet(out / f"{name}_assets.parquet")
    obs.to_parquet(out / f"{name}_obs.parquet")
    (out / f"{name}_panos.txt").write_text("\n".join(pano_ids) + "\n")
    write_geojson(assets, out / f"{name}_assets.geojson")
    (out / f"{name}_summary.json").write_text(json.dumps(summary, indent=1, default=float))
