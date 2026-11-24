"""From per-panorama detections to assets: pole association, localization and clustering.

    associate_poles     pole-mounted detections borrow the contact point of the pole below them
    select_obs          owner-view rows without the duplicate flag
    localize            contact ray -> world position (flat ground, metric range, or the hybrid of both)
    fuse_with_params    per class: range filter, clustering across panoramas, support and confidence filter

The functions take and return pandas tables in the schema of streetnet.instances, plus the columns
mount, dup and drng written by streetnet.detect.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .geometry import ray_to_equirect_pixel, to_world

POLE_CARRIERS = ("sign_pole", "street_light", "utility_pole", "pole")
RAY_COLS = ("dx", "dy", "dz")
# Classes localized from metric range in the hybrid estimator. On the pilot's development split (478 panoramas,
# 510 reference trees) range gave trees an F1 at 3 m of 0.62 against 0.41 for the flat ground; pole-mounted classes
# have no ground contact of their own.
DEPTH_CLASSES = {"tree", "street_light", "traffic_sign", "bench", "bike_rack", "traffic_signal", "bus_stop_sign",
                 "utility_pole"}
DEFAULT_PARAMS = dict(r_c=2.0, R_max=20.0, min_support=1, score_thr=0.0, method="greedy", pos="median",
                      min_range=1.0, w_floor=3.0)


# ----------------------------------------------------------------------------- mounts and pole association


def row_mounts(df: pd.DataFrame, mounts: dict | None = None) -> pd.Series:
    """Per-row mount type: the detection's `mount` column, else the vocabulary's, else 'ground'."""
    m = df["label"].map(mounts or {})
    if "mount" in df.columns:
        m = df["mount"].where(df["mount"].notna() & df["mount"].astype(str).ne(""), m)
    return m.fillna("ground")


def associate_poles(df: pd.DataFrame, mounts: dict | None = None, carriers=POLE_CARRIERS,
                    margin_frac: float = 0.15, gap_frac: float = 0.35) -> pd.DataFrame:
    """Give pole-mounted detections the contact point of their pole.

    For a pole-mounted detection s and each carrier detection p (sign pole, street light, utility pole) in
    the same panorama and view:
        margin = max(0.15 width(s), 4 px)
        p qualifies if  p.x0 <= s.x1 + margin  and  p.x1 >= s.x0 - margin     (columns overlap)
                        p.y1 >  s.y1                                           (the pole reaches lower)
                        p.y0 <= s.y1 + 0.35 height(p)                          (and up to the object)
        cost = |p.bot_px - centre column of s| / width(s) + 0.001 (p.y1 - s.y1)
    The cheapest qualifying carrier's contact columns (bot_*, cut_bottom) are copied to s. Adds loc_src
    ('self', 'pole', or 'none' for a pole-mounted detection without a carrier) and pole_obs (obs_id)."""
    out = df.copy()
    is_pole = row_mounts(out, mounts).eq("pole").to_numpy()
    out["loc_src"] = np.where(is_pole, "none", "self")
    out["pole_obs"] = -1
    sg = out[is_pole]
    pl = out[out["label"].isin(carriers)]
    if not len(sg) or not len(pl):
        return out
    a = sg[["obs_id", "pano_id", "view", "x0", "x1", "y0", "y1"]].rename(
        columns=lambda c: c if c in ("pano_id", "view") else "s_" + c)
    b = pl[["obs_id", "pano_id", "view", "x0", "x1", "y0", "y1", "bot_px", "cut_bottom"]].rename(
        columns=lambda c: c if c in ("pano_id", "view") else "p_" + c)
    m = a.merge(b, on=["pano_id", "view"])
    if not len(m):
        return out
    sw = (m.s_x1 - m.s_x0 + 1).astype(float)
    mg = np.maximum(margin_frac * sw, 4.0)
    ph = (m.p_y1 - m.p_y0 + 1).astype(float)
    ok = (m.p_x0 <= m.s_x1 + mg) & (m.p_x1 >= m.s_x0 - mg) & (m.p_y1 > m.s_y1) & (m.p_y0 <= m.s_y1 + gap_frac * ph)
    m = m[ok].copy()
    if not len(m):
        return out
    m["cost"] = np.abs(m.p_bot_px - 0.5 * (m.s_x0 + m.s_x1)) / sw.loc[m.index] + 0.001 * (m.p_y1 - m.s_y1)
    best = m.sort_values("cost").drop_duplicates("s_obs_id")
    pos = pd.Series(np.arange(len(out)), index=out["obs_id"].to_numpy())
    si = pos.loc[best.s_obs_id.to_numpy()].to_numpy()
    pi = pos.loc[best.p_obs_id.to_numpy()].to_numpy()
    for c in ("bot_px", "bot_py", "bot_dx", "bot_dy", "bot_dz", "bot_az", "bot_el", "cut_bottom"):
        out.iloc[si, out.columns.get_loc(c)] = out.iloc[pi][c].to_numpy()
    out.iloc[si, out.columns.get_loc("loc_src")] = "pole"
    out.iloc[si, out.columns.get_loc("pole_obs")] = out.iloc[pi]["obs_id"].to_numpy()
    return out


def select_obs(df: pd.DataFrame, thr: float = 0.0, owner_only: bool = True) -> pd.DataFrame:
    """Owner-view rows without the duplicate flag that score at least thr."""
    m = np.ones(len(df), bool)
    if owner_only and "owner" in df.columns:
        m &= df["owner"].fillna(False).astype(bool).to_numpy()
    if "dup" in df.columns:
        m &= ~df["dup"].fillna(False).astype(bool).to_numpy()
    m &= (df["score"] >= float(thr)).to_numpy()
    return df[m].copy()


# ----------------------------------------------------------------------------- range samples


def strip_samples(D: np.ndarray, view_idx: np.ndarray, bpx, bpy, bh, views, n: int = 5, frac: float = 0.12) -> np.ndarray:
    """(len, n) samples of the equirectangular range map D on a short vertical strip inside the mask,
    directly above each contact pixel: view pixel (bpx, bpy - 0.5 - t * box height), t = linspace(0.02, frac, n),
    nearest equirectangular pixel."""
    vals = np.full((len(view_idx), n), np.nan, np.float32)
    Hd, Wd = D.shape[:2]
    ts = np.linspace(0.02, frac, n)
    for vi in np.unique(view_idx):
        sub = np.nonzero(view_idx == vi)[0]
        px = np.repeat(bpx[sub][:, None], n, 1)
        py = bpy[sub][:, None] - 0.5 - ts[None, :] * bh[sub][:, None]
        u, v = ray_to_equirect_pixel(views[int(vi)].pixel_to_ray(px, py), Wd, Hd)
        vals[sub] = D[np.clip(np.rint(v).astype(int), 0, Hd - 1), np.mod(np.rint(u).astype(int), Wd)]
    return vals


def strip_median(vals: np.ndarray) -> np.ndarray:
    import warnings
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmedian(vals, axis=1).astype(float)


# ----------------------------------------------------------------------------- localization


def _contact_rays(df: pd.DataFrame, mounts: dict | None = None) -> np.ndarray:
    """Contact rays (N, 3): the centroid ray for surface classes, whose lowest mask row is the near border of
    a flat patch; the bottom ray otherwise."""
    rays = df[["bot_dx", "bot_dy", "bot_dz"]].to_numpy(float).copy()
    surf = row_mounts(df, mounts).eq("surface").to_numpy()
    if surf.any():
        rays[surf] = df.loc[surf, ["cen_dx", "cen_dy", "cen_dz"]].to_numpy(float)
    return rays


def _positions(df: pd.DataFrame, index: pd.DataFrame, R_lev: dict, rays: np.ndarray, ranges=None):
    """World positions of the contact rays. ranges None: intersect the leveled ray with the ground plane
    z = -h; otherwise move `ranges` metres along the leveled ray. Returns X, Y, horizontal range."""
    X = np.full(len(df), np.nan)
    Y = np.full(len(df), np.nan)
    rg = np.full(len(df), np.nan)
    for pid, idx in df.groupby("pano_id").indices.items():
        p = index.loc[pid]
        R = R_lev.get(pid, np.eye(3))
        d = rays[idx] @ np.asarray(R).T
        if ranges is None:
            valid = d[:, 2] < -1e-3
            t = np.where(valid, float(p.h) / np.maximum(-d[:, 2], 1e-9), np.nan)
            pl = d * t[:, None]
        else:
            pl = d * np.asarray(ranges)[idx][:, None]
        X[idx], Y[idx] = to_world(pl, p.heading, p.x, p.y)
        rg[idx] = np.hypot(pl[:, 0], pl[:, 1])
    return X, Y, rg


def localize(df: pd.DataFrame, index: pd.DataFrame, R_lev: dict, estimator: str = "hybrid",
             mounts: dict | None = None) -> pd.DataFrame:
    """Adds X, Y (world metres), rng (horizontal range, m), loc_ok, loc_est and bearing.

    ground  the leveled contact ray meets the plane z = -h (h from the index, 2.5 m by default)
    depth   the contact ray scaled to the MoGe-2 range sampled at detection time (column drng)
    hybrid  depth for DEPTH_CLASSES when the range is finite, ground otherwise
    Pole-mounted detections without a pole (loc_src 'none') get loc_ok False."""
    out = df.copy()
    if not len(df):
        for c in ("X", "Y", "rng"):
            out[c] = np.nan
        out["loc_ok"] = False
        out["loc_est"] = pd.Series(dtype=object)
        out["bearing"] = np.nan
        return out
    rays = _contact_rays(df, mounts)
    has_pole = out["loc_src"].ne("none").to_numpy() if "loc_src" in out.columns else np.ones(len(out), bool)
    gX, gY, grg = _positions(df, index, R_lev, rays)
    g_ok = np.isfinite(gX) & has_pole
    if estimator == "ground":
        X, Y, rg, ok, est = gX, gY, grg, g_ok, np.full(len(df), "ground", object)
    else:
        dX, dY, drg = _positions(df, index, R_lev, rays, df["drng"].to_numpy(float))
        d_ok = np.isfinite(dX) & has_pole
        if estimator == "depth":
            X, Y, rg, ok, est = dX, dY, drg, d_ok, np.full(len(df), "depth", object)
        elif estimator == "hybrid":
            use_d = df["label"].isin(DEPTH_CLASSES).to_numpy() & d_ok
            X, Y, rg = np.where(use_d, dX, gX), np.where(use_d, dY, gY), np.where(use_d, drg, grg)
            ok = np.where(use_d, d_ok, g_ok)
            est = np.where(use_d, "depth", "ground").astype(object)
        else:
            raise ValueError(f"unknown estimator {estimator!r}")
    out["X"], out["Y"], out["rng"], out["loc_ok"], out["loc_est"] = X, Y, rg, ok.astype(bool), est
    heading = index.loc[out["pano_id"], "heading"].to_numpy(float)
    out["bearing"] = (heading + out["bot_az"].astype(float).to_numpy()) % 360.0
    return out


def range_filter(obs: pd.DataFrame, R_max: float = 20.0, min_range: float = 1.0, drop_cut: bool = True) -> pd.DataFrame:
    """Localized rows with min_range <= rng <= R_max and, by default, a visible contact point."""
    m = obs["loc_ok"].to_numpy() & (obs["rng"] <= float(R_max)).to_numpy() & (obs["rng"] >= min_range).to_numpy()
    if drop_cut:
        m &= ~obs["cut_bottom"].astype(bool).to_numpy()
    return obs[m].copy()


# ----------------------------------------------------------------------------- clustering


def _weighted_median(v: np.ndarray, w: np.ndarray) -> float:
    o = np.argsort(v)
    c = np.cumsum(w[o])
    return float(v[o][np.searchsorted(c, 0.5 * c[-1])])


def cluster_greedy(xy: np.ndarray, w: np.ndarray, pano: np.ndarray, r_c: float) -> np.ndarray:
    """Observations in order of decreasing weight join the nearest cluster centre within r_c that has no
    observation from the same panorama, or start a new cluster; centres are weighted means. Returns labels."""
    n = len(xy)
    lab = np.full(n, -1, np.int64)
    cx, cy, cw, cpanos = [], [], [], []
    grid: dict = {}
    cell = max(r_c, 1e-3)
    for i in np.argsort(-w, kind="stable"):
        x, y = xy[i]
        gi, gj = int(np.floor(x / cell)), int(np.floor(y / cell))
        best, bd = -1, r_c
        for a in (gi - 1, gi, gi + 1):
            for b in (gj - 1, gj, gj + 1):
                for k in grid.get((a, b), ()):
                    dd = np.hypot(cx[k] - x, cy[k] - y)
                    if dd <= bd and pano[i] not in cpanos[k]:
                        best, bd = k, dd
        if best < 0:
            k = len(cx)
            cx.append(x), cy.append(y), cw.append(w[i]), cpanos.append({pano[i]})
            grid.setdefault((gi, gj), []).append(k)
        else:
            k = best
            ox, oy = cx[k], cy[k]
            tw = cw[k] + w[i]
            cx[k] = (cx[k] * cw[k] + x * w[i]) / tw
            cy[k] = (cy[k] * cw[k] + y * w[i]) / tw
            cw[k] = tw
            cpanos[k].add(pano[i])
            o = (int(np.floor(ox / cell)), int(np.floor(oy / cell)))
            nw = (int(np.floor(cx[k] / cell)), int(np.floor(cy[k] / cell)))
            if o != nw:
                grid[o].remove(k)
                grid.setdefault(nw, []).append(k)
        lab[i] = k
    return lab


def cluster_cc(xy: np.ndarray, r_c: float) -> np.ndarray:
    """Connected components of the graph joining observations closer than r_c (single linkage)."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    from scipy.spatial import cKDTree
    n = len(xy)
    pairs = cKDTree(xy).query_pairs(r_c, output_type="ndarray")
    g = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(n, n))
    return connected_components(g, directed=False)[1]


def aniso_position(x, y, rng, brg, x0, y0, sig_t=0.03, sig_r=0.12, sig0=0.5, iters=3):
    """Weighted least squares with per-observation Gaussians elongated along the viewing ray:
    sigma_along = 0.5 + 0.12 rng, sigma_across = 0.5 + 0.03 rng (metres); three iterations with Cauchy
    weights 1 / (1 + m^2 / 4), m the Mahalanobis distance, starting from (x0, y0)."""
    th = np.radians(brg)
    u = np.stack([np.sin(th), np.cos(th)], 1)
    v = np.stack([u[:, 1], -u[:, 0]], 1)
    sr, st = sig0 + sig_r * rng, sig0 + sig_t * rng
    q = np.stack([x, y], 1)
    p = np.array([x0, y0], float)
    for _ in range(iters):
        r = q - p
        m2 = (r * u).sum(1) ** 2 / sr**2 + (r * v).sum(1) ** 2 / st**2
        rw = 1.0 / (1.0 + m2 / 4.0)
        A = np.zeros((2, 2))
        bvec = np.zeros(2)
        for i in range(len(q)):
            Ii = np.outer(u[i], u[i]) / sr[i] ** 2 + np.outer(v[i], v[i]) / st[i] ** 2
            A += rw[i] * Ii
            bvec += rw[i] * Ii @ q[i]
        try:
            p = np.linalg.solve(A, bvec)
        except np.linalg.LinAlgError:
            break
    return float(p[0]), float(p[1])


ASSET_COLUMNS = ["asset_id", "label", "x", "y", "support", "n_obs", "score_mean", "score_max", "conf",
                 "rng_min", "rng_med", "spread", "best_obs", "best_pano"]


def fuse(obs: pd.DataFrame, r_c: float = 2.0, method: str = "greedy", pos: str = "median", w_floor: float = 3.0,
         min_support: int = 1):
    """Cluster localized observations of each class into assets.

    weight      w = 1 / max(rng, w_floor)
    position    weighted median of x and y; with pos 'aniso' and support >= 2, aniso_position from there
    support     number of distinct panoramas in the cluster
    conf        1 - prod over panoramas of (1 - min(best score of the panorama, 0.999))
    Returns (assets, asset label of each observation aligned with obs, -1 = below min_support)."""
    rows = []
    labels = pd.Series(-1, index=obs.index, dtype=np.int64)
    aid = 0
    for cls, g in obs.groupby("label", sort=True):
        if not len(g):
            continue
        xy = g[["X", "Y"]].to_numpy(float)
        w = 1.0 / np.maximum(g["rng"].to_numpy(float), w_floor)
        pano = g["pano_id"].to_numpy()
        lab = cluster_greedy(xy, w, pano, r_c) if method == "greedy" else cluster_cc(xy, r_c)
        sc = g["score"].to_numpy(float)
        rg = g["rng"].to_numpy(float)
        br = g["bearing"].to_numpy(float) if "bearing" in g.columns else None
        order = np.argsort(lab, kind="stable")
        for members in np.split(order, np.flatnonzero(np.diff(lab[order])) + 1):
            sup = len(set(pano[members]))
            if sup < min_support:
                continue
            ww = w[members]
            x = _weighted_median(xy[members, 0], ww)
            y = _weighted_median(xy[members, 1], ww)
            if pos == "aniso" and br is not None and sup >= 2:
                x, y = aniso_position(xy[members, 0], xy[members, 1], rg[members], br[members], x, y)
            s = sc[members]
            best = members[np.argmin(rg[members])]
            conf = 1.0 - np.prod(1.0 - np.clip(pd.Series(s).groupby(pano[members]).max().to_numpy(), 0, 0.999))
            rows.append(dict(asset_id=aid, label=cls, x=x, y=y, support=sup, n_obs=len(members),
                             score_mean=float(s.mean()), score_max=float(s.max()), conf=float(conf),
                             rng_min=float(rg[members].min()), rng_med=float(np.median(rg[members])),
                             spread=float(np.median(np.hypot(xy[members, 0] - x, xy[members, 1] - y))),
                             best_obs=int(g["obs_id"].to_numpy()[best]), best_pano=pano[best]))
            labels.loc[g.index[members]] = aid
            aid += 1
    return pd.DataFrame(rows, columns=ASSET_COLUMNS), labels


def class_params(params: dict, cls: str) -> dict:
    """Fusion parameters of one class: defaults, then the file's 'default', then its 'per_class' entry."""
    p = dict(DEFAULT_PARAMS, conf_thr=0.0)
    p.update(params.get("default", {}))
    p.update(params.get("per_class", {}).get(cls, {}))
    return p


def fuse_with_params(obs_all: pd.DataFrame, params: dict, classes):
    """Per class: score >= score_thr, range_filter(R_max, min_range), fuse(r_c, method, pos, w_floor), then
    keep assets with support >= min_support and conf >= conf_thr. Returns (observations, assets)."""
    keep_obs, assets_l, aid = [], [], 0
    for cls in classes:
        p = class_params(params, cls)
        o = obs_all[(obs_all["label"] == cls) & (obs_all["score"] >= p.get("score_thr", 0.0))]
        o = range_filter(o, p["R_max"], p.get("min_range", 1.0))
        if not len(o):
            continue
        a, lab = fuse(o, r_c=p["r_c"], method=p["method"], pos=p["pos"], w_floor=p["w_floor"], min_support=1)
        a = a[(a["support"] >= p["min_support"]) & (a["conf"] >= p["conf_thr"])].copy()
        o = o.assign(asset=lab.to_numpy())
        o.loc[~o["asset"].isin(a["asset_id"]), "asset"] = -1
        a["asset_id"] += aid
        o.loc[o["asset"] >= 0, "asset"] += aid
        aid = int(a["asset_id"].max()) + 1 if len(a) else aid
        keep_obs.append(o)
        assets_l.append(a)
    obs = pd.concat(keep_obs) if keep_obs else obs_all.iloc[:0]
    assets = pd.concat(assets_l, ignore_index=True) if assets_l else pd.DataFrame(columns=ASSET_COLUMNS)
    return obs, assets
