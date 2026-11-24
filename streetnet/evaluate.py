"""Evaluation of a map against ground-truth points.

Region and ignore rules:
    region       the union of disks of radius region_r (15 m) around the evaluation panoramas
    matching     one-to-one within radius R (2, 3 and 5 m), between the predictions of a class and all
                 ground truth of that class, also outside the region: within each connected group of candidate
                 pairs, the largest number of pairs and among those the smallest total distance (Hungarian
                 assignment with a prohibitive cost for pairs farther than R)
    TP           a matched pair whose ground-truth point lies inside the region
    FP           an unmatched prediction inside the region
    FN           an unmatched ground-truth point inside the region
    ignored      matched pairs with the ground truth outside the region, unmatched predictions outside it
precision = TP / (TP + FP), recall = TP / (TP + FN), F1 = 2 P R / (P + R); F1 is NaN for a class without predictions
or without ground truth in the region and 0 with both but no TP. '_all' sums TP, FP and FN over the classes;
'_macro_f1' averages F1 over the classes with ground truth in the region and a defined F1.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree

RADII = (2.0, 3.0, 5.0)


def in_region(xy: np.ndarray, centers: np.ndarray, radius: float) -> np.ndarray:
    if len(xy) == 0 or len(centers) == 0:
        return np.zeros(len(xy), bool)
    d, _ = cKDTree(centers).query(np.asarray(xy, float), k=1)
    return d <= radius


def match(pred_xy, gt_xy, R: float):
    """One-to-one matching within radius R. Returns (prediction index, ground-truth index, distance)."""
    pred_xy = np.asarray(pred_xy, float).reshape(-1, 2)
    gt_xy = np.asarray(gt_xy, float).reshape(-1, 2)
    empty = (np.zeros(0, int), np.zeros(0, int), np.zeros(0))
    if len(pred_xy) == 0 or len(gt_xy) == 0:
        return empty
    sdm = cKDTree(pred_xy).sparse_distance_matrix(cKDTree(gt_xy), R, output_type="ndarray")
    if len(sdm) == 0:
        return empty
    pi, gi, dd = sdm["i"].astype(int), sdm["j"].astype(int), sdm["v"].astype(float)
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    P, G = len(pred_xy), len(gt_xy)
    _, comp = connected_components(coo_matrix((np.ones(len(pi)), (pi, P + gi)), shape=(P + G, P + G)), directed=False)
    out, big = [], 1e6
    for c in np.unique(comp[pi]):
        sel = comp[pi] == c
        up, ug = np.unique(pi[sel]), np.unique(gi[sel])
        if len(up) == 1 and len(ug) == 1:
            out.append((up[0], ug[0], dd[sel][0]))
            continue
        C = np.full((len(up), len(ug)), big)
        C[np.searchsorted(up, pi[sel]), np.searchsorted(ug, gi[sel])] = dd[sel]
        r, k = linear_sum_assignment(C)
        out.extend((up[a], ug[b], C[a, b]) for a, b in zip(r, k) if C[a, b] < big)
    if not out:
        return empty
    a = np.array(out)
    return a[:, 0].astype(int), a[:, 1].astype(int), a[:, 2].astype(float)


def prf(tp: int, fp: int, fn: int) -> dict:
    p = tp / (tp + fp) if tp + fp else float("nan")
    r = tp / (tp + fn) if tp + fn else float("nan")
    if (tp + fp) and (tp + fn):
        f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    else:
        f = float("nan")
    return dict(tp=int(tp), fp=int(fp), fn=int(fn), precision=p, recall=r, f1=f)


def evaluate(assets: pd.DataFrame, gt: pd.DataFrame, centers: np.ndarray, classes=None, radii=RADII,
             region_r: float = 15.0) -> dict:
    """assets: label, x, y; gt: gt_key, x, y (same metric CRS); centers: (N, 2) evaluation panoramas.
    Returns {"R2": {class: metrics, "_all": ..., "_macro_f1": ...}, ..., "err": localization error}."""
    classes = sorted(set(gt["gt_key"]) & set(assets["label"])) if classes is None else classes
    res = {f"R{r:g}": {} for r in radii}
    err = {}
    for cls in classes:
        pxy = assets.loc[assets["label"] == cls, ["x", "y"]].to_numpy(float)
        gxy = gt.loc[gt["gt_key"] == cls, ["x", "y"]].to_numpy(float)
        p_in, g_in = in_region(pxy, centers, region_r), in_region(gxy, centers, region_r)
        for r in radii:
            pi, gi, d = match(pxy, gxy, r)
            keep = g_in[gi] if len(gi) else np.zeros(0, bool)
            matched_p = np.zeros(len(pxy), bool)
            matched_p[pi] = True
            matched_g = np.zeros(len(gxy), bool)
            matched_g[gi] = True
            m = prf(int(keep.sum()), int((~matched_p & p_in).sum()), int((~matched_g & g_in).sum()))
            res[f"R{r:g}"][cls] = dict(m, n_pred=int(p_in.sum()), n_gt=int(g_in.sum()))
            if r == max(radii):
                dk = d[keep]
                err[cls] = dict(n=int(len(dk)), median=float(np.median(dk)) if len(dk) else float("nan"),
                                p90=float(np.percentile(dk, 90)) if len(dk) else float("nan"))
    for r in radii:
        k = f"R{r:g}"
        tot = {s: sum(v[s] for v in res[k].values()) for s in ("tp", "fp", "fn", "n_pred", "n_gt")}
        res[k]["_all"] = dict(prf(tot["tp"], tot["fp"], tot["fn"]), n_pred=tot["n_pred"], n_gt=tot["n_gt"])
        f1s = [v["f1"] for c, v in res[k].items() if c != "_all" and v["n_gt"] > 0 and np.isfinite(v["f1"])]
        res[k]["_macro_f1"] = float(np.mean(f1s)) if f1s else float("nan")
    res["err"] = err
    return res


def format_table(res: dict, R: str = "R3") -> str:
    lines = [f"{'class':<18} {'pred':>7} {'gt':>7} {'P':>6} {'R':>6} {'F1':>6}"]
    for c, v in res[R].items():
        if c.startswith("_"):
            continue
        lines.append(f"{c:<18} {v['n_pred']:>7} {v['n_gt']:>7} {v['precision']:>6.3f} {v['recall']:>6.3f} {v['f1']:>6.3f}")
    a = res[R]["_all"]
    lines.append(f"{'all':<18} {a['n_pred']:>7} {a['n_gt']:>7} {a['precision']:>6.3f} {a['recall']:>6.3f} {a['f1']:>6.3f}")
    return "\n".join(lines)
