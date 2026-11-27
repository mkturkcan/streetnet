"""Sign filter: traffic-sign detections that are signal heads or advertisements on vehicles.

For each traffic_sign or bus_stop_sign detection, with the panorama's Mask2Former label map L (2048 x 1024
equirectangular) and the signal detections in the same view:
    mask pixels          up to 400 pixels of the detection mask (seeded random sample), view -> L
    f_light              fraction labeled Traffic Light
    f_sign               fraction labeled Traffic Sign (Front), (Back), or Traffic Sign Frame
    f_vehicle            fraction labeled as a vehicle (Bus, Car, Caravan, Motorcycle, On Rails, Other
                         Vehicle, Trailer, Truck, or Wheeled Slow)
    f_vehicle_adj        vehicle fraction of a 12 x 3 grid in the band directly below the box:
                         columns x0..x1, rows y1 + 1 .. y1 + max(2, 0.15 h)
    signal_overlap       max over traffic_signal / pedestrian_signal detections (score >= 0.3) in the same
                         view of |box_sign intersect box_signal| / |box_sign|
    signal_overlap_win   the same over signal detections that score higher than the sign detection
    aspect, top_el       box width / height; elevation of the top point (degrees)
Decision:
    signal head    f_light >= 0.4, or signal_overlap_win >= 0.5 and f_sign < 0.3
    vehicle mount  f_vehicle >= 0.3, or f_vehicle_adj >= 0.7 and aspect >= 1.8 and top_el < 2 degrees
Vehicle-mounted detections are dropped. A signal head is dropped when a signal detection overlaps it
(signal_overlap >= 0.3) and relabeled traffic_signal otherwise.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .geometry import ray_to_equirect_pixel
from .instances import decode_rle
from .models.mask2former import TRAFFIC_LIGHT, TRAFFIC_SIGN, VEHICLE

SIGN_CLASSES = ("traffic_sign", "bus_stop_sign")
SIGNAL_CLASSES = ("traffic_signal", "pedestrian_signal")
LIGHT_THR, OVERLAP_THR, SIGN_MAX = 0.4, 0.5, 0.3
VEH_THR, VEH_ADJ_THR, ROOF_ASPECT, ROOF_TOP_EL = 0.3, 0.7, 1.8, 2.0


def _labels_at(sem, view, px, py):
    u, w = ray_to_equirect_pixel(view.pixel_to_ray(px, py), sem.shape[1], sem.shape[0])
    return sem[np.clip(np.rint(w).astype(int), 0, sem.shape[0] - 1), np.mod(np.rint(u).astype(int), sem.shape[1])]


def sign_features(det: pd.DataFrame, sem: np.ndarray, views) -> pd.DataFrame:
    """One row per sign detection (index aligned with det) with the features and the two decisions."""
    rng = np.random.default_rng(0)
    signs = det[det.label.isin(SIGN_CLASSES)]
    sig = det[det.label.isin(SIGNAL_CLASSES) & (det.score >= 0.3)]
    rows = []
    for i, r in signs.iterrows():
        view = views[int(r.view)]
        m = decode_rle(r.rle, view.h, view.w)
        ys, xs = np.nonzero(m)
        if len(ys) == 0:
            continue
        if len(ys) > 400:
            k = rng.choice(len(ys), 400, replace=False)
            ys, xs = ys[k], xs[k]
        lab = _labels_at(sem, view, xs.astype(float), ys.astype(float))
        h = r.y1 - r.y0 + 1
        BX, BY = np.meshgrid(np.linspace(r.x0, r.x1, 12), np.linspace(r.y1 + 1, min(view.h - 1, r.y1 + max(2, 0.15 * h)), 3))
        below = _labels_at(sem, view, BX.ravel(), BY.ravel())
        ov, ov_win = 0.0, 0.0
        s = sig[sig.view == r.view]
        if len(s):
            ix = np.clip(np.minimum(s.x1, r.x1) - np.maximum(s.x0, r.x0) + 1, 0, None)
            iy = np.clip(np.minimum(s.y1, r.y1) - np.maximum(s.y0, r.y0) + 1, 0, None)
            o = (ix * iy).to_numpy() / ((r.x1 - r.x0 + 1) * (r.y1 - r.y0 + 1))
            ov = float(o.max())
            win = s.score.to_numpy() > r.score
            ov_win = float(o[win].max()) if win.any() else 0.0
        f = dict(f_light=float(np.isin(lab, list(TRAFFIC_LIGHT)).mean()),
                 f_sign=float(np.isin(lab, list(TRAFFIC_SIGN)).mean()),
                 f_vehicle=float(np.isin(lab, list(VEHICLE)).mean()),
                 f_vehicle_adj=float(np.isin(below, list(VEHICLE)).mean()),
                 signal_overlap=ov, signal_overlap_win=ov_win,
                 aspect=float((r.x1 - r.x0 + 1) / (r.y1 - r.y0 + 1)),
                 top_el=float(np.degrees(np.arcsin(np.clip(r.top_dz, -1, 1)))))
        f["is_signal"] = bool(f["f_light"] >= LIGHT_THR or (f["signal_overlap_win"] >= OVERLAP_THR and f["f_sign"] < SIGN_MAX))
        f["is_vehicle"] = bool(f["f_vehicle"] >= VEH_THR or (f["f_vehicle_adj"] >= VEH_ADJ_THR and f["aspect"] >= ROOF_ASPECT
                                                             and f["top_el"] < ROOF_TOP_EL))
        rows.append(dict(idx=i, **f))
    return pd.DataFrame(rows).set_index("idx") if rows else pd.DataFrame()


def apply_sign_filter(det: pd.DataFrame, sem: np.ndarray, views) -> pd.DataFrame:
    """Adds column sign_filter ('' or 'signal_relabel') and removes the dropped rows."""
    det = det.copy()
    det["sign_filter"] = ""
    f = sign_features(det, sem, views)
    if not len(f):
        return det
    det.loc[f.index[f.is_vehicle], "sign_filter"] = "vehicle"
    sig = f[f.is_signal & ~f.is_vehicle]
    det.loc[sig.index[sig.signal_overlap >= 0.3], "sign_filter"] = "signal_dup"
    rl = sig.index[sig.signal_overlap < 0.3]
    det.loc[rl, "sign_filter"] = "signal_relabel"
    det.loc[rl, "label"] = "traffic_signal"
    return det[det.sign_filter.isin(["", "signal_relabel"])]
