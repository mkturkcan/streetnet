"""Detection records: one row per detected object, the same schema for all models.

Columns
-------
pano_id, model, label (class key), raw_label (prompt or native class name), score
view                      index into the view set
x0, y0, x1, y1, area_px   box and mask area in view pixels
bot_px, bot_py            ground contact pixel: median column of the mask pixels in rows y1 - k .. y1, bottom edge
                          of row y1, with k = max(1, round(0.03 (y1 - y0 + 1)))
top_px, top_py            top pixel: median column of the mask pixels in rows y0 .. y0 + k, top edge of row y0
bot_d*, top_d*, cen_d*    cam-frame unit rays through the contact pixel, the top pixel and the mask centroid
bot_az, bot_el            azimuth (degrees clockwise from forward) and elevation of the contact ray
cut_bottom                the mask touches the bottom edge of the view (contact point outside the view)
cut_side                  the mask touches a side edge (object truncated)
owner                     the contact azimuth lies in this view's central sector (see geometry.owner_view)
rle                       COCO run-length encoding of the mask at view resolution
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .geometry import View, azimuth_of_ray, elevation_of_ray, owner_view

COLUMNS = ["pano_id", "model", "label", "raw_label", "score", "view", "x0", "y0", "x1", "y1",
           "area_px", "bot_px", "bot_py", "top_px", "top_py", "bot_dx", "bot_dy", "bot_dz",
           "top_dx", "top_dy", "top_dz", "cen_dx", "cen_dy", "cen_dz", "bot_az", "bot_el",
           "cut_bottom", "cut_side", "owner", "rle"]


def mask_record(mask: np.ndarray, view: View, view_idx: int, n_views: int, with_rle: bool = True):
    """Geometry of one boolean mask (h, w) in `view`; None for an empty mask."""
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return None
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    bh = y1 - y0 + 1
    band = ys >= y1 - max(1, int(round(0.03 * bh)))
    bot_px = float(np.median(xs[band]))
    bot_py = float(y1) + 0.5                       # bottom edge of the lowest row
    tband = ys <= y0 + max(1, int(round(0.03 * bh)))
    top_px = float(np.median(xs[tband]))
    top_py = float(y0) - 0.5                       # top edge of the highest row
    rays = view.pixel_to_ray(np.array([bot_px, top_px, xs.mean()]), np.array([bot_py, top_py, ys.mean()]))
    az = float(azimuth_of_ray(rays[0]))
    rec = dict(view=view_idx, x0=int(x0), y0=int(y0), x1=int(x1), y1=int(y1), area_px=int(len(ys)),
               bot_px=bot_px, bot_py=bot_py, top_px=top_px, top_py=top_py,
               bot_dx=rays[0, 0], bot_dy=rays[0, 1], bot_dz=rays[0, 2],
               top_dx=rays[1, 0], top_dy=rays[1, 1], top_dz=rays[1, 2],
               cen_dx=rays[2, 0], cen_dy=rays[2, 1], cen_dz=rays[2, 2],
               bot_az=az, bot_el=float(elevation_of_ray(rays[0])),
               cut_bottom=bool(y1 >= view.h - 2), cut_side=bool(x0 <= 1 or x1 >= view.w - 2),
               owner=bool(owner_view(az, n_views) == view_idx))
    if with_rle:
        from pycocotools import mask as mask_utils
        rec["rle"] = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))["counts"].decode("ascii")
    return rec


def decode_rle(rle: str, h: int, w: int) -> np.ndarray:
    from pycocotools import mask as mask_utils
    return mask_utils.decode({"size": [h, w], "counts": rle.encode("ascii")}).astype(bool)


def to_frame(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    for c in COLUMNS:
        if c not in df.columns:
            df[c] = None
    return df[COLUMNS]


def empty_frame() -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype=object) for c in COLUMNS})
