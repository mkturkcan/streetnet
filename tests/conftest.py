"""A small synthetic street: three cameras on a straight road and objects at known positions.

Detections are made by projecting each object's ground contact into the cameras' detection views, so the
map stage can be tested end to end without images or models."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from streetnet.geometry import azimuth_of_ray, detection_views, elevation_of_ray, owner_view

LAT0, LON0 = 40.80, -73.95


def contact_detection(cam_xy, heading, obj_xy, label, mount, views, pano_id, h=2.5, score=0.9, height_px=200):
    """The detection of an object standing on the ground at obj_xy, seen from a level camera."""
    dx, dy = obj_xy[0] - cam_xy[0], obj_xy[1] - cam_xy[1]
    a = np.radians(heading)
    # world (east, north) -> cam (right, forward) for compass heading a
    right = dx * np.cos(a) - dy * np.sin(a)
    fwd = dx * np.sin(a) + dy * np.cos(a)
    ray = np.array([right, fwd, -h])
    ray /= np.linalg.norm(ray)
    az = float(azimuth_of_ray(ray))
    vi = owner_view(az, len(views))
    view = views[vi]
    px, py, _ = view.ray_to_pixel(ray)
    px, py = float(px), float(py)
    rec = dict(pano_id=pano_id, model="sam3", label=label, raw_label=label, score=score, view=vi,
               x0=int(px - 20), y0=int(py - height_px), x1=int(px + 20), y1=int(py - 0.5), area_px=4000,
               bot_px=px, bot_py=py, top_px=px, top_py=py - height_px,
               bot_dx=ray[0], bot_dy=ray[1], bot_dz=ray[2], top_dx=ray[0], top_dy=ray[1], top_dz=0.5,
               cen_dx=ray[0], cen_dy=ray[1], cen_dz=ray[2], bot_az=az, bot_el=float(elevation_of_ray(ray)),
               cut_bottom=False, cut_side=False, owner=True, mount=mount, dup=False, prompt_rank=0,
               sign_filter="", drng=float(np.hypot(np.hypot(dx, dy), h)))
    return rec


@pytest.fixture
def street(tmp_path):
    """Index, detection folder and object list of the synthetic street (UTM zone 18N)."""
    from streetnet.index import ll_to_xy, xy_to_ll
    x0, y0 = ll_to_xy(LON0, LAT0, 32618)
    x0, y0 = float(x0), float(y0)
    cams = {"pA": (x0, y0), "pB": (x0, y0 + 10.0), "pC": (x0, y0 + 20.0)}
    objects = [("fire_hydrant", "ground", (x0 + 4.0, y0 + 6.0)), ("tree", "ground", (x0 - 5.0, y0 + 12.0)),
               ("fire_hydrant", "ground", (x0 + 4.0, y0 + 30.0))]
    views = detection_views(4)
    lon, lat = xy_to_ll(np.array([c[0] for c in cams.values()]), np.array([c[1] for c in cams.values()]), 32618)
    index = pd.DataFrame({"pano_id": list(cams), "lat": lat, "lon": lon, "heading": 0.0, "pitch": 0.0, "roll": 0.0})
    index.to_parquet(tmp_path / "index.parquet", index=False)
    det = tmp_path / "run" / "det"
    det.mkdir(parents=True)
    for pid, cxy in cams.items():
        rows = [contact_detection(cxy, 0.0, oxy, lab, mount, views, pid) for lab, mount, oxy in objects
                if np.hypot(oxy[0] - cxy[0], oxy[1] - cxy[1]) < 20]
        pd.DataFrame(rows).to_parquet(det / f"{pid}.parquet", index=False)
    return dict(index=tmp_path / "index.parquet", run=tmp_path / "run", objects=objects, cams=cams)
