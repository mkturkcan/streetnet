#!/usr/bin/env python
"""Images of one pilot panorama for the hero and overview slides (tools/figures/slides/hero.html, overview.html).

    python tools/figures/pilot_panels.py --pano <pano>.jpg --det det/final/<pano>.parquet --depth depth/<pano>.npy \
        --index panos.parquet --assets all_assets.parquet --obs all_obs.parquet --streets street_centerlines.geojson

Inputs are outputs of the Upper Manhattan pilot study (eight 90 degree views, yaw 45 k, pitch -10; the views with
yaw 0, 90, 180 and 270 coincide with the four views of streetnet detect): the panorama, its detection table with
run-length-encoded masks, its MoGe-2 range map (equirectangular, metres), the panorama index and the fused map
(assets and observations). The images show the detections of the 22 published classes in their owner view with a
score of at least 0.35 (duplicates left out); the top view shows the observations that were fused into assets.

Writes docs/assets/img/hero.jpg (a perspective view 100 degrees wide with the masks, used by the hero slide and the
project page) and, to tools/figures/slides/img/:
    ov-pano.jpg           the panorama between 35 degrees above and 35 degrees below the horizon
    ov-view.jpg           one detection view with masks and ground contact pixels
    ov-range.jpg          the same view coloured by MoGe-2 range, with the contact pixels
    ov-map-{light,dark}.png  top view: cameras, rays of the observations and the fused assets
    ov.json               pixel positions of the labels drawn by the slides
The panorama id is used to select rows and is not written to any output.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageFilter

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from streetnet.fusion import strip_median, strip_samples  # noqa: E402
from streetnet.geometry import View, detection_views  # noqa: E402
from streetnet.instances import decode_rle  # noqa: E402

CLASSES = json.loads((ROOT / "streetnet" / "configs" / "classes.json").read_text())
IMG = ROOT / "tools" / "figures" / "slides" / "img"
EPSG = 32618
PHOTO = ["#3987e5", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#0ca30c", "#9085e9", "#e34948"]   # group hues on photos
RELABEL = {"pedestrian_signal": "traffic_signal"}
Image.MAX_IMAGE_PIXELS = None


def hex_rgb(h):
    return np.array([int(h[i:i + 2], 16) for i in (1, 3, 5)], float)


def sample_equirect(eq: np.ndarray, rays: np.ndarray) -> np.ndarray:
    """Bilinear sample of an equirectangular image (H, W, C) along cam-frame rays (..., 3)."""
    H, W = eq.shape[:2]
    lam = np.arctan2(rays[..., 0], rays[..., 1])
    phi = np.arcsin(np.clip(rays[..., 2] / np.linalg.norm(rays, axis=-1), -1, 1))
    u = (lam / (2 * np.pi) + 0.5) * W - 0.5
    v = (0.5 - phi / np.pi) * H - 0.5
    u0, v0 = np.floor(u).astype(int), np.floor(v).astype(int)
    fu, fv = (u - u0)[..., None], (v - v0)[..., None]
    v0c, v1c = np.clip(v0, 0, H - 1), np.clip(v0 + 1, 0, H - 1)
    u0w, u1w = np.mod(u0, W), np.mod(u0 + 1, W)
    out = (eq[v0c, u0w] * (1 - fu) * (1 - fv) + eq[v0c, u1w] * fu * (1 - fv)
           + eq[v1c, u0w] * (1 - fu) * fv + eq[v1c, u1w] * fu * fv)
    return out


def view_rays(v: View) -> np.ndarray:
    ys, xs = np.mgrid[0:v.h, 0:v.w]
    return v.pixel_to_ray(xs, ys)


def masks_in(target: View, det: pd.DataFrame, views: tuple[View, ...], rays=None):
    """Masks of detections (rows of the pilot table) resampled into another view: list of (bool[h, w], row)."""
    rays = view_rays(target) if rays is None else rays     # target may be None when rays are given
    out = []
    coords = {}
    for _, r in det.iterrows():
        k = int(r.view)
        if k not in coords:
            px, py, front = views[k].ray_to_pixel(rays)
            coords[k] = (np.rint(px).astype(int), np.rint(py).astype(int),
                         front & (px > -0.5) & (px < views[k].w - 0.5) & (py > -0.5) & (py < views[k].h - 0.5))
        px, py, ok = coords[k]
        m = decode_rle(r.rle, views[k].h, views[k].w)
        hit = np.zeros(ok.shape, bool)
        hit[ok] = m[np.clip(py[ok], 0, views[k].h - 1), np.clip(px[ok], 0, views[k].w - 1)] > 0
        if hit.any():
            out.append((hit, r))
    return out


def paint(img: np.ndarray, masks, colors, alpha=0.36, edge_px=3) -> np.ndarray:
    """Fill each mask with its group colour and draw its outline."""
    out = img.astype(float).copy()
    for m, r in sorted(masks, key=lambda t: -t[0].sum()):
        c = hex_rgb(colors[CLASSES["classes"][r.label]["group"]])
        out[m] = out[m] * (1 - alpha) + c * alpha
        mi = Image.fromarray((m * 255).astype(np.uint8))
        er = np.asarray(mi.filter(ImageFilter.MinFilter(2 * edge_px + 1))) > 127
        edge = m & ~er
        out[edge] = c
    return np.clip(out, 0, 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for k in ("pano", "det", "depth", "index", "assets", "obs", "streets"):
        ap.add_argument(f"--{k}", required=True)
    ap.add_argument("--hero-yaw", type=float, default=162.0)
    ap.add_argument("--hero-pitch", type=float, default=-3.0)
    ap.add_argument("--view", type=int, default=4, help="pilot view of the overview (4: yaw 180, view 2 of streetnet detect)")
    a = ap.parse_args()
    IMG.mkdir(parents=True, exist_ok=True)
    pid = Path(a.det).stem

    pano = np.asarray(Image.open(a.pano).convert("RGB"), dtype=np.float32)
    views = detection_views(8)
    det = pd.read_parquet(a.det)
    det["label"] = det.label.replace(RELABEL)
    obs = pd.read_parquet(a.obs)
    obs["label"] = obs.label.replace(RELABEL)
    mine = obs[(obs.pano_id == pid) & obs.label.isin(CLASSES["classes"])]
    key = ["view", "x0", "y0", "x1", "y1"]
    fused = det.merge(mine[key + ["asset", "rng", "X", "Y"]], on=key, how="inner")
    fused = fused[fused.label.isin(CLASSES["classes"])]
    shown = det[~det.dup & det.owner & det.label.isin(CLASSES["classes"]) & (det.score >= 0.35)]
    print(f"[pilot] {len(shown)} detections shown, {len(fused)} fused, of {len(det)} rows")

    # ---- hero: 2400 x 1000, 100 degrees wide
    hero = View(yaw=a.hero_yaw, pitch=a.hero_pitch, fov=100.0, w=2400, h=1000)
    rays = view_rays(hero)
    img = sample_equirect(pano, rays)
    img = paint(img, masks_in(hero, shown, views, rays), PHOTO, alpha=0.34, edge_px=3)
    (ROOT / "docs" / "assets" / "img").mkdir(parents=True, exist_ok=True)
    Image.fromarray(img).save(ROOT / "docs" / "assets" / "img" / "hero.jpg", quality=84, subsampling=2, optimize=True,
                              progressive=True)
    print("[pilot] hero.jpg")

    # ---- panorama strip from yaw -45 to 315 degrees (the four views of streetnet detect, left to right),
    #      33 degrees above to 50 degrees below the horizon
    H, W = 600, 2592
    lam = np.radians(-45) + (np.arange(W) + 0.5) / W * 2 * np.pi
    phi = np.radians(33) - (np.arange(H) + 0.5) / H * np.radians(83)
    L, P = np.meshgrid(lam, phi)
    prays = np.stack([np.cos(P) * np.sin(L), np.cos(P) * np.cos(L), np.sin(P)], -1)
    strip = sample_equirect(pano, prays)
    strip = paint(strip, masks_in(None, shown, views, prays), PHOTO, alpha=0.34, edge_px=2)
    Image.fromarray(strip).save(IMG / "ov-pano.jpg", quality=86, subsampling=0, optimize=True, progressive=True)
    labels = {"pano": {"w": W, "h": H, "yaw0": -45}}

    # ---- one detection view with the masks that project into it and the ground contact pixels
    v = views[a.view]
    vrays = view_rays(v)
    vimg = sample_equirect(pano, vrays)
    vimg = paint(vimg, masks_in(v, shown, views, vrays), PHOTO, alpha=0.34, edge_px=2)
    top, bottom = 150, 1008                                 # crop the sky: rows 150 to 1008
    Image.fromarray(vimg[top:bottom]).save(IMG / "ov-view.jpg", quality=88, subsampling=0, optimize=True)

    # ---- MoGe-2 range of the same view, with 5, 10, 20 and 40 m contours
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    D = np.load(a.depth).astype(np.float32)
    rng = sample_equirect(np.nan_to_num(D, nan=200.0)[..., None], vrays)[..., 0][top:bottom]
    t = np.clip((np.log(np.clip(rng, 2, 80)) - np.log(2)) / (np.log(80) - np.log(2)), 0, 1)
    stops = np.array([hex_rgb(c) for c in ("#0d366b", "#184f95", "#256abf", "#3987e5", "#6da7ec", "#9ec5f4", "#cde2fb", "#eef5fd")])
    tt = t * (len(stops) - 1)
    i = np.minimum(tt.astype(int), len(stops) - 2)
    f = (tt - i)[..., None]
    rimg = (stops[i] * (1 - f) + stops[i + 1] * f) / 255
    h, w = rimg.shape[:2]
    fig = plt.figure(figsize=(w / 100, h / 100), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(rimg, interpolation="bilinear")
    ax.contour(np.where(rng < 150, rng, np.nan), levels=[5, 10, 20, 40], colors="white", linewidths=1.1, alpha=0.7)
    ax.set_axis_off()
    fig.savefig(IMG / "ov-range.jpg", dpi=100, pil_kwargs={"quality": 90, "subsampling": 0})
    plt.close(fig)

    # contact pixels of the shown detections in this view; range = median of the strip samples in their own view
    drng = strip_median(strip_samples(D, shown.view.to_numpy(), shown.bot_px.to_numpy(float), shown.bot_py.to_numpy(float),
                                      (shown.y1 - shown.y0 + 1).to_numpy(float), views))
    cpx, cpy, front = v.ray_to_pixel(shown[["bot_dx", "bot_dy", "bot_dz"]].to_numpy(float))
    pts = []
    for (_, r), d, x, y, fr in zip(shown.iterrows(), drng, cpx, cpy, front):
        if not fr or r.cut_bottom or not (8 < x < 1000 and top + 4 < y < bottom - 4) or not np.isfinite(d):
            continue
        pts.append({"x": round(float(x), 1), "y": round(float(y) - top, 1), "g": CLASSES["classes"][r.label]["group"],
                    "drng": round(float(d), 1), "label": r.label})
    labels["view"] = {"w": 1008, "h": bottom - top, "contacts": pts}
    print(f"[pilot] view {a.view}: {len(pts)} contact pixels")

    # ---- top view: cameras, rays and assets
    I = pd.read_parquet(a.index)
    cam = I.loc[pid]
    A = pd.read_parquet(a.assets).set_index("asset_id")
    seen = sorted(set(mine.asset))
    rays_all = obs[obs.asset.isin(seen) & obs.label.isin(CLASSES["classes"])].copy()
    rays_all = rays_all.join(I[["x", "y"]], on="pano_id")
    half = 30.0
    cx, cy = float(cam.x), float(cam.y)
    box = (cx - half * 1.33, cy - half, cx + half * 1.33, cy + half)
    inb = lambda x, y: (x > box[0]) & (x < box[2]) & (y > box[1]) & (y < box[3])  # noqa: E731
    sel = A.loc[[s for s in seen if s in A.index]]
    sel = sel[inb(sel.x, sel.y) & sel.label.isin(CLASSES["classes"])]
    rays_all = rays_all[rays_all.asset.isin(sel.index)]
    from map_figure import read_lines  # noqa: E402
    streets, kinds, _ = read_lines(a.streets)
    near = [ln for ln in streets if (ln[:, 0].max() > box[0] - 60) and (ln[:, 0].min() < box[2] + 60)
            and (ln[:, 1].max() > box[1] - 60) and (ln[:, 1].min() < box[3] + 60)]
    W2, H2 = 344 * 2, 258 * 2
    s = (box[2] - box[0]) / W2
    to_px = lambda x, y: (((np.asarray(x) - box[0]) / s) / 2, ((box[3] - np.asarray(y)) / s) / 2)  # noqa: E731  (CSS px)
    cams = rays_all.drop_duplicates("pano_id")[["pano_id", "x", "y"]]
    labels["map"] = {
        "w": 344, "h": 258, "m10_px": round(10 / s / 2, 2),
        "cam": [round(float(v), 1) for v in to_px(cx, cy)],
        "cams": [[round(float(u), 1), round(float(w), 1)] for u, w in zip(*to_px(cams.x, cams.y)) if 0 <= u <= 344 and 0 <= w <= 258],
        "rays": [[*(round(float(v), 1) for v in to_px(r.x, r.y)), *(round(float(v), 1) for v in to_px(r.X, r.Y)),
                  CLASSES["classes"][r.label]["group"], int(r.pano_id == pid)] for r in rays_all.itertuples()],
        "assets": [[*(round(float(v), 1) for v in to_px(r.x, r.y)), CLASSES["classes"][r.label]["group"]] for r in sel.itertuples()],
    }
    from matplotlib.collections import LineCollection
    for theme, (block, road, casing) in {"light": ("#eaeef2", "#ffffff", "#d0d7de"), "dark": ("#161b22", "#262c35", "#30363d")}.items():
        fig = plt.figure(figsize=(W2 / 100, H2 / 100), dpi=100)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_xlim(box[0], box[2])
        ax.set_ylim(box[1], box[3])
        ax.set_axis_off()
        fig.patch.set_facecolor(block)
        wpx = 15.0 / s * 72 / 100
        ax.add_collection(LineCollection(near, colors=casing, linewidths=wpx + 2 * 72 / 100, capstyle="round", joinstyle="round"))
        ax.add_collection(LineCollection(near, colors=road, linewidths=wpx, capstyle="round", joinstyle="round"))
        fig.savefig(IMG / f"ov-map-{theme}.png", dpi=100, facecolor=block)
        plt.close(fig)
    (IMG / "ov.json").write_text(json.dumps(labels, separators=(",", ":")))
    print(f"[pilot] map: {len(sel)} assets, {len(rays_all)} observations from {len(cams)} panoramas")


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    main()
