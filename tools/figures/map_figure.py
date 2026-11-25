#!/usr/bin/env python
"""Map figure: asset density on the street network of the city and a close-up with individual assets.

    python tools/figures/map_figure.py --assets runs/city/map/city_assets.parquet --boroughs boroughs.geojson \
        --streets street_centerlines.geojson --parks parks.geojson [--summary runs/city/map/city_summary.json]

Computation, citywide panel:
    streets   = centrelines of roadway type 1 (street) and 10 (alley), resampled into pieces of 20 m
    count(p)  = assets whose nearest piece is p, within 30 m of its midpoint
    density   = (sum of counts over p and the two pieces on each side) / (length of those pieces) x 100 m
    colour    = density on a sequential blue ramp, linear from 0 to the largest round value below the 95th
                percentile of the pieces with assets, saturated above; line width grows with the same value
Close-up panel: the assets inside a rectangle around --zoom, one rounded square per asset in the colour of its
class group (streetnet/configs/classes.json), over the street centrelines (as roadways) and parks. When the
rectangle holds more than --zoom-max-assets assets, its sides shrink by sqrt(max / count), to at least 90 m half height.

Classes outside streetnet/configs/classes.json (utility poles) are left out. The basemap layers are GeoJSON in
WGS 84 (NYC Open Data: Borough Boundaries, Street Centerline (CSCL), Parks Properties). The tool writes the panel
images and their labels to tools/figures/slides/img/ and renders tools/figures/slides/nyc_map.html to
docs/assets/figures/nyc_map.jpg and nyc_map-dark.jpg (tools/figures/render.py).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from streetnet.index import ll_to_xy  # noqa: E402

CLASSES = json.loads((ROOT / "streetnet" / "configs" / "classes.json").read_text())
IMG = ROOT / "tools" / "figures" / "slides" / "img"
EPSG = 32618
SCALE = 2                                   # device pixels per CSS pixel, as in render.py

# panel sizes in CSS pixels; keep in step with tools/figures/slides/nyc_map.html
CITY_W, CITY_H = 600, 540
ZOOM_W, ZOOM_H = 460, 400

THEMES = {
    "light": {
        "water": "#eaeef2", "land": "#ffffff", "park": "#e8efe6", "street0": "#d8dee4",
        "ramp": ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"],
        "block": "#f3f5f8", "road": "#ffffff", "casing": "#d0d7de", "highway": "#ffffff", "path": "#d8dee4",
        "groups": CLASSES["colors"],
    },
    "dark": {
        "water": "#0d1117", "land": "#161b22", "park": "#18241d", "street0": "#2d333b",
        "ramp": ["#1b3f6e", "#184f95", "#256abf", "#3987e5", "#6da7ec", "#9ec5f4", "#cde2fb"],
        "block": "#161b22", "road": "#262c35", "casing": "#30363d", "highway": "#2d343e", "path": "#30363d",
        "groups": ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#0ca30c", "#9085e9", "#e66767"],
    },
}
BOROUGH_LABELS = {                                         # label anchor, lon/lat
    "Manhattan": (-74.0395, 40.775), "The Bronx": (-73.862, 40.873), "Queens": (-73.805, 40.705),
    "Brooklyn": (-73.958, 40.648), "Staten Island": (-74.152, 40.574),
}


# ---------------------------------------------------------------- geometry
def _project_parts(parts: list[np.ndarray]) -> list[np.ndarray]:
    if not parts:
        return []
    n = np.array([len(p) for p in parts])
    allc = np.concatenate(parts)
    x, y = ll_to_xy(allc[:, 0], allc[:, 1], EPSG)
    xy = np.c_[x, y]
    return np.split(xy, np.cumsum(n)[:-1])


def read_lines(path, types: set[str] | None = None) -> tuple[list[np.ndarray], list[str], list[str]]:
    parts, kinds, names = [], [], []
    for f in json.loads(Path(path).read_text())["features"]:
        g, p = f["geometry"], f.get("properties") or {}
        if g is None or (types is not None and str(p.get("rw_type")) not in types):
            continue
        for c in (g["coordinates"] if g["type"] == "MultiLineString" else [g["coordinates"]]):
            if len(c) >= 2:
                parts.append(np.asarray(c, float)[:, :2])
                kinds.append(str(p.get("rw_type")))
                names.append(p.get("stname_label") or "")
    return _project_parts(parts), kinds, names


def read_rings(path) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Outer rings and holes of a (Multi)Polygon GeoJSON, projected to UTM."""
    outer, holes = [], []
    for f in json.loads(Path(path).read_text())["features"]:
        g = f["geometry"]
        if g is None:
            continue
        for poly in (g["coordinates"] if g["type"] == "MultiPolygon" else [g["coordinates"]]):
            for i, ring in enumerate(poly):
                (outer if i == 0 else holes).append(np.asarray(ring, float)[:, :2])
    return _project_parts(outer), _project_parts(holes)


def resample(lines: list[np.ndarray], step: float) -> tuple[np.ndarray, np.ndarray]:
    """Pieces of at most `step` metres along each line: (N, 2, 2) endpoints and the line index of each piece."""
    segs, owner = [], []
    for i, ln in enumerate(lines):
        d = np.r_[0, np.cumsum(np.hypot(*np.diff(ln, axis=0).T))]
        if d[-1] < 1e-6:
            continue
        t = np.linspace(0, d[-1], max(2, int(np.ceil(d[-1] / step)) + 1))
        p = np.c_[np.interp(t, d, ln[:, 0]), np.interp(t, d, ln[:, 1])]
        segs.append(np.stack([p[:-1], p[1:]], axis=1))
        owner.append(np.full(len(p) - 1, i))
    return np.concatenate(segs), np.concatenate(owner)


def street_density(segs: np.ndarray, owner: np.ndarray, ax: np.ndarray, ay: np.ndarray,
                   radius: float = 30.0, window: int = 2) -> np.ndarray:
    from scipy.spatial import cKDTree
    mid = segs.mean(axis=1)
    length = np.hypot(*(segs[:, 1] - segs[:, 0]).T)
    d, k = cKDTree(mid).query(np.c_[ax, ay], distance_upper_bound=radius)
    count = np.bincount(k[np.isfinite(d)], minlength=len(segs)).astype(float)
    # moving sums over +-window pieces of the same line
    cs, ls = np.zeros_like(count), np.zeros_like(count)
    for s in range(-window, window + 1):
        j = np.clip(np.arange(len(segs)) + s, 0, len(segs) - 1)
        same = owner[j] == owner
        cs += np.where(same, count[j], 0)
        ls += np.where(same, length[j], 0)
    return 100.0 * cs / np.maximum(ls, 1.0)


def ramp(values: np.ndarray, stops: list[str]) -> np.ndarray:
    from matplotlib.colors import to_rgb
    c = np.array([to_rgb(s) for s in stops])
    t = np.clip(values, 0, 1) * (len(c) - 1)
    i = np.minimum(t.astype(int), len(c) - 2)
    f = (t - i)[:, None]
    return c[i] * (1 - f) + c[i + 1] * f


def nice_ticks(vmax: float) -> list[float]:
    """0 and multiples of a round step up to vmax, at most four steps."""
    for step in (1, 2, 5, 10, 20, 25, 50, 100):
        if vmax / step <= 4:
            return [step * i for i in range(int(vmax // step) + 1)]
    return [0, vmax]


# ---------------------------------------------------------------- drawing
class Panel:
    def __init__(self, bbox, w_css, h_css):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        x0, y0, x1, y1 = bbox
        self.w, self.h = w_css * SCALE, h_css * SCALE
        sx = (x1 - x0) / self.w
        sy = (y1 - y0) / self.h
        s = max(sx, sy)                                     # metres per device pixel, aspect kept
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        self.x0, self.x1 = cx - s * self.w / 2, cx + s * self.w / 2
        self.y0, self.y1 = cy - s * self.h / 2, cy + s * self.h / 2
        self.mpp = s
        self.plt = plt
        self.fig = plt.figure(figsize=(self.w / 100, self.h / 100), dpi=100)
        self.ax = self.fig.add_axes([0, 0, 1, 1])
        self.ax.set_xlim(self.x0, self.x1)
        self.ax.set_ylim(self.y0, self.y1)
        self.ax.set_axis_off()

    def px(self, x, y):
        """Map metres to CSS pixels of the panel."""
        return ((np.asarray(x) - self.x0) / self.mpp / SCALE, (self.y1 - np.asarray(y)) / self.mpp / SCALE)

    def lw(self, device_px: float) -> float:
        return device_px * 72 / 100

    def fill(self, rings, color, z):
        from matplotlib.collections import PolyCollection
        vis = [r for r in rings if r[:, 0].max() >= self.x0 and r[:, 0].min() <= self.x1
               and r[:, 1].max() >= self.y0 and r[:, 1].min() <= self.y1]
        self.ax.add_collection(PolyCollection(vis, facecolors=color, edgecolors="none", zorder=z, antialiaseds=True))

    def lines(self, segs, colors, width_px, z, capstyle="round"):
        from matplotlib.collections import LineCollection
        self.ax.add_collection(LineCollection(segs, colors=colors, linewidths=self.lw(width_px), zorder=z,
                                              capstyle=capstyle, joinstyle="round", antialiaseds=True))

    def save(self, path, bg):
        kw = {"pil_kwargs": {"quality": 92, "subsampling": 0}} if str(path).endswith(".jpg") else {}
        self.fig.savefig(path, dpi=100, facecolor=bg, **kw)
        self.plt.close(self.fig)


def clip_lines(lines, kinds, bbox, pad):
    x0, y0, x1, y1 = bbox
    out = []
    for ln, k in zip(lines, kinds):
        if ln[:, 0].max() >= x0 - pad and ln[:, 0].min() <= x1 + pad and ln[:, 1].max() >= y0 - pad and ln[:, 1].min() <= y1 + pad:
            out.append((ln, k))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--assets", required=True, help="map assets (Parquet): label or class, and x, y or lon, lat")
    ap.add_argument("--boroughs", required=True)
    ap.add_argument("--streets", required=True)
    ap.add_argument("--parks", required=True)
    ap.add_argument("--summary", help="the map's summary JSON; its n_panos goes into the caption")
    ap.add_argument("--note", help="caption sentence describing the map (default: from --summary)")
    ap.add_argument("--zoom", default="-73.9452,40.8080,210", help="close-up: lon,lat,half side in metres")
    ap.add_argument("--zoom-label", default="W 125th St and Lenox Ave, Harlem")
    ap.add_argument("--zoom-max-assets", type=int, default=1400,
                    help="shrink the close-up (keeping its centre) until it holds about this many assets")
    ap.add_argument("--no-render", action="store_true", help="write the panels only")
    a = ap.parse_args()

    A = pd.read_parquet(a.assets)
    if "label" not in A.columns:
        A = A.rename(columns={"class": "label"})
    A = A[A.label.isin(CLASSES["classes"])]
    if not {"x", "y"} <= set(A.columns):
        A["x"], A["y"] = ll_to_xy(A.lon.to_numpy(), A.lat.to_numpy(), EPSG)
    ax, ay = A.x.to_numpy(float), A.y.to_numpy(float)
    grp = A.label.map(lambda c: CLASSES["classes"][c]["group"]).to_numpy()

    land, land_holes = read_rings(a.boroughs)
    parks, park_holes = read_rings(a.parks)
    streets, kinds, names = read_lines(a.streets)
    print(f"[map] {len(A):,} assets, {len(streets):,} street lines, {len(parks):,} park polygons")

    dens_lines = [ln for ln, k in zip(streets, kinds) if k in ("1", "10")]
    segs, owner = resample(dens_lines, 20.0)
    dens = street_density(segs, owner, ax, ay)
    has = dens > 0
    p95 = float(np.percentile(dens[has], 95)) if has.any() else 1.0
    ticks = nice_ticks(p95)
    vmax = ticks[-1] if ticks[-1] > 0 else p95
    print(f"[map] {len(segs):,} street pieces, {has.mean():.1%} with assets, colour scale 0 to {vmax:g} per 100 m")

    allx = np.concatenate([r[:, 0] for r in land])
    ally = np.concatenate([r[:, 1] for r in land])
    city_bbox = (allx.min() - 600, ally.min() - 600, allx.max() + 600, ally.max() + 600)
    zx, zy, zh = (float(v) for v in a.zoom.split(","))
    cx, cy = (float(v) for v in ll_to_xy(zx, zy, EPSG))
    zbox = (cx - zh * ZOOM_W / ZOOM_H, cy - zh, cx + zh * ZOOM_W / ZOOM_H, cy + zh)
    n_zoom = int(((ax >= zbox[0]) & (ax <= zbox[2]) & (ay >= zbox[1]) & (ay <= zbox[3])).sum())
    if n_zoom > a.zoom_max_assets:                       # dense maps: a smaller square keeps the markers apart
        zh = max(90.0, zh * (a.zoom_max_assets / n_zoom) ** 0.5)
        zbox = (cx - zh * ZOOM_W / ZOOM_H, cy - zh, cx + zh * ZOOM_W / ZOOM_H, cy + zh)

    labels = {}
    IMG.mkdir(parents=True, exist_ok=True)
    for theme, T in THEMES.items():
        # ---- citywide density
        P = Panel(city_bbox, CITY_W, CITY_H)
        P.fill(land, T["land"], 1)
        P.fill(land_holes, T["water"], 1.1)
        P.fill(parks, T["park"], 1.2)
        order = np.argsort(dens)
        t = np.clip(dens[order] / vmax, 0, 1)
        col = ramp(t, T["ramp"])
        col[dens[order] == 0] = np.array([int(T["street0"][i:i + 2], 16) / 255 for i in (1, 3, 5)])
        P.lines(segs[order], col, 0.75 + 1.1 * t, 2, capstyle="butt")
        P.save(IMG / f"nyc_map-city-{theme}.jpg", T["water"])
        if theme == "light":
            bx = P.px([zbox[0], zbox[2]], [zbox[3], zbox[1]])
            labels["city"] = {
                "boroughs": [{"name": n, "x": round(float(P.px(*ll_to_xy(lo, la, EPSG))[0]), 1),
                              "y": round(float(P.px(*ll_to_xy(lo, la, EPSG))[1]), 1)} for n, (lo, la) in BOROUGH_LABELS.items()],
                "zoom_box": [round(float(v), 1) for v in (bx[0][0], bx[1][0], bx[0][1] - bx[0][0], bx[1][1] - bx[1][0])],
                "km_px": round(1000 / P.mpp / SCALE, 3),
            }

        # ---- close-up
        Z = Panel(zbox, ZOOM_W, ZOOM_H)
        Z.fill(land, T["block"], 1)
        Z.fill(land_holes, T["water"], 1.1)
        Z.fill(parks, T["park"], 1.2)
        near = clip_lines(streets, kinds, zbox, 200)
        width = {"1": 15.0, "10": 7.0, "2": 24.0, "9": 9.0, "3": 20.0}
        m2px = 1 / Z.mpp
        for kset, z in ((("2", "3", "9"), 3), (("1", "10"), 4)):
            sel = [(ln, k) for ln, k in near if k in kset]
            if not sel:
                continue
            for casing in (True, False):
                segs_k = [ln for ln, _ in sel]
                w = np.array([width[k] for _, k in sel]) * m2px
                from matplotlib.collections import LineCollection
                Z.ax.add_collection(LineCollection(segs_k, colors=T["casing"] if casing else (T["highway"] if z == 3 else T["road"]),
                                                   linewidths=Z.lw(1) * (w + (2.0 if casing else 0)),
                                                   zorder=z + (0 if casing else 0.5), capstyle="round", joinstyle="round"))
        paths = [ln for ln, k in near if k in ("6", "7", "5")]
        if paths:
            Z.lines(paths, T["path"], 1.6, 3.2)
        Z.save(IMG / f"nyc_map-zoom-{theme}.png", T["water"])
        if theme == "light":
            inz = (ax >= zbox[0]) & (ax <= zbox[2]) & (ay >= zbox[1]) & (ay <= zbox[3])
            freq = np.bincount(grp, minlength=8)
            idx = np.flatnonzero(inz)
            idx = idx[np.argsort(-freq[grp[idx]], kind="stable")]          # frequent groups first, below
            px, py = Z.px(ax[idx], ay[idx])
            labels["zoom"] = {
                "assets": [[round(float(u), 1), round(float(v), 1), int(g)] for u, v, g in zip(px, py, grp[idx])],
                "m100_px": round(100 / Z.mpp / SCALE, 2), "label": a.zoom_label,
                "streets": street_labels(near, names_for(near, streets, names), Z),
            }
            print(f"[map] close-up: {len(idx):,} assets in {2 * zh * ZOOM_W / ZOOM_H:.0f} m x {2 * zh:.0f} m")

    labels["legend"] = {"ticks": ticks, "vmax": vmax, "ramp": {t: THEMES[t]["ramp"] for t in THEMES}}
    note = a.note
    if not note and a.summary:
        S = json.loads(Path(a.summary).read_text())
        note = f"Assets detected in {S['n_panos']:,} panoramas."
    labels["note"] = (note + " " if note else "") + "Basemap: NYC Open Data. North is up."
    (IMG / "nyc_map.json").write_text(json.dumps(labels, separators=(",", ":")))
    print(f"[map] panels -> {IMG.relative_to(ROOT)}/nyc_map-*")
    if not a.no_render:
        import render
        asyncio.run(render.render(["nyc_map"]))


def names_for(near, streets, names):
    lookup = {id(ln): nm for ln, nm in zip(streets, names)}
    return [lookup.get(id(ln), "") for ln, _ in near]


def street_labels(near, near_names, Z, wanted=("W 125 ST", "LENOX AVE", "ADAM CLAYTON POWELL JR BLVD", "5 AVE")):
    """One label per wanted street name: the midpoint and angle of its longest visible piece."""
    best = {}
    for (ln, _), nm in zip(near, near_names):
        if nm not in wanted:
            continue
        px, py = Z.px(ln[:, 0], ln[:, 1])
        inside = (px > 40) & (px < ZOOM_W - 40) & (py > 30) & (py < ZOOM_H - 30)
        if inside.sum() < 2:
            continue
        p = np.c_[px[inside], py[inside]]
        L = np.hypot(*(p[-1] - p[0]))
        if L > best.get(nm, (0,))[0]:
            ang = np.degrees(np.arctan2(p[-1, 1] - p[0, 1], p[-1, 0] - p[0, 0]))
            if ang > 90:
                ang -= 180
            if ang < -90:
                ang += 180
            m = p.mean(axis=0)
            best[nm] = (L, round(float(m[0]), 1), round(float(m[1]), 1), round(float(ang), 1))
    return [{"name": pretty(nm), "x": v[1], "y": v[2], "angle": v[3]} for nm, v in best.items()]


def pretty(name: str) -> str:
    """'W 125 ST' -> 'W 125th St'."""
    abbr = {"ST": "St", "AVE": "Ave", "BLVD": "Blvd", "PL": "Pl", "DR": "Dr", "RD": "Rd", "JR": "Jr"}
    out = []
    for w in name.split():
        if w.isdigit():
            n = int(w)
            suf = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
            out.append(f"{n}{suf}")
        elif w in ("W", "E", "N", "S"):
            out.append(w)
        else:
            out.append(abbr.get(w, w.capitalize()))
    return " ".join(out)

if __name__ == "__main__":
    main()
