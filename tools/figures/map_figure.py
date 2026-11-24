#!/usr/bin/env python
"""Overview figure of an asset map: the whole map (left) and a zoomed area with street centrelines (right).

    python tools/figures/map_figure.py --assets run/map/<name>_assets.parquet --boroughs boroughs.geojson \
        --streets street_centerlines.geojson --out docs/assets/figures/nyc_map.jpg

Points are coloured by class group (streetnet/configs/classes.json); classes outside that table are left out.
--boroughs and --streets are GeoJSON files in WGS 84 (NYC Open Data: Borough Boundaries, Street Centerline).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from streetnet.index import ll_to_xy  # noqa: E402

CLASSES = json.loads((Path(__file__).resolve().parents[2] / "streetnet" / "configs" / "classes.json").read_text())
INK, MUTED, LAND, WATER, STREET = "#1d1f21", "#6b7075", "#f3f2ef", "#dfe6ec", "#d4d2cc"
EPSG = 32618


def polygons(path):
    """Outer rings of a (Multi)Polygon GeoJSON, projected to UTM."""
    rings = []
    for f in json.loads(Path(path).read_text())["features"]:
        g = f["geometry"]
        polys = g["coordinates"] if g["type"] == "MultiPolygon" else [g["coordinates"]]
        for p in polys:
            for ring in p:
                a = np.asarray(ring)
                rings.append(np.c_[ll_to_xy(a[:, 0], a[:, 1], EPSG)])
    return rings


def lines(path, bbox):
    out = []
    x0, y0, x1, y1 = bbox
    for f in json.loads(Path(path).read_text())["features"]:
        g = f["geometry"]
        if g is None:
            continue
        parts = g["coordinates"] if g["type"] == "MultiLineString" else [g["coordinates"]]
        for p in parts:
            a = np.asarray(p)
            x, y = ll_to_xy(a[:, 0], a[:, 1], EPSG)
            if x.max() >= x0 and x.min() <= x1 and y.max() >= y0 and y.min() <= y1:
                out.append(np.c_[x, y])
    return out


def scale_bar(ax, x, y, length, label):
    ax.plot([x, x + length], [y, y], color=INK, lw=1.6, solid_capstyle="butt", zorder=6)
    ax.text(x + length / 2, y + length * 0.06, label, ha="center", va="bottom", fontsize=9, color=INK, zorder=6)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--assets", required=True)
    ap.add_argument("--boroughs", required=True)
    ap.add_argument("--streets")
    ap.add_argument("--zoom", default="-73.9450,40.8084,450", help="lon,lat,half side in metres")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    A = pd.read_parquet(a.assets)
    if "label" not in A.columns:
        A = A.rename(columns={"class": "label"})
    A = A[A.label.isin(CLASSES["classes"])]
    if not {"x", "y"} <= set(A.columns):
        A["x"], A["y"] = ll_to_xy(A.lon.to_numpy(), A.lat.to_numpy(), EPSG)
    grp = A.label.map(lambda c: CLASSES["classes"][c]["group"]).to_numpy()
    cols = np.array(CLASSES["colors"])
    rings = polygons(a.boroughs)
    zx, zy, zh = (float(v) for v in a.zoom.split(","))
    cx, cy = ll_to_xy(zx, zy, EPSG)
    cx, cy = float(cx), float(cy)
    zb = (cx - zh, cy - zh, cx + zh, cy + zh)

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    W, H, side = 15.4, 8.9, 7.55                      # inches; two square panels of `side`
    fig = plt.figure(figsize=(W, H), dpi=200, facecolor="white")
    axL = fig.add_axes([0.0, 1 - side / H, side / W, side / H])
    axR = fig.add_axes([1 - side / W, 1 - side / H, side / W, side / H])
    for ax in (axL, axR):
        ax.set_facecolor(WATER)
        ax.set_adjustable("datalim")
        ax.set_xticks([]), ax.set_yticks([])
        ax.set_aspect("equal")
        for s in ax.spines.values():
            s.set_visible(False)
        for r in rings:
            ax.fill(r[:, 0], r[:, 1], color=LAND, lw=0, zorder=1)
    order = np.argsort(np.bincount(grp, minlength=len(cols))[grp])[::-1]     # frequent groups first, below
    axL.scatter(A.x.to_numpy()[order], A.y.to_numpy()[order], s=0.09, c=cols[grp[order]], marker="s", lw=0,
                alpha=0.75, rasterized=True, zorder=2)
    allx = np.concatenate([r[:, 0] for r in rings])
    ally = np.concatenate([r[:, 1] for r in rings])
    pad = 800
    axL.set_xlim(allx.min() - pad, allx.max() + pad)
    axL.set_ylim(ally.min() - pad, ally.max() + pad)
    axL.add_patch(plt.Rectangle((zb[0], zb[1]), 2 * zh, 2 * zh, fill=False, ec=INK, lw=1.0, zorder=5))
    scale_bar(axL, allx.min() + 1500, ally.max() - 2500, 5000, "5 km")

    if a.streets:
        for ln in lines(a.streets, (zb[0] - 50, zb[1] - 50, zb[2] + 50, zb[3] + 50)):
            axR.plot(ln[:, 0], ln[:, 1], color=STREET, lw=5.5, solid_capstyle="round", zorder=1.5)
    inz = (A.x >= zb[0]) & (A.x <= zb[2]) & (A.y >= zb[1]) & (A.y <= zb[3])
    Z = A[inz]
    gz = grp[inz.to_numpy()]
    axR.scatter(Z.x, Z.y, s=16, c=cols[gz], marker="s", lw=0.5, edgecolors="white", zorder=3)
    axR.set_xlim(zb[0], zb[2])
    axR.set_ylim(zb[1], zb[3])
    scale_bar(axR, zb[0] + 40, zb[1] + 40, 100, "100 m")

    counts = np.bincount(grp, minlength=len(cols))
    lax = fig.add_axes([0.0, 0.0, 1.0, (H - side) / H * 0.92])
    lax.set_axis_off()
    lax.set_xlim(0, W), lax.set_ylim(0, 1)
    for i, (name, c) in enumerate(zip(CLASSES["groups"], CLASSES["colors"])):
        x, y = 0.25 + (i % 4) * W / 4, 0.68 - (i // 4) * 0.42
        hy = 0.14 / ((H - side) * 0.92)                 # a 0.14 inch square in legend data units
        lax.add_patch(FancyBboxPatch((x, y - hy / 2), 0.14, hy, boxstyle="round,pad=0,rounding_size=0.035",
                                     fc=c, ec="none", mutation_aspect=hy / 0.14))
        lax.text(x + 0.22, y, name, va="center", ha="left", fontsize=10.5, color=INK)
        lax.text(x + 0.22 + 2.35, y, f"{counts[i]:,}", va="center", ha="right", fontsize=10.5, color=MUTED)
    fig.savefig(a.out, dpi=200, pil_kwargs={"quality": 90, "optimize": True})
    print(f"[figure] {len(A):,} assets, {int(inz.sum()):,} in the zoom -> {a.out}")


if __name__ == "__main__":
    main()
