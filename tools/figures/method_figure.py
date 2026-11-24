#!/usr/bin/env python
"""Method figure: detection views (a), ground contact and range (b), fusion across panoramas (c).

    python tools/figures/method_figure.py --out docs/assets/figures/method.png

Schematic, drawn from the parameters in streetnet: four views 90 degrees wide, pitched 10 degrees down;
camera 2.5 m above the ground; clustering radius r_c; weighted-median position. Distances in (b) and (c) are
drawn to scale in metres.
"""
from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Circle, FancyBboxPatch, Wedge  # noqa: E402

INK, MUTED, LINE, FILL = "#1d1f21", "#6b7075", "#c9c7c1", "#f3f2ef"
BLUE, ORANGE, GREEN, PURPLE = "#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7"


def square(ax, x, y, size, color, z=4, ec="white"):
    ax.add_patch(FancyBboxPatch((x - size / 2, y - size / 2), size, size, boxstyle=f"round,pad=0,rounding_size={size * 0.25}",
                                fc=color, ec=ec, lw=0.8, zorder=z))


def panel_views(ax):
    colors = [BLUE, ORANGE, GREEN, PURPLE]
    for i in range(4):
        # compass azimuth i * 90 (clockwise from forward = up) -> matplotlib angle 90 - az
        c = 90 - i * 90
        ax.add_patch(Wedge((0, 0), 1.0, c - 45, c + 45, fc=colors[i], alpha=0.13, ec=colors[i], lw=1.2))
        a = np.radians(c)
        ax.text(0.68 * np.cos(a), 0.68 * np.sin(a), f"view {i}\nyaw {i * 90}°", ha="center", va="center",
                fontsize=9.5, color=INK)
    for az in (45, 135, 225, 315):
        a = np.radians(90 - az)
        ax.text(1.12 * np.cos(a), 1.12 * np.sin(a), f"{az}°", ha="center", va="center", fontsize=8.5, color=MUTED)
    ax.annotate("", xy=(0, 0.3), xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.3))
    ax.text(0.04, 0.17, "forward", fontsize=8.5, color=MUTED, rotation=90, va="center")
    square(ax, 0, 0, 0.07, INK)
    ax.text(0, -1.3, "90° field of view, 10° below the horizon, 1008 x 1008 px.\n"
            "Full coverage from 3.6° to 45° below the horizon;\nan object is kept from its owner view (central 90°).",
            ha="center", va="top", fontsize=9, color=MUTED, linespacing=1.5)
    ax.set_xlim(-1.35, 1.35), ax.set_ylim(-1.95, 1.3)


def panel_contact(ax):
    h = 2.5
    ax.plot([-1, 17.5], [0, 0], color=LINE, lw=1.4)
    ax.plot([0, 0], [0, h], color=MUTED, lw=1.0, ls=(0, (2, 2)))
    square(ax, 0, h, 0.42, INK)
    ax.text(-0.35, h / 2, "h = 2.5 m", rotation=90, ha="right", va="center", fontsize=9, color=MUTED)
    xh, xt, xp = 4.5, 10.0, 15.5
    # fire hydrant: ground class, flat-ground estimate
    ax.add_patch(FancyBboxPatch((xh - 0.22, 0), 0.44, 0.8, boxstyle="round,pad=0,rounding_size=0.08", fc=ORANGE, ec="none"))
    ax.plot([0, xh], [h, 0], color=ORANGE, lw=1.3)
    # tree: range class, samples on a strip above the contact pixel
    ax.add_patch(FancyBboxPatch((xt - 0.25, 0), 0.5, 4.2, boxstyle="round,pad=0,rounding_size=0.1", fc=GREEN, ec="none"))
    ax.plot([0, xt], [h, 0], color=GREEN, lw=1.3)
    for t in np.linspace(0.02, 0.12, 5):
        square(ax, xt, t * 4.2 + 0.05, 0.16, INK, z=5)
    # traffic sign on a pole: pole-mounted class
    ax.plot([xp, xp], [0, 3.6], color=BLUE, lw=2.4, solid_capstyle="butt")
    ax.add_patch(FancyBboxPatch((xp - 0.55, 3.0), 1.1, 0.75, boxstyle="round,pad=0,rounding_size=0.08", fc=ORANGE, ec="none"))
    ax.plot([0, xp], [h, 0], color=BLUE, lw=1.0, ls=(0, (4, 3)))
    kw = dict(ha="center", va="top", fontsize=8.8, color=INK, linespacing=1.4)
    ax.text(xh, -0.35, "Fire hydrant\ncontact ray meets\nthe plane z = -h", **kw)
    ax.text(xt, -0.35, "Street tree\nmedian MoGe-2 range of\n5 samples above the contact", **kw)
    ax.text(xp, -0.35, "Traffic sign\ncontact point of\nthe pole below it", **kw)
    ax.text(8, -2.65, "Horizontal range of the contact point: h / tan(depression) on flat ground, or the\n"
            "MoGe-2 range along the leveled ray projected to the horizontal plane.",
            ha="center", va="top", fontsize=9, color=MUTED, linespacing=1.5)
    ax.set_xlim(-1.5, 17.8), ax.set_ylim(-3.9, 4.6)


def panel_fusion(ax):
    ax.fill_between([-3, 3], -2, 26, color=FILL, zorder=0)
    cams = [(0, 0), (0, 10), (0, 20)]
    true = np.array([4.0, 11.0])
    rng = np.random.default_rng(4)
    obs = [true + rng.normal(0, 0.9, 2) for _ in cams]
    for (cx, cy), o in zip(cams, obs):
        ax.plot([cx, o[0]], [cy, o[1]], color=LINE, lw=1.0, zorder=1)
        square(ax, cx, cy, 0.9, INK)
        square(ax, o[0], o[1], 0.55, GREEN, z=5)
    w = np.array([1 / max(np.hypot(o[0] - c[0], o[1] - c[1]), 3.0) for o, c in zip(obs, cams)])
    xs, ys = np.array([o[0] for o in obs]), np.array([o[1] for o in obs])

    def wmed(v):
        k = np.argsort(v)
        c = np.cumsum(w[k])
        return v[k][np.searchsorted(c, 0.5 * c[-1])]

    fx, fy = wmed(xs), wmed(ys)
    ax.add_patch(Circle((fx, fy), 3.0, fc="none", ec=GREEN, lw=1.0, ls=(0, (3, 2))))
    ax.plot([fx - 0.7, fx + 0.7], [fy, fy], color=INK, lw=1.4)
    ax.plot([fx, fx], [fy - 0.7, fy + 0.7], color=INK, lw=1.4)
    ax.text(fx - 2.6, fy + 2.9, "r_c", fontsize=9, color=MUTED, ha="right")
    ax.text(fx + 3.7, fy, "Asset: weighted median\nof the observations,\nweights 1 / max(range, 3 m);\nsupport 3 panoramas",
            fontsize=8.8, color=INK, va="center", linespacing=1.4)
    for (cx, cy) in cams:
        ax.text(cx - 1.0, cy, "camera", fontsize=8.5, color=MUTED, ha="right", va="center")
    ax.text(6.5, -3.2, "Kept when support >= min_support and\nconfidence = 1 - prod(1 - best score per panorama) >= conf_thr.",
            ha="center", va="top", fontsize=9, color=MUTED, linespacing=1.5)
    ax.set_xlim(-6, 19), ax.set_ylim(-7.0, 24)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    plt.rcParams.update({"font.family": "DejaVu Sans"})
    fig, axs = plt.subplots(1, 3, figsize=(16, 5.6), dpi=200, gridspec_kw=dict(width_ratios=[1, 1.45, 1]))
    for ax in axs:
        ax.set_aspect("equal")
        ax.set_axis_off()
    panel_views(axs[0])
    panel_contact(axs[1])
    panel_fusion(axs[2])
    fig.subplots_adjust(left=0.01, right=0.99, top=0.88, bottom=0.02, wspace=0.08)
    for ax, title in zip(axs, ("a  Detection views", "b  Ground contact and range", "c  Fusion across panoramas")):
        fig.text(ax.get_position().x0 + 0.01, 0.94, title, fontsize=12.5, color=INK, ha="left", va="center")
    fig.savefig(a.out, dpi=200, facecolor="white")
    print(f"[figure] -> {a.out}")


if __name__ == "__main__":
    main()
