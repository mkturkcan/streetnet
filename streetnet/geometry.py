"""Panorama, camera and ground geometry.

Frames
------
cam    Panorama sensor frame: x right, y forward (the center column of the equirectangular image), z up.
       Pixel (u, v) of a W x H equirectangular image, pixel centers at +0.5:
           lam = ((u + 0.5) / W - 0.5) * 2 pi     longitude: 0 at the center column, positive to the right
           phi = (0.5 - (v + 0.5) / H) * pi       latitude: positive up
           d   = (cos(phi) sin(lam), cos(phi) cos(lam), sin(phi))
lev    The cam frame rotated so that the local ground plane is z = -h, with h the camera height.
       lev <- cam is a small rotation R_lev; the identity for a level camera.
world  A metric map frame (UTM): x east, y north, z up. world <- lev is a rotation about z by the
       panorama heading, the compass bearing of the center column (clockwise from north). A lev ray with
       azimuth lam (clockwise from forward) has compass bearing heading + lam.
view   A pinhole camera in the OpenCV convention (x right, y down, z forward), pointed at (yaw, pitch)
       relative to cam forward.
"""
from __future__ import annotations

import math
import zlib
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

# ----------------------------------------------------------------------------- equirectangular <-> rays


def equirect_pixel_to_ray(u, v, W: int, H: int) -> np.ndarray:
    """Unit rays (..., 3) in the cam frame for equirectangular pixel coordinates."""
    lam = ((np.asarray(u, dtype=np.float64) + 0.5) / W - 0.5) * 2 * np.pi
    phi = (0.5 - (np.asarray(v, dtype=np.float64) + 0.5) / H) * np.pi
    c = np.cos(phi)
    return np.stack([c * np.sin(lam), c * np.cos(lam), np.sin(phi)], axis=-1)


def ray_to_equirect_pixel(d, W: int, H: int):
    """Equirectangular pixel coordinates (u, v) of cam-frame rays of any length."""
    d = np.asarray(d, dtype=np.float64)
    lam = np.arctan2(d[..., 0], d[..., 1])
    phi = np.arcsin(np.clip(d[..., 2] / np.linalg.norm(d, axis=-1), -1, 1))
    return (lam / (2 * np.pi) + 0.5) * W - 0.5, (0.5 - phi / np.pi) * H - 0.5


def azimuth_of_ray(d) -> np.ndarray:
    """Azimuth in degrees, clockwise from cam forward, in [0, 360)."""
    d = np.asarray(d)
    return np.degrees(np.arctan2(d[..., 0], d[..., 1])) % 360.0


def elevation_of_ray(d) -> np.ndarray:
    """Elevation in degrees above the cam horizon."""
    d = np.asarray(d)
    return np.degrees(np.arcsin(np.clip(d[..., 2] / np.linalg.norm(d, axis=-1), -1, 1)))


# ----------------------------------------------------------------------------- rotations and leveling


def rot_z(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)


def leveling_from_normal(n_cam) -> np.ndarray:
    """Rotation R_lev (lev <- cam) taking the ground normal n (cam frame) to +z (Rodrigues)."""
    n = np.asarray(n_cam, dtype=np.float64)
    n = n / np.linalg.norm(n)
    z = np.array([0.0, 0.0, 1.0])
    v = np.cross(n, z)
    s, c = np.linalg.norm(v), float(n @ z)
    if s < 1e-12:
        return np.eye(3)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * ((1 - c) / s**2)


LEVEL_TILT_MAX = 12.0     # degrees; larger metadata tilts are treated as unreliable
LEVEL_ROLL_MAX = 8.0
ROLL_GAIN = -0.7          # calibrated on the Upper Manhattan pilot


def metadata_normal(pitch_deg, roll_deg) -> np.ndarray:
    """Ground normal (N, 3) in the cam frame from the capture metadata's camera pitch and roll.

    The rule was calibrated on Street View metadata in the pilot study (pitch 0 = level):
        tilt = -pitch,  roll = roll wrapped to [-180, 180)
        n    = normalize(sin(-0.7 roll), sin(tilt), 1)
        n    = (0, 0, 1)  when |tilt| > 12 deg or |roll| > 8 deg
    """
    tilt = -np.asarray(pitch_deg, dtype=np.float64)
    roll = (np.asarray(roll_deg, dtype=np.float64) + 180.0) % 360.0 - 180.0
    n = np.stack([np.sin(np.radians(ROLL_GAIN * roll)), np.sin(np.radians(tilt)), np.ones_like(tilt)], axis=-1)
    n /= np.linalg.norm(n, axis=-1, keepdims=True)
    bad = (np.abs(tilt) > LEVEL_TILT_MAX) | (np.abs(roll) > LEVEL_ROLL_MAX)
    n[bad] = (0.0, 0.0, 1.0)
    return n


def world_from_lev(heading_deg: float) -> np.ndarray:
    """world <- lev rotation for a compass heading (clockwise from north)."""
    return rot_z(-math.radians(heading_deg))


def to_world(p_lev: np.ndarray, heading_deg: float, cam_x: float, cam_y: float):
    """lev-frame points relative to the camera -> world (x, y)."""
    q = np.asarray(p_lev) @ world_from_lev(heading_deg).T
    return q[:, 0] + cam_x, q[:, 1] + cam_y


def fit_road_plane_scale(
    range_map: np.ndarray,
    road_mask: np.ndarray,
    R_lev: np.ndarray,
    h: float = 2.5,
    r_min: float = 7.0,
    r_max: float = 15.0,
    dz_max: float = -0.15,
    min_pixels: int = 50,
    clip_bounds: tuple[float, float] = (0.6, 1.6),
) -> float:
    """Zero-GT per-panorama monocular range scale factor calibrated against unobstructed road pixels.

    Compares the analytical leveled road-plane range h / (-d_lev.z) against predicted radial distances
    on pixels marked as roadway in `road_mask` across `[r_min, r_max]` m (default 7 to 15 m, excluding
    near-field ego-vehicle and nadir privacy-blur regions), returning the median ratio clipped to `clip_bounds`.
    Automatically aligns `road_mask` to `range_map` resolution if shapes differ.
    Returns 1.0 if fewer than `min_pixels` valid road pixels exist.
    """
    H, W = range_map.shape[:2]
    rm = np.asarray(road_mask, dtype=bool)
    if rm.shape != (H, W):
        vi = np.clip(np.rint(np.linspace(0, rm.shape[0] - 1, H)).astype(int), 0, rm.shape[0] - 1)
        ui = np.clip(np.rint(np.linspace(0, rm.shape[1] - 1, W)).astype(int), 0, rm.shape[1] - 1)
        rm = rm[np.ix_(vi, ui)]
    ys, xs = np.nonzero(rm)
    if len(ys) < min_pixels:
        return 1.0
    if len(ys) > 8192:
        step = len(ys) // 8192
        ys, xs = ys[::step], xs[::step]
    d = equirect_pixel_to_ray(xs, ys, W, H) @ np.asarray(R_lev, dtype=np.float64).T
    dz = d[:, 2]
    r_pred = np.asarray(range_map[ys, xs], dtype=np.float64)
    ok = (dz < dz_max) & np.isfinite(r_pred) & (r_pred > 0.5)
    if ok.sum() < min_pixels:
        return 1.0
    r_road = h / (-dz[ok])
    horiz = r_road * np.hypot(d[ok, 0], d[ok, 1])
    keep = (horiz >= r_min) & (horiz <= r_max)
    if keep.sum() < min_pixels:
        return 1.0
    alpha = float(np.median(r_road[keep] / r_pred[ok][keep]))
    return float(np.clip(alpha, clip_bounds[0], clip_bounds[1]))


# ----------------------------------------------------------------------------- scene-context helpers


OCC_H, OCC_W = 128, 256

# Representative target heights above ground (meters) for 3D sightline occlusion checks.
ASSET_SIGHTLINE_Z = {
    "storm_drain": 0.15,
    "curb_ramp": 0.25,
    "manhole": 0.10,
    "fire_hydrant": 0.85,
    "trash_can": 0.90,
    "bike_rack": 0.85,
    "bench": 0.85,
    "bollard": 0.85,
    "barrier": 0.90,
    "planter": 0.85,
    "bike_share_dock": 0.90,
    "mailbox": 1.15,
    "parking_meter": 1.45,
    "fire_alarm_box": 1.60,
    "tree": 2.20,
    "kiosk": 2.40,
    "bus_shelter": 2.30,
    "street_light": 3.00,
    "sign_pole": 2.20,
    "utility_pole": 2.50,
    "traffic_sign": 2.80,
    "bus_stop_sign": 2.40,
    "traffic_signal": 3.20,
}


def extract_crosswalk_points(
    sem: np.ndarray,
    cam_x: float,
    cam_y: float,
    heading_deg: float,
    R_lev: np.ndarray,
    h: float = 2.5,
    max_points: int = 60,
    min_pixels: int = 20,
) -> np.ndarray:
    """Project Mask2Former crosswalk pixels (Vistas IDs 8, 23) onto the leveled ground plane -> (K, 2) float32."""
    Hs, Ws = sem.shape[:2]
    ys, xs = np.nonzero((sem == 8) | (sem == 23))
    if len(ys) < min_pixels:
        return np.zeros((0, 2), np.float32)
    step = max(1, len(ys) // max_points)
    ys, xs = ys[::step], xs[::step]
    rays = equirect_pixel_to_ray(xs.astype(float), ys.astype(float), Ws, Hs)
    dl = rays @ np.asarray(R_lev, dtype=np.float64).T
    ok = dl[:, 2] < -0.08
    if not ok.any():
        return np.zeros((0, 2), np.float32)
    dl = dl[ok]
    t = float(h) / (-dl[:, 2])
    ok2 = (t > 1.5) & (t < 22.0)
    if not ok2.any():
        return np.zeros((0, 2), np.float32)
    pl = dl[ok2] * t[ok2, None]
    wx, wy = to_world(pl, heading_deg, cam_x, cam_y)
    return np.column_stack([wx, wy]).astype(np.float32)


def pack_occlusion_mask(sem: np.ndarray, occ_lut: np.ndarray, out_h: int = OCC_H, out_w: int = OCC_W) -> bytes:
    """Downsample and zlib-compress a 1-bit equirectangular vehicle/structure occlusion mask (~1 KiB/pano)."""
    Hs, Ws = sem.shape[:2]
    vi = np.clip(np.rint(np.linspace(0, Hs - 1, out_h)).astype(int), 0, Hs - 1)
    ui = np.clip(np.rint(np.linspace(0, Ws - 1, out_w)).astype(int), 0, Ws - 1)
    occ = occ_lut[sem[np.ix_(vi, ui)]]
    return zlib.compress(np.packbits(occ.ravel()).tobytes(), level=6)


def unpack_occlusion_mask(blob: bytes, h: int = OCC_H, w: int = OCC_W) -> np.ndarray:
    """Decode a compressed occlusion mask back into an (h, w) boolean array."""
    raw = np.frombuffer(zlib.decompress(blob), dtype=np.uint8)
    return np.unpackbits(raw, count=h * w).reshape(h, w).astype(bool)


# ----------------------------------------------------------------------------- perspective views


@dataclass(frozen=True)
class View:
    """A square or rectangular pinhole view cut from the panorama."""

    yaw: float      # degrees, clockwise from panorama forward
    pitch: float    # degrees, positive up
    fov: float      # horizontal field of view, degrees
    w: int
    h: int

    @property
    def f(self) -> float:
        return (self.w / 2) / math.tan(math.radians(self.fov) / 2)

    @property
    def K(self) -> np.ndarray:
        return np.array([[self.f, 0, self.w / 2], [0, self.f, self.h / 2], [0, 0, 1.0]])

    def basis(self):
        """cam-frame unit vectors (right, forward, up) of the view."""
        p, t = math.radians(self.yaw), math.radians(self.pitch)
        right = np.array([math.cos(p), -math.sin(p), 0.0])
        fwd = np.array([math.sin(p) * math.cos(t), math.cos(p) * math.cos(t), math.sin(t)])
        return right, fwd, np.cross(right, fwd)

    def R_cam_from_view(self) -> np.ndarray:
        """3 x 3 matrix mapping OpenCV view vectors (x right, y down, z forward) to the cam frame."""
        right, fwd, up = self.basis()
        return np.stack([right, -up, fwd], axis=1)

    def pixel_to_ray(self, px, py) -> np.ndarray:
        """cam-frame unit rays for view pixel indices (pixel centers at integers)."""
        px = np.asarray(px, dtype=np.float64)
        py = np.asarray(py, dtype=np.float64)
        a = (px + 0.5 - self.w / 2) / self.f
        b = (py + 0.5 - self.h / 2) / self.f
        r = np.stack([a, b, np.ones_like(a)], axis=-1) @ self.R_cam_from_view().T
        return r / np.linalg.norm(r, axis=-1, keepdims=True)

    def ray_to_pixel(self, d):
        """Project cam-frame rays into the view. Returns px, py and an in-front mask."""
        v = np.asarray(d, dtype=np.float64) @ self.R_cam_from_view()
        z = v[..., 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            px = v[..., 0] / z * self.f + self.w / 2 - 0.5
            py = v[..., 1] / z * self.f + self.h / 2 - 0.5
        return px, py, z > 1e-6


def detection_views(n: int = 4, pitch: float = -10.0, fov: float = 90.0, size: int = 1008) -> tuple[View, ...]:
    """n horizon views spaced 360/n degrees apart, pitched down. size 1008 is SAM 3.1's input size.

    With n = 4, 90 degree views and pitch -10, a view spans elevations -55 to +35 degrees along its center column;
    all azimuths are covered from 3.6 to 45.3 degrees below the horizon, ground contacts from 2.5 m to 40 m for a
    camera 2.5 m above the ground. Where two views meet, about 1 degree of the horizon is missed."""
    return tuple(View(yaw=i * 360.0 / n, pitch=pitch, fov=fov, w=size, h=size) for i in range(n))


def owner_view(azimuth_deg, n_views: int) -> int:
    """Index of the view whose central azimuth sector (+-180/n degrees) contains the azimuth. An object
    seen in two overlapping views is kept from its owner view."""
    step = 360.0 / n_views
    return int(np.floor(((np.asarray(azimuth_deg) + step / 2) % 360.0) / step)) % n_views


@lru_cache(maxsize=8)
def equirect_lookup(views: tuple[View, ...], W: int = 2048, H: int = 1024):
    """For each equirectangular pixel, the most central view that covers it and the pixel inside it.
    Returns (view index int16 [H, W], -1 = outside all views; px float32 [H, W]; py float32 [H, W])."""
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    d = equirect_pixel_to_ray(u, v, W, H)
    best = np.full((H, W), -np.inf)
    vidx = np.full((H, W), -1, np.int16)
    PX = np.zeros((H, W), np.float32)
    PY = np.zeros((H, W), np.float32)
    for i, vw in enumerate(views):
        px, py, front = vw.ray_to_pixel(d)
        inside = front & (px >= -0.5) & (px <= vw.w - 0.5) & (py >= -0.5) & (py <= vw.h - 0.5)
        score = d @ vw.basis()[1]
        take = inside & (score > best)
        best[take] = score[take]
        vidx[take] = i
        PX[take] = px[take]
        PY[take] = py[take]
    return vidx, PX, PY


def fuse_to_equirect(per_view: np.ndarray, views: tuple[View, ...], W=2048, H=1024, fill=255) -> np.ndarray:
    """Nearest-neighbour fusion of per-view maps (N, h, w) into one equirectangular map (H, W)."""
    vidx, PX, PY = equirect_lookup(tuple(views), W, H)
    out = np.full((H, W) + per_view.shape[3:], fill, dtype=per_view.dtype)
    ok = vidx >= 0
    xi = np.clip(np.rint(PX[ok]).astype(int), 0, per_view.shape[2] - 1)
    yi = np.clip(np.rint(PY[ok]).astype(int), 0, per_view.shape[1] - 1)
    out[ok] = per_view[vidx[ok], yi, xi]
    return out


def render_views(eq, views, mode: str = "bilinear", pad: int = 8):
    """Render pinhole views from an equirectangular tensor (C, H, W) on any torch device.

    Each view pixel samples the panorama along its cam-frame ray (bilinear, horizontal wrap-around).
    Returns (N, C, h, w); all views must share h and w."""
    import torch
    import torch.nn.functional as F

    _, _, W = eq.shape
    dev = eq.device
    eqp = torch.cat([eq[..., -pad:], eq, eq[..., :pad]], dim=-1).unsqueeze(0)
    grids = []
    for v in views:
        ys, xs = torch.meshgrid(torch.arange(v.h, device=dev, dtype=torch.float64),
                                torch.arange(v.w, device=dev, dtype=torch.float64), indexing="ij")
        a = (xs + 0.5 - v.w / 2) / v.f
        b = (ys + 0.5 - v.h / 2) / v.f
        d = torch.stack([a, b, torch.ones_like(a)], dim=-1) @ torch.as_tensor(v.R_cam_from_view(), device=dev).T
        lam = torch.atan2(d[..., 0], d[..., 1])
        phi = torch.asin(torch.clamp(d[..., 2] / d.norm(dim=-1), -1, 1))
        u = (lam / (2 * math.pi) + 0.5) * W - 0.5 + pad
        grids.append(torch.stack([(u + 0.5) / (W + 2 * pad) * 2 - 1, -2 * phi / math.pi], dim=-1).float())
    grid = torch.stack(grids)
    src = eqp.expand(len(views), -1, -1, -1) if len(views) > 1 else eqp
    return F.grid_sample(src.float(), grid, mode=mode, padding_mode="border", align_corners=False)
