import math

import numpy as np
import pytest

from streetnet.geometry import (
    View,
    azimuth_of_ray,
    detection_views,
    equirect_pixel_to_ray,
    leveling_from_normal,
    metadata_normal,
    owner_view,
    ray_to_equirect_pixel,
    to_world,
)


def test_equirect_round_trip():
    W, H = 2048, 1024
    u = np.random.default_rng(0).uniform(0, W - 1, 1000)
    v = np.random.default_rng(1).uniform(1, H - 2, 1000)
    d = equirect_pixel_to_ray(u, v, W, H)
    assert np.allclose(np.linalg.norm(d, axis=-1), 1)
    u2, v2 = ray_to_equirect_pixel(d, W, H)
    assert np.allclose(np.mod(u2 - u + W / 2, W) - W / 2, 0, atol=1e-6)
    assert np.allclose(v2, v, atol=1e-6)


def test_centre_column_is_forward():
    d = equirect_pixel_to_ray(1023.5, 511.5, 2048, 1024)
    assert np.allclose(d, [0, 1, 0], atol=1e-12)


def test_view_round_trip_and_axes():
    v = View(yaw=90.0, pitch=-10.0, fov=90.0, w=1008, h=1008)
    px = np.array([0.0, 503.5, 1007.0])
    py = np.array([0.0, 503.5, 1007.0])
    rays = v.pixel_to_ray(px, py)
    qx, qy, front = v.ray_to_pixel(rays)
    assert front.all() and np.allclose(qx, px) and np.allclose(qy, py)
    centre = v.pixel_to_ray(503.5, 503.5)            # the view centre looks along yaw 90, pitch -10
    assert np.isclose(azimuth_of_ray(centre), 90.0)
    assert np.isclose(math.degrees(math.asin(centre[2])), -10.0)


def _covered(az_deg, el_deg, views):
    a, e = np.radians(az_deg), np.radians(el_deg)
    d = np.array([np.cos(e) * np.sin(a), np.cos(e) * np.cos(a), np.sin(e)])
    for v in views:
        px, py, front = v.ray_to_pixel(d)
        if front and -0.5 <= px <= v.w - 0.5 and -0.5 <= py <= v.h - 0.5:
            return True
    return False


def test_detection_views_cover_the_ground():
    """Four 90 degree views pitched down 10 degrees cover all azimuths from 3.6 to 45.3 degrees below the horizon
    (ground contacts from 2.5 to 40 m for a 2.5 m camera). The tightest azimuth is a seam between two views
    (45 degrees), where coverage starts 3.55 degrees below the horizon; about 1 degree of the horizon is missed."""
    views = detection_views(4)
    assert [v.yaw for v in views] == [0.0, 90.0, 180.0, 270.0]
    for az in np.arange(0.0, 360.0, 0.5):
        for el in (-45.2, -30.0, -10.0, -3.6):
            assert _covered(az, el, views), (az, el)
    assert not _covered(45.0, -3.5, views) and not _covered(45.0, 0.0, views)
    assert _covered(45.0, -3.56, views)
    horizon = np.mean([_covered(az, 0.0, views) for az in np.arange(0.0, 360.0, 0.05)])
    assert 0.98 < horizon < 1.0


@pytest.mark.parametrize("n", [4, 8])
def test_owner_view_sectors(n):
    step = 360 / n
    for i in range(n):
        assert owner_view(i * step, n) == i
        assert owner_view(i * step + step / 2 - 1e-6, n) == i
        assert owner_view(i * step - step / 2 + 1e-6, n) == i


def test_leveling_maps_normal_to_up():
    n = np.array([0.05, -0.08, 1.0])
    R = leveling_from_normal(n)
    assert np.allclose(R @ (n / np.linalg.norm(n)), [0, 0, 1])
    assert np.allclose(R @ R.T, np.eye(3))
    assert np.allclose(leveling_from_normal([0, 0, 1]), np.eye(3))


def test_metadata_normal_rule():
    n = metadata_normal([0.0, -2.0, 20.0, 0.0], [0.0, 1.0, 0.0, 359.0])
    assert np.allclose(n[0], [0, 0, 1])
    expected = np.array([math.sin(math.radians(-0.7)), math.sin(math.radians(2.0)), 1.0])
    assert np.allclose(n[1], expected / np.linalg.norm(expected))
    assert np.allclose(n[2], [0, 0, 1])                  # |tilt| > 12 degrees: treated as level
    assert n[3][0] > 0                                    # roll 359 = -1 degree


def test_to_world_heading():
    east, north = to_world(np.array([[0.0, 10.0, -2.5]]), 90.0, 100.0, 200.0)   # 10 m forward, heading east
    assert np.allclose([east[0], north[0]], [110.0, 200.0])


def test_render_views_matches_lookup():
    torch = pytest.importorskip("torch")
    from streetnet.geometry import render_views
    W, H = 512, 256
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    d = equirect_pixel_to_ray(u, v, W, H)
    eq = torch.as_tensor(np.stack([d[..., 0], d[..., 1], d[..., 2]]), dtype=torch.float32)
    view = View(yaw=45.0, pitch=0.0, fov=60.0, w=64, h=64)
    out = render_views(eq, [view])[0].numpy()             # each pixel samples its own ray direction
    want = view.pixel_to_ray(*np.meshgrid(np.arange(64), np.arange(64)))
    assert np.abs(out.transpose(1, 2, 0) - want).max() < 0.02
