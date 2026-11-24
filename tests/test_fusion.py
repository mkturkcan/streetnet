import json

import numpy as np
import pandas as pd

from streetnet.fusion import (
    associate_poles,
    cluster_greedy,
    fuse,
    fuse_with_params,
    localize,
    select_obs,
    strip_median,
    strip_samples,
)
from streetnet.geometry import detection_views
from streetnet.index import leveling, read_index
from streetnet.mapping import build_map, cap_min_support
from streetnet.vocabulary import DEFAULT_PARAMS


def _load(street):
    index = read_index(street["index"])
    dets = pd.concat([pd.read_parquet(f) for f in sorted((street["run"] / "det").glob("*.parquet"))], ignore_index=True)
    dets["obs_id"] = np.arange(len(dets))
    return index, dets


def test_ground_and_depth_localization_recover_positions(street):
    index, dets = _load(street)
    R = leveling(index, dets.pano_id.unique())
    truth = {lab: xy for lab, _, xy in street["objects"]}
    for est in ("ground", "depth"):
        obs = localize(dets, index, R, estimator=est)
        assert obs.loc_ok.all()
        for r in obs.itertuples():
            d = [np.hypot(r.X - xy[0], r.Y - xy[1]) for lab, _, xy in street["objects"] if lab == r.label]
            assert min(d) < 1e-6, (est, r.label, min(d))
    assert truth


def test_hybrid_uses_range_for_trees_only(street):
    index, dets = _load(street)
    obs = localize(dets, index, leveling(index, dets.pano_id.unique()), estimator="hybrid")
    assert set(obs.loc[obs.label == "tree", "loc_est"]) == {"depth"}
    assert set(obs.loc[obs.label == "fire_hydrant", "loc_est"]) == {"ground"}


def test_greedy_clustering_keeps_one_observation_per_panorama():
    xy = np.array([[0.0, 0.0], [0.5, 0.0], [0.2, 0.1]])
    lab = cluster_greedy(xy, np.ones(3), np.array(["a", "a", "b"]), r_c=2.0)
    assert lab[0] != lab[1]                       # same panorama: never one asset
    assert lab[2] in (lab[0], lab[1])


def test_fuse_support_position_and_confidence():
    obs = pd.DataFrame(dict(label="tree", X=[0.0, 1.0, 50.0], Y=[0.0, 0.0, 0.0], rng=[5.0, 10.0, 8.0],
                            score=[0.8, 0.5, 0.9], pano_id=["a", "b", "c"], obs_id=[0, 1, 2], bearing=0.0))
    assets, lab = fuse(obs, r_c=3.0)
    assert len(assets) == 2
    a = assets.sort_values("x").iloc[0]
    assert a.support == 2 and a.n_obs == 2
    assert np.isclose(a.conf, 1 - 0.2 * 0.5)
    assert a.x == 0.0                             # weighted median: the closer observation weighs 1/5 > 1/10


def test_pole_association_borrows_the_pole_contact():
    base = dict(pano_id="p", view=0, score=0.9, owner=True, dup=False, cut_bottom=False, bot_px=0.0, bot_py=0.0,
                bot_dx=0.0, bot_dy=1.0, bot_dz=0.0, bot_az=0.0, bot_el=0.0)
    sign = dict(base, obs_id=0, label="traffic_sign", mount="pole", x0=100, x1=140, y0=100, y1=140)
    pole = dict(base, obs_id=1, label="sign_pole", mount="ground", x0=115, x1=125, y0=90, y1=600, bot_px=120.0,
                bot_py=600.5, bot_dz=-0.5, bot_el=-26.0)
    far = dict(pole, obs_id=2, x0=600, x1=610, bot_px=605.0)
    out = associate_poles(pd.DataFrame([sign, pole, far]))
    s = out[out.obs_id == 0].iloc[0]
    assert s.loc_src == "pole" and s.pole_obs == 1 and s.bot_py == 600.5


def test_strip_samples_read_inside_the_mask():
    views = detection_views(4)
    D = np.full((512, 1024), 7.0, np.float32)
    D[:256] = 99.0                                # everything above the horizon is far
    v = strip_samples(D, np.array([0]), np.array([503.5]), np.array([900.0]), np.array([100.0]), views)
    assert np.allclose(strip_median(v), 7.0)


def test_parameters_and_cap():
    params = json.loads(DEFAULT_PARAMS.read_text())
    capped = cap_min_support(params, 1)
    assert max(v.get("min_support", 1) for v in capped["per_class"].values()) == 1
    assert params["per_class"]["tree"]["min_support"] == 2


def test_map_end_to_end(street):
    index = read_index(street["index"])
    params = json.loads(DEFAULT_PARAMS.read_text())
    assets, obs, summary = build_map(index, street["run"] / "det", params, workers=1, log=lambda s: None)
    hydrants = assets[assets.label == "fire_hydrant"].sort_values("y")
    assert len(hydrants) == 2 and hydrants.support.tolist() == [3, 1]     # the second is in range of one camera
    for (lab, _, xy) in street["objects"]:
        d = np.hypot(assets.x - xy[0], assets.y - xy[1])
        assert d[assets.label == lab].min() < 0.01
    tree = assets[assets.label == "tree"].iloc[0]
    assert tree.support == 3 and tree.estimator == "depth"
    assert summary["n_assets"] == 3 and set(obs.asset) <= set(assets.asset_id)


def test_select_obs_drops_duplicates_and_non_owners():
    df = pd.DataFrame(dict(owner=[True, False, True], dup=[False, False, True], score=[0.5, 0.9, 0.9]))
    assert len(select_obs(df)) == 1


def test_fuse_with_params_filters_support(street):
    index, dets = _load(street)
    obs = localize(dets, index, leveling(index, dets.pano_id.unique()))
    params = {"per_class": {"fire_hydrant": {"min_support": 3, "R_max": 25.0}}}
    _, assets = fuse_with_params(obs, params, ["fire_hydrant"])
    assert len(assets) == 1 and assets.support.iloc[0] == 3
