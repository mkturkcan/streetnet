import json

import numpy as np
import pandas as pd
import pytest

from streetnet.cli import main
from streetnet.evaluate import evaluate, match
from streetnet.geometry import detection_views
from streetnet.instances import mask_record
from streetnet.vocabulary import DEFAULT_VOCABULARY, Vocabulary


def test_match_is_one_to_one_and_minimal():
    pred = np.array([[0.0, 0.0], [1.0, 0.0]])
    gt = np.array([[0.9, 0.0], [2.0, 0.0]])
    pi, gi, d = match(pred, gt, R=1.5)
    assert sorted(zip(pi.tolist(), gi.tolist())) == [(0, 0), (1, 1)]
    assert np.isclose(d.sum(), 1.9)


def test_region_rules():
    assets = pd.DataFrame(dict(label="tree", x=[0.0, 100.0, 5.0], y=[0.0, 0.0, 0.0]))
    gt = pd.DataFrame(dict(gt_key="tree", x=[0.5, 100.5, -8.0], y=[0.0, 0.0, 0.0]))
    res = evaluate(assets, gt, centers=np.array([[0.0, 0.0]]), radii=(3.0,), region_r=15.0)
    m = res["R3"]["tree"]
    assert (m["tp"], m["fp"], m["fn"]) == (1, 1, 1)          # the pair at x = 100 lies outside the region


def test_mask_record_contact_point_and_owner():
    views = detection_views(4)
    mask = np.zeros((1008, 1008), bool)
    mask[400:600, 480:530] = True
    r = mask_record(mask, views[1], 1, n_views=4, with_rle=False)
    assert (r["x0"], r["y0"], r["x1"], r["y1"]) == (480, 400, 529, 599)
    assert r["bot_py"] == 599.5 and r["owner"] and not r["cut_bottom"]
    assert 80 < r["bot_az"] < 100 and r["bot_el"] < 0


def test_vocabulary():
    voc = Vocabulary.load(DEFAULT_VOCABULARY)
    assert len(voc.prompts) == 34 and len(voc.mount_of) == 23
    assert voc.class_of["parking pay station"] == "parking_meter" and voc.rank_of["parking pay station"] == 1
    assert voc.mount_of["traffic_sign"] == "pole" and voc.mount_of["storm_drain"] == "surface"


def test_cli_map_and_eval(street, tmp_path, capsys):
    assert main(["map", "--index", str(street["index"]), "--run", str(street["run"]), "--name", "t", "--workers", "1"]) == 0
    a = pd.read_parquet(street["run"] / "map" / "t_assets.parquet")
    assert len(a) == 3
    gt = pd.DataFrame([dict(gt_key=lab, x=xy[0], y=xy[1]) for lab, _, xy in street["objects"]])
    gt.to_parquet(tmp_path / "gt.parquet")
    out = tmp_path / "metrics.json"
    assert main(["eval", "--index", str(street["index"]), "--map", str(street["run"] / "map" / "t_assets.parquet"),
                 "--gt", str(tmp_path / "gt.parquet"), "--out", str(out)]) == 0
    res = json.loads(out.read_text())
    assert res["R2"]["_all"]["f1"] == 1.0
    assert "3 disks of 15 m" in capsys.readouterr().out


def test_cli_requires_a_command():
    with pytest.raises(SystemExit):
        main([])
