#!/usr/bin/env python
"""Regenerate the tables between <!-- name:start --> and <!-- name:end --> markers in README.md and docs/.

    python tools/readme_tables.py [--dataset path/to/hf/folder]

classes   README.md        class, prompts or source model, mount, placement and group (streetnet/configs)
params    docs/method.md   per-class fusion parameters (streetnet/configs/fusion_params.json)
nyc       README.md        the published New York City map: counts and accuracy (the dataset's summary.json)
accuracy  docs/evaluation.md   the same accuracy table with error statistics
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from streetnet.fusion import DEPTH_CLASSES, class_params  # noqa: E402

CFG = ROOT / "streetnet" / "configs"


def replace_block(path: Path, name: str, body: str):
    text = path.read_text()
    pat = re.compile(rf"(<!-- {name}:start -->\n).*?(<!-- {name}:end -->)", re.S)
    if not pat.search(text):
        raise SystemExit(f"{path}: no {name} markers")
    path.write_text(pat.sub(lambda m: m.group(1) + body.rstrip() + "\n" + m.group(2), text))


def f3(x):
    return "n/a" if x is None or x != x else f"{x:.3f}"


def classes_table() -> str:
    voc = json.loads((CFG / "prompts.json").read_text())
    cls = json.loads((CFG / "classes.json").read_text())
    rows = ["| Class | Prompts (SAM 3) or source | Mount | Placement | Group |", "|---|---|---|---|---|"]
    m2f = {"curb_ramp": "Mask2Former-Vistas: Curb Cut", "storm_drain": "Mask2Former-Vistas: Catch Basin"}
    for k, c in cls["classes"].items():
        src = m2f.get(k) or ", ".join(f'"{p}"' for p in voc[k]["prompts"])
        if k == "traffic_signal":
            src += ', "pedestrian walk signal"'
        mount = "surface" if k in m2f else voc[k]["mount"]
        pos = "range" if k in DEPTH_CLASSES else "ground"
        name = c["name"] + (" (experimental)" if c.get("experimental") else "")
        rows.append(f"| {name} | {src} | {mount} | {pos} | {cls['groups'][c['group']]} |")
    return "\n".join(rows)


def params_table() -> str:
    P = json.loads((CFG / "fusion_params.json").read_text())
    cls = json.loads((CFG / "classes.json").read_text())["classes"]
    rows = ["| Class | R_max (m) | r_c (m) | Clustering | Cluster position | score_thr | min_support | conf_thr | Tuned | Dev F1 at 3 m | Dev reference points |",
            "|---|---:|---:|---|---|---:|---:|---:|---|---:|---:|"]
    for k in cls:
        p = class_params(P, k)
        dev = P["per_class"].get(k, {})
        rows.append(f"| {cls[k]['name']} | {p['R_max']:g} | {p['r_c']:g} | {p['method']} | {p['pos']} | {p['score_thr']:g} | "
                    f"{p['min_support']} | {p['conf_thr']:g} | {'yes' if dev.get('tuned') else 'no'} | "
                    f"{f3(dev['dev_f1']) if 'dev_f1' in dev else ''} | "
                    f"{dev['dev_tp'] + dev['dev_fn'] if 'dev_tp' in dev else ''} |")
    return "\n".join(rows)


def accuracy_rows(S: dict, with_error: bool) -> list[str]:
    cls = json.loads((CFG / "classes.json").read_text())["classes"]
    R3 = S["accuracy_R3"]
    head = "| Class | Ground truth | Predicted | Precision | Recall | F1 |" + (" Median error (m) |" if with_error else "")
    rows = [head, "|---|---:|---:|---:|---:|---:|" + ("---:|" if with_error else "")]
    for c in sorted((k for k in R3 if not k.startswith("_")), key=lambda k: -R3[k]["n_gt"]):
        v = R3[c]
        e = S["median_error_m"].get(c, {}).get("median")
        rows.append(f"| {cls[c]['name']} | {v['n_gt']:,} | {v['n_pred']:,} | {f3(v['precision'])} | {f3(v['recall'])} | "
                    f"{f3(v['f1'])} |" + (f" {'n/a' if e is None or e != e else f'{e:.2f}'} |" if with_error else ""))
    a = R3["_all"]
    rows.append(f"| All classes | {a['n_gt']:,} | {a['n_pred']:,} | {f3(a['precision'])} | {f3(a['recall'])} | {f3(a['f1'])} |"
                + (" |" if with_error else ""))
    return rows


def nyc_block(S: dict) -> str:
    a = S["accuracy_R3"]["_all"]
    lines = [f"Version {S['version']}: {S['n_assets']:,} assets in {S['n_classes']} classes from {S['n_panos']:,} panoramas "
             f"captured between {S['capture_years'][0]} and {S['capture_years'][1]}. {S['selection']}",
             "",
             f"{S['region_note']} Over {a['n_gt']:,} ground-truth assets of NYC Open Data in that region, the map has "
             f"precision {f3(a['precision'])}, recall {f3(a['recall'])} and F1 {f3(a['f1'])} at a matching radius of 3 m:",
             ""]
    return "\n".join(lines + accuracy_rows(S, with_error=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", help="the exported dataset folder (tools/export_nyc_dataset.py --out)")
    a = ap.parse_args()
    replace_block(ROOT / "README.md", "classes", classes_table())
    replace_block(ROOT / "docs" / "method.md", "params", params_table())
    if a.dataset:
        S = json.loads((Path(a.dataset) / "summary.json").read_text())
        replace_block(ROOT / "README.md", "nyc", nyc_block(S))
        replace_block(ROOT / "docs" / "evaluation.md", "accuracy", "\n".join(accuracy_rows(S, with_error=True)))
    print("tables updated")


if __name__ == "__main__":
    main()
