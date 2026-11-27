#!/usr/bin/env python
"""Regenerate the blocks between <!-- name:start --> and <!-- name:end --> markers in the README, the documents and
the project page, then rebuild the documentation pages (tools/pages.py).

    python tools/readme_tables.py [--dataset path/to/hf/folder]

block     file                 content
classes   docs/method.md       class, prompts or source model, mount, placement and group (streetnet/configs)
classes   docs/index.html      the same table as HTML, rows grouped by class group
params    docs/method.md       per-class fusion parameters (streetnet/configs/fusion_params.json)
nyc       README.md            the published New York City map: version, counts and F1 (the dataset's summary.json)
nyc       docs/index.html      the same with the panorama selection and the evaluation region
accuracy  docs/evaluation.md   per-class accuracy at 3 m with the median error (summary.json)
accuracy  docs/index.html      the same table as HTML
The nyc and accuracy blocks are written when --dataset is given.
"""
from __future__ import annotations

import argparse
import html
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
from streetnet.fusion import DEPTH_CLASSES, class_params  # noqa: E402

CFG = ROOT / "streetnet" / "configs"
M2F_ONLY = {"curb_ramp": "Mask2Former-Vistas: Curb Cut"}
M2F_ENS = {"storm_drain": "Mask2Former-Vistas: Catch Basin"}


def replace_block(path: Path, name: str, body: str):
    text = path.read_text()
    pat = re.compile(rf"(<!-- {name}:start -->\n).*?(<!-- {name}:end -->)", re.S)
    if not pat.search(text):
        raise SystemExit(f"{path}: no {name} markers")
    path.write_text(pat.sub(lambda m: m.group(1) + body.rstrip() + "\n" + m.group(2), text))


def f3(x):
    return "n/a" if x is None or x != x else f"{x:.3f}"


def class_rows() -> list[dict]:
    voc = json.loads((CFG / "prompts.json").read_text())
    cls = json.loads((CFG / "classes.json").read_text())
    rows = []
    for k, c in cls["classes"].items():
        prompts = [] if k in M2F_ONLY else list(voc[k]["prompts"])
        if k == "traffic_signal":
            prompts += voc["pedestrian_signal"]["prompts"]
        source = M2F_ONLY.get(k)
        if k in M2F_ENS:
            source = ", ".join(f'"{p}"' for p in prompts) + f" + {M2F_ENS[k]}"
        rows.append({"key": k, "name": c["name"], "experimental": bool(c.get("experimental")), "prompts": prompts,
                     "source": source, "m2f_ens": M2F_ENS.get(k),
                     "mount": "surface" if k in M2F_ONLY else voc[k]["mount"],
                     "placement": "range" if k in DEPTH_CLASSES else "ground", "group": c["group"],
                     "group_name": cls["groups"][c["group"]]})
    return rows


def classes_table() -> str:
    rows = ["| Class | SAM 3.1 prompts / source | Mount | Placement | Group |", "|---|---|---|---|---|"]
    for r in class_rows():
        src = r["source"] or ", ".join(f'"{p}"' for p in r["prompts"])
        name = r["name"] + (" (experimental)" if r["experimental"] else "")
        rows.append(f"| {name} | {src} | {r['mount']} | {r['placement']} | {r['group_name']} |")
    return "\n".join(rows)


def classes_html() -> str:
    out = ['<div class="table-wrap"><table>',
           "<thead><tr><th>Class</th><th>SAM 3.1 prompts / source</th><th>Mount</th><th>Placement</th></tr></thead><tbody>"]
    group = None
    for r in class_rows():
        if r["group"] != group:
            group = r["group"]
            out.append(f'<tr class="grp-row"><td colspan="4"><span class="sq" style="background:var(--g{group})"></span>'
                       f'{html.escape(r["group_name"])}</td></tr>')
        if r["m2f_ens"]:
            src = ", ".join(f"<code>{html.escape(p)}</code>" for p in r["prompts"]) + f" + {html.escape(r['m2f_ens'])}"
        elif r["source"]:
            src = html.escape(r["source"])
        else:
            src = ", ".join(f"<code>{html.escape(p)}</code>" for p in r["prompts"])
        tag = '<span class="tag">experimental</span>' if r["experimental"] else ""
        out.append(f'<tr><td class="cls">{html.escape(r["name"])}{tag}</td><td>{src}</td><td>{r["mount"]}</td>'
                   f'<td>{r["placement"]}</td></tr>')
    out.append("</tbody></table></div>")
    return "\n".join(out)


def params_table() -> str:
    P = json.loads((CFG / "fusion_params.json").read_text())
    cls = json.loads((CFG / "classes.json").read_text())["classes"]
    rows = ["| Class | Source | R_max (m) | r_c (m) | Clustering | score_thr | min_support | conf_thr | w_sig | sig_thr | Tuned | Dev F1 at 3 m |",
            "|---|---|---:|---:|---|---:|---:|---:|---:|---:|---|---:|"]
    for k in cls:
        p = class_params(P, k)
        dev = P["per_class"].get(k, {})
        sup = f"{p['min_support']}{'+occ' if p.get('use_occ') else ''}"
        ws = f"{p['w_sig']:g}" if p.get("w_sig", 0.0) > 0 else "0"
        st = f"{p['sig_thr']:g}" if p.get("w_sig", 0.0) > 0 else ""
        rows.append(f"| {cls[k]['name']} | {p['source']} | {p['R_max']:g} | {p['r_c']:g} | {p['method']} | "
                    f"{p['score_thr']:g} | {sup} | {p['conf_thr']:g} | {ws} | {st} | "
                    f"{'yes' if dev.get('tuned') else 'no'} | {f3(dev['dev_f1']) if 'dev_f1' in dev else ''} |")
    return "\n".join(rows)


def accuracy_data(S: dict) -> tuple[list[tuple], tuple]:
    cls = json.loads((CFG / "classes.json").read_text())["classes"]
    R3 = S["accuracy_R3"]
    rows = []
    for c in sorted((k for k in R3 if not k.startswith("_")), key=lambda k: -R3[k]["n_gt"]):
        v = R3[c]
        e = S["median_error_m"].get(c, {}).get("median")
        rows.append((cls[c]["name"], cls[c]["group"], v["n_gt"], v["n_pred"], v["precision"], v["recall"], v["f1"], e))
    a = R3["_all"]
    return rows, ("All classes", None, a["n_gt"], a["n_pred"], a["precision"], a["recall"], a["f1"], None)


def err(e):
    return "n/a" if e is None or e != e else f"{e:.2f}"


def accuracy_rows(S: dict, with_error: bool) -> list[str]:
    rows, total = accuracy_data(S)
    head = "| Class | Ground truth | Predicted | Precision | Recall | F1 |" + (" Median error (m) |" if with_error else "")
    out = [head, "|---|---:|---:|---:|---:|---:|" + ("---:|" if with_error else "")]
    for name, _, gt, pr, p, r, f, e in rows + [total]:
        out.append(f"| {name} | {gt:,} | {pr:,} | {f3(p)} | {f3(r)} | {f3(f)} |"
                   + ((f" {err(e)} |" if name != "All classes" else " |") if with_error else ""))
    return out


def accuracy_html(S: dict) -> str:
    rows, total = accuracy_data(S)
    out = ['<div class="table-wrap"><table>',
           '<thead><tr><th>Class</th><th class="n">Ground truth</th><th class="n">Predicted</th><th class="n">Precision</th>'
           '<th class="n">Recall</th><th class="n">F1</th><th class="n">Median error (m)</th></tr></thead><tbody>']
    for name, g, gt, pr, p, r, f, e in rows:
        out.append(f'<tr><td class="cls"><span class="sq" style="background:var(--g{g})"></span>{html.escape(name)}</td>'
                   f'<td class="n">{gt:,}</td><td class="n">{pr:,}</td><td class="n">{f3(p)}</td><td class="n">{f3(r)}</td>'
                   f'<td class="n">{f3(f)}</td><td class="n">{err(e)}</td></tr>')
    _, _, gt, pr, p, r, f, _ = total
    out.append(f'<tr class="total"><td>All classes</td><td class="n">{gt:,}</td><td class="n">{pr:,}</td><td class="n">{f3(p)}</td>'
               f'<td class="n">{f3(r)}</td><td class="n">{f3(f)}</td><td class="n"></td></tr>')
    out.append("</tbody></table></div>")
    return "\n".join(out)


def nyc_readme(S: dict) -> str:
    a = S["accuracy_R3"]["_all"]
    return (f"Version {S['version']}: {S['n_assets']:,} assets across {S['n_classes']} classes built from {S['n_panos']:,} panoramas "
            f"captured between {S['capture_years'][0]} and {S['capture_years'][1]}, achieving {f3(a['f1'])} F1 at a 3 m matching radius "
            f"over {a['n_gt']:,} NYC Open Data reference points "
            f"([per-class results](docs/evaluation.md#results)).")


def nyc_html(S: dict) -> str:
    a = S["accuracy_R3"]["_all"]
    e = html.escape
    return "\n".join([
        f"<p><strong>Version {e(S['version'])}</strong>: {S['n_assets']:,} assets across {S['n_classes']} classes from "
        f"{S['n_panos']:,} panoramas captured between {S['capture_years'][0]} and {S['capture_years'][1]}. {e(S['selection'])}</p>",
        f"<p>{e(S['region_note'])} Across {a['n_gt']:,} NYC Open Data reference assets in this region, the map achieves "
        f"{f3(a['precision'])} precision, {f3(a['recall'])} recall, and {f3(a['f1'])} F1 at a 3 m matching radius "
        f'(<a href="#accuracy">table</a>).</p>',
        "<p>The dataset distributes the fused asset map, municipal reference points, and evaluation region geometry; no imagery or panorama identifiers are included.</p>"])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", help="the exported dataset folder (tools/export_nyc_dataset.py --out)")
    a = ap.parse_args()
    method, evaluation, index, readme = (ROOT / "docs" / "method.md", ROOT / "docs" / "evaluation.md",
                                         ROOT / "docs" / "index.html", ROOT / "README.md")
    replace_block(method, "classes", classes_table())
    replace_block(method, "params", params_table())
    replace_block(index, "classes", classes_html())
    if a.dataset:
        S = json.loads((Path(a.dataset) / "summary.json").read_text())
        replace_block(readme, "nyc", nyc_readme(S))
        replace_block(index, "nyc", nyc_html(S))
        replace_block(evaluation, "accuracy", "\n".join(accuracy_rows(S, with_error=True)))
        replace_block(index, "accuracy", accuracy_html(S))
    import pages  # tools/pages.py
    pages.build()
    print("tables updated; documentation pages rebuilt")


if __name__ == "__main__":
    main()
