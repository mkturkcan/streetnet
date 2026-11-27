"""Per-panorama detection: one pass over a panorama with SAM 3.1, Mask2Former-Vistas, and MoGe-3 (ViT-G).

GPU stage (one process per GPU)
    1. decode the equirectangular JPEG; render the detection views (4 x 1008 px by default)
    2. SAM 3.1 with all prompts of the vocabulary on all views, score > 0.22, lazy mask head
    3. duplicate flags: a mask overlapping a higher-scoring mask of the same class in the same view
       (IoU > 0.6) is flagged
    4. Mask2Former-Vistas: semantic/panoptic label map, Catch Basin panoptic segments, Curb Cut components
    5. MoGe-3 (ViT-G) metric range of the views, fused into a 1024 x 512 equirectangular range map
CPU stage (a pool of worker processes)
    6. detection records (streetnet.instances) for SAM 3.1 and Mask2Former classes
    7. zero-GT per-panorama road-plane self-calibration (`geometry.fit_road_plane_scale`) scaling the range map
    8. sign consistency filter (`streetnet.signs`), pole association, lower-trunk strip range (`drng`),
       vehicle-masked RLE-mask interior range (`d_mask`), base vehicle-occlusion fraction (`base_veh`),
       and storm-drain context features (`mh_frac`, `curb_near`)
    9. write compact zero-GIS crosswalk points (`cw/<pano_id>.npy`) and 128 x 256 packed occlusion mask
       (`occ/<pano_id>.bin`) for downstream mapping
   10. write `<out>/det/<pano_id>.parquet` (last, atomically marking the panorama as done)
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from . import __version__
from .geometry import (
    View,
    detection_views,
    extract_crosswalk_points,
    fit_road_plane_scale,
    fuse_to_equirect,
    leveling_from_normal,
    pack_occlusion_mask,
    ray_to_equirect_pixel,
)
from .models.moge import DEFAULT_DEPTH_MODEL
from .vocabulary import Vocabulary

FROM_MASK2FORMER = {9: "curb_ramp", 36: "storm_drain"}     # Curb Cut, Catch Basin -> class key
M2F_ONLY_CLASSES = {"curb_ramp"}                           # storm_drain keeps both SAM 3.1 and M2F for ensembling
MIN_AREA = 40
POLE_DEPTH_CLASSES = {"street_light", "sign_pole", "utility_pole"}
FRONT_PCTL_CLASSES = {"tree", "street_light", "sign_pole", "fire_hydrant", "parking_meter", "traffic_sign",
                      "fire_alarm_box"}
SURFACE_DEPTH_SKIP = {"curb_ramp", "storm_drain", "manhole", "crosswalk"}


# ----------------------------------------------------------------------------- run folder


def run_config(
    out: Path,
    n_views: int,
    vocabulary: Path,
    threshold: float,
    sam_checkpoint: str = "sam3.1",
    depth_model: str = DEFAULT_DEPTH_MODEL,
) -> dict:
    """Read or create <out>/run.json, so that one run folder holds detections made with one setting."""
    out.mkdir(parents=True, exist_ok=True)
    f = out / "run.json"
    cfg = dict(
        views=dict(n=n_views, pitch=-10.0, fov=90.0, size=1008),
        threshold=threshold,
        sam_checkpoint=sam_checkpoint,
        depth_model=depth_model,
        vocabulary=json.loads(Path(vocabulary).read_text()),
        streetnet=__version__,
    )
    if f.exists():
        old = json.loads(f.read_text())
        for k in ("views", "threshold", "vocabulary"):
            if old.get(k) != cfg[k]:
                raise SystemExit(f"{f}: '{k}' differs from this run's setting; use another output folder")
        return old
    tmp = f.with_name(f".run.json.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(cfg, indent=1))
    os.replace(tmp, f)
    return cfg


def views_of(cfg: dict) -> tuple[View, ...]:
    v = cfg["views"]
    return detection_views(v["n"], v["pitch"], v["fov"], v["size"])


def _choose_max_pairs(n_views: int, n_prompts: int, limit: int = 100) -> int:
    """Pick the largest divisor of n_views * n_prompts that does not exceed `limit` (keeps CUDA graph shapes fixed)."""
    total = n_views * n_prompts
    if total <= limit:
        return total
    for d in range(limit, 0, -1):
        if total % d == 0:
            return d
    return limit


# ----------------------------------------------------------------------------- CPU stage


def _worker_init():
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import cv2
    cv2.setNumThreads(1)


def _write_parquet(df: pd.DataFrame, path: Path):
    tmp = path.with_name(f".{path.name}.tmp")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, path)


def _write_bytes(data: bytes, path: Path):
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def _write_npy(arr: np.ndarray, path: Path):
    tmp = path.with_name(f".{path.stem}.tmp.npy")
    np.save(tmp, arr)
    os.replace(tmp, path)


def _extract_context_and_mask_depth(
    q: pd.DataFrame,
    R_cal: np.ndarray,
    sem: np.ndarray,
    views: tuple[View, ...],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute vehicle-masked RLE interior depth `d_mask`, base vehicle occlusion `base_veh`,
    panoptic manhole overlap `mh_frac`, and curb-edge adjacency `curb_near` for each row of `q`."""
    from .instances import decode_rle
    from .models.mask2former import BAD_DEPTH_LUT, CURB_SIDEWALK_LUT, MANHOLE, VEHICLE_LUT

    n = len(q)
    d_mask = q["drng"].to_numpy(np.float32).copy()
    base_veh = np.zeros(n, np.float32)
    mh_frac = np.zeros(n, np.float32)
    curb_near = np.zeros(n, np.float32)
    if not n:
        return d_mask, base_veh, mh_frac, curb_near

    Hd, Wd = R_cal.shape[:2]
    Hs, Ws = sem.shape[:2]
    rng_np = np.random.default_rng(0)

    v_arr = q["view"].to_numpy(int)
    x0 = q["x0"].to_numpy(float)
    x1 = q["x1"].to_numpy(float)
    y0 = q["y0"].to_numpy(float)
    y1 = q["y1"].to_numpy(float)
    bh = np.maximum(y1 - y0 + 1.0, 2.0)
    bw = np.maximum(x1 - x0 + 1.0, 2.0)
    labels = q["label"].to_numpy()
    rles = q["rle"].to_numpy() if "rle" in q.columns else [None] * n

    bx_b = np.linspace(0.2, 0.8, 7)
    by_b = np.linspace(0.0, 1.0, 3)
    BBX, BBY = np.meshgrid(bx_b, by_b)
    BBX, BBY = BBX.ravel(), BBY.ravel()

    for vi in np.unique(v_arr):
        idx = np.nonzero(v_arr == vi)[0]
        V = views[int(vi)]
        b_px = x0[idx][:, None] + BBX[None, :] * bw[idx][:, None]
        b_py = np.minimum(float(V.h - 1), y1[idx][:, None] + 1.0 + BBY[None, :] * np.maximum(8.0, 0.12 * bh[idx][:, None]))
        ub, vb = ray_to_equirect_pixel(V.pixel_to_ray(b_px, b_py), Ws, Hs)
        sb = sem[np.clip(np.rint(vb).astype(int), 0, Hs - 1), np.mod(np.rint(ub).astype(int), Ws)]
        base_veh[idx] = VEHICLE_LUT[sb].mean(axis=1)

    for i in range(n):
        lbl = labels[i]
        V = views[int(v_arr[i])]
        if lbl == "storm_drain":
            cpx = 0.5 * (x0[i] + x1[i])
            cpy = 0.5 * (y0[i] + y1[i])
            uc, vc = ray_to_equirect_pixel(V.pixel_to_ray(np.array([cpx]), np.array([cpy])), Ws, Hs)
            ui = int(round(float(uc[0]))) % Ws
            vi_s = int(np.clip(round(float(vc[0])), 0, Hs - 1))
            u_p = (np.arange(ui - 4, ui + 5)) % Ws
            v_p = np.clip(np.arange(vi_s - 4, vi_s + 5), 0, Hs - 1)
            mh_frac[i] = float((sem[np.ix_(v_p, u_p)] == MANHOLE).mean())
            u_r = (np.arange(ui - 24, ui + 25, 3)) % Ws
            v_r = np.clip(np.arange(vi_s - 18, vi_s + 19, 3), 0, Hs - 1)
            curb_near[i] = float(CURB_SIDEWALK_LUT[sem[np.ix_(v_r, u_r)]].mean())
        rle = rles[i]
        if lbl in SURFACE_DEPTH_SKIP or not isinstance(rle, str) or not rle:
            continue
        m = decode_rle(rle, V.h, V.w)
        ys, xs = np.nonzero(m)
        if len(ys) < 6:
            continue
        if lbl in POLE_DEPTH_CLASSES:
            sel = ys >= (y1[i] - 0.40 * bh[i])
        else:
            sel = (ys >= y0[i] + 0.15 * bh[i]) & (ys <= y1[i] - 0.08 * bh[i])
        if sel.sum() >= 6:
            ys, xs = ys[sel], xs[sel]
        if len(ys) > 80:
            k = rng_np.choice(len(ys), 80, replace=False)
            ys, xs = ys[k], xs[k]
        rays = V.pixel_to_ray(xs.astype(float), ys.astype(float))
        ud, vd = ray_to_equirect_pixel(rays, Wd, Hd)
        dv = R_cal[np.clip(np.rint(vd).astype(int), 0, Hd - 1), np.mod(np.rint(ud).astype(int), Wd)]
        use, vse = ray_to_equirect_pixel(rays, Ws, Hs)
        sl = sem[np.clip(np.rint(vse).astype(int), 0, Hs - 1), np.mod(np.rint(use).astype(int), Ws)]
        dv = dv[~BAD_DEPTH_LUT[sl]]
        dv = dv[np.isfinite(dv) & (dv > 0.5) & (dv < 60.0)]
        if len(dv) >= 4:
            pctl = 30 if lbl in FRONT_PCTL_CLASSES else 45
            d_mask[i] = float(np.percentile(dv, pctl))

    return d_mask, base_veh, mh_frac, curb_near


def finish(pid, packed, shape, meta, sam_tag, sem_v, conf_v, m2f_inst, R, pose, views, out, keep_raw):
    """Records, road-plane self-calibration, sign filter, context features, and atomic writes for one panorama."""
    import cv2

    from .fusion import associate_poles, strip_median, strip_samples
    from .instances import empty_frame, mask_record, to_frame
    from .models.mask2former import CURB_CUT, OCCLUDING_LUT, ROAD_LUT, VISTAS, components
    from .signs import apply_sign_filter

    out = Path(out)
    nv = len(views)
    px, py, heading, R_lev, h_cam = pose

    rows, extra = [], []
    if len(meta):
        masks = np.unpackbits(packed, count=int(np.prod(shape))).reshape(shape).astype(bool)
        for m, md in zip(masks, meta):
            rec = mask_record(m, views[md["view"]], md["view"], n_views=nv)
            if rec is None:
                continue
            rec.update(pano_id=pid, model=sam_tag, label=md["label"], raw_label=md["raw_label"], score=md["score"])
            rows.append(rec)
            extra.append((md["mount"], md["dup"], md["prompt_rank"]))
    sam = to_frame(rows) if rows else empty_frame()
    sam["mount"] = pd.Series([e[0] for e in extra], dtype=object)
    sam["dup"] = pd.Series([e[1] for e in extra], dtype=bool)
    sam["prompt_rank"] = pd.Series([e[2] for e in extra], dtype="int16")

    rows = []
    for v in range(nv):
        for mask, c, s in list(m2f_inst[v]) + components(sem_v[v], conf_v[v], [CURB_CUT], MIN_AREA):
            if mask.sum() < MIN_AREA:
                continue
            rec = mask_record(mask, views[v], v, n_views=nv)
            if rec is None:
                continue
            rec.update(pano_id=pid, model="m2f_vistas_pan", label=FROM_MASK2FORMER[c], raw_label=VISTAS[c], score=float(s))
            rows.append(rec)
    m2f = to_frame(rows)

    sem = fuse_to_equirect(sem_v, views, W=2048, H=1024, fill=255)
    # Zero-GT road-plane self-calibration of monocular metric range.
    R_f = R.astype(np.float32)
    alpha = fit_road_plane_scale(R_f, ROAD_LUT[sem], R_lev, h=float(h_cam))
    R_cal = R_f * np.float32(alpha)

    a = sam[~sam.label.isin(M2F_ONLY_CLASSES)].assign(model=sam_tag)
    b = m2f[m2f.label.isin(set(FROM_MASK2FORMER.values()))].copy()
    b["mount"] = "surface"
    b["dup"] = False
    b["prompt_rank"] = pd.Series(np.full(len(b), -1, dtype=np.int16))
    b["model"] = "m2f_vistas_pan"
    d = pd.concat([a, b], ignore_index=True)
    d = apply_sign_filter(d, sem, views).reset_index(drop=True)

    q = d.copy()
    for c in ("cut_bottom", "cut_side", "owner"):
        q[c] = q[c].fillna(False).astype(bool)
    q["obs_id"] = np.arange(len(q))
    q = associate_poles(q)
    if len(q):
        vals = strip_samples(R_cal, q["view"].to_numpy(int), q["bot_px"].to_numpy(float),
                             q["bot_py"].to_numpy(float), (q["y1"] - q["y0"] + 1).to_numpy(float), views)
        drng = strip_median(vals).astype(np.float32)
    else:
        drng = np.zeros(0, np.float32)
    q["drng"] = drng
    d_mask, base_veh, mh_frac, curb_near = _extract_context_and_mask_depth(q, R_cal, sem, views)

    final = d.drop(columns=["rle"])
    final["drng"] = drng
    final["d_mask"] = d_mask
    final["base_veh"] = base_veh
    final["mh_frac"] = mh_frac
    final["curb_near"] = curb_near
    final["road_alpha"] = np.float32(alpha)

    # Compact sidecars for zero-GIS crosswalk priors and occlusion-aware clear-view counting.
    cw_pts = extract_crosswalk_points(sem, px, py, heading, R_lev, h=float(h_cam))
    if len(cw_pts):
        _write_npy(cw_pts, out / "cw" / f"{pid}.npy")
    _write_bytes(pack_occlusion_mask(sem, OCCLUDING_LUT), out / "occ" / f"{pid}.bin")

    if keep_raw:
        _write_parquet(sam, out / "raw" / "sam3" / f"{pid}.parquet")
        _write_parquet(m2f, out / "raw" / "mask2former" / f"{pid}.parquet")
        p = out / "raw" / "labels" / f"{pid}.png"
        tmp = p.with_name(f".{pid}.tmp.png")
        cv2.imwrite(str(tmp), sem)
        os.replace(tmp, p)
    _write_parquet(final, out / "det" / f"{pid}.parquet")
    return pid, len(sam), len(m2f), len(final), float(alpha)


# ----------------------------------------------------------------------------- GPU stage


class Detector:
    """SAM 3.1, Mask2Former-Vistas, and MoGe-3 (ViT-G) on one GPU."""

    def __init__(
        self,
        views,
        vocabulary: Vocabulary,
        threshold: float = 0.22,
        compile: bool = False,
        device: str = "cuda",
        sam_checkpoint: str = "sam3.1",
        depth_model: str = DEFAULT_DEPTH_MODEL,
    ):
        import torch

        from .models.mask2former import CATCH_BASIN, VistasSegmenter
        from .models.moge import RangeModel
        from .models.sam3 import Sam3Detector

        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        self.views = tuple(views)
        self.voc = vocabulary
        self.sam = Sam3Detector(device, threshold, checkpoint=sam_checkpoint)
        if compile:
            self.sam.compile()
        self.seg = VistasSegmenter(device, instance_ids=(CATCH_BASIN,))
        self.rng = RangeModel(device, model_name=depth_model)
        self.max_pairs = _choose_max_pairs(len(self.views), len(vocabulary.prompts), limit=100)
        self.bitw = torch.tensor([128, 64, 32, 16, 8, 4, 2, 1], dtype=torch.uint8, device=device)
        self.device = device

    def __call__(self, rgb: np.ndarray):
        """One panorama (H, W, 3) uint8 -> the payload of `finish` (without pid, pose, views, out)."""
        import torch

        from .geometry import render_views
        from .models.moge import fuse_range
        from .models.sam3 import duplicate_flags

        voc, views = self.voc, self.views
        with torch.inference_mode():
            eq = torch.from_numpy(rgb).to(self.device, non_blocking=True).permute(2, 0, 1).float().div_(255)
            V = render_views(eq, views).clamp_(0, 1)
            del eq
            res = self.sam.detect(V, voc.prompts, max_pairs=self.max_pairs, lazy_masks=True)
            masks, scores, meta = [], [], []
            for vi in range(len(views)):
                for p in voc.prompts:
                    r = res[vi][p]
                    if r is None or r["scores"].numel() == 0:
                        continue
                    masks.append(r["masks"])
                    scores.append(r["scores"].float())
                    cls = voc.class_of[p]
                    for s in r["scores"].float().cpu().numpy():
                        meta.append(dict(view=vi, label=cls, raw_label=p, score=float(s), mount=voc.mount_of[cls],
                                         prompt_rank=voc.rank_of[p]))
            if meta:
                M = torch.cat(masks)
                dup = duplicate_flags(M, torch.cat(scores), [m["label"] for m in meta], [m["view"] for m in meta], 0.6)
                for m, dd in zip(meta, dup):
                    m["dup"] = bool(dd)
                shape = tuple(M.shape)
                flat = M.reshape(-1)
                pad = (-flat.numel()) % 8
                if pad:
                    flat = torch.cat([flat, flat.new_zeros(pad)])
                packed = (flat.view(-1, 8).to(torch.uint8) * self.bitw).sum(1, dtype=torch.uint8).cpu().numpy()
                del M, flat
            else:
                packed, shape = np.zeros(0, np.uint8), (0, views[0].h, views[0].w)
            del res, masks, scores
            seg = self.seg.predict(V)
            R = fuse_range(self.rng.predict(V, views[0].fov), views)
            del V
        return packed, shape, meta, self.sam.model_tag, seg["sem"], seg["conf"], seg["inst"], R


def run_shard(
    index: pd.DataFrame,
    images,
    out,
    order=None,
    shard: str = "0/1",
    n_views: int = 4,
    vocabulary=None,
    threshold: float = 0.22,
    compile: bool = False,
    workers: int = 8,
    keep_raw: bool = False,
    limit: int = 0,
    sam_checkpoint: str = "sam3.1",
    depth_model: str = DEFAULT_DEPTH_MODEL,
    log=print,
):
    """Detect the panoramas of this shard that have no <out>/det/<pano_id>.parquet yet."""
    from .index import image_path, load_image
    from .vocabulary import DEFAULT_VOCABULARY

    out = Path(out)
    vocabulary = Path(vocabulary or DEFAULT_VOCABULARY)
    cfg = run_config(out, n_views, vocabulary, threshold, sam_checkpoint=sam_checkpoint, depth_model=depth_model)
    views = views_of(cfg)
    voc = Vocabulary.load(vocabulary)
    for sub in ["det", "cw", "occ"] + (["raw/sam3", "raw/mask2former", "raw/labels"] if keep_raw else []):
        (out / sub).mkdir(parents=True, exist_ok=True)
    ids = [p for p in (order or list(index.index)) if p in index.index]
    i, n = map(int, shard.split("/"))
    ids = ids[i::n]
    done = {f.stem for f in (out / "det").glob("*.parquet")}
    todo = [p for p in ids if p not in done]
    if limit:
        todo = todo[:limit]
    log(f"[detect] shard {shard}: {len(ids)} panoramas, {len(todo)} to do; "
        f"{len(voc.prompts)} prompts x {len(views)} views ({sam_checkpoint} + {depth_model})")
    if not todo:
        return 0
    det = Detector(views, voc, threshold, compile, sam_checkpoint=sam_checkpoint, depth_model=depth_model)
    loader = ThreadPoolExecutor(4)
    import multiprocessing as mp
    pool = ProcessPoolExecutor(workers, mp_context=mp.get_context("spawn"), initializer=_worker_init)
    futs, pending = {}, []
    t0 = time.time()
    n_done = n_fail = 0

    def prefetch(k):
        if k < len(todo) and k not in futs:
            futs[k] = loader.submit(lambda p: load_image(image_path(index, p, images)), todo[k])

    def drain(limit_):
        nonlocal n_done, n_fail
        while len(pending) > limit_ or (pending and pending[0][0].done()):
            fut, tg = pending.pop(0)
            try:
                pid, ns, nm, nf, alpha = fut.result()
                n_done += 1
                if n_done % 25 == 0 or n_done < 5:
                    log(f"[detect] {n_done}/{len(todo)} {pid} sam={ns} m2f={nm} kept={nf} alpha={alpha:.3f} "
                        f"gpu={tg:.2f}s avg={(time.time() - t0) / n_done:.3f}s/panorama")
            except Exception as e:  # noqa: BLE001
                n_fail += 1
                log(f"[detect] write failed: {e!r}\n{traceback.format_exc()}")

    for k in range(6):
        prefetch(k)
    for k, pid in enumerate(todo):
        prefetch(k + 6)
        try:
            rgb = futs.pop(k).result()
            row = index.loc[pid]
            R_lev = leveling_from_normal([float(row["nx"]), float(row["ny"]), float(row["nz"])])
            pose = (float(row["x"]), float(row["y"]), float(row["heading"]), R_lev, float(row["h"]))
            tg = time.time()
            payload = det(rgb)
            tg = time.time() - tg
            pending.append((pool.submit(finish, pid, *payload, pose, views, str(out), keep_raw), tg))
        except Exception as e:  # noqa: BLE001
            n_fail += 1
            log(f"[detect] failed {pid}: {e!r}\n{traceback.format_exc()}")
        drain(2 * workers)
    drain(0)
    pool.shutdown()
    loader.shutdown()
    el = time.time() - t0
    log(f"[detect] done: {n_done} panoramas ({n_fail} failed) in {el:.0f}s, {el / max(1, n_done):.3f}s per panorama")
    return n_fail


def launch(argv: list[str], gpus: list[str], out: Path, per_gpu: int = 1) -> int:
    """Run `streetnet detect <argv>` once per GPU slot as shard i/n; logs in <out>/logs/detect_<i>.log."""
    (out / "logs").mkdir(parents=True, exist_ok=True)
    slots = [g for g in gpus for _ in range(per_gpu)]
    procs = []
    for i, g in enumerate(slots):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=g)
        cmd = [sys.executable, "-m", "streetnet", "detect", *argv, "--shard", f"{i}/{len(slots)}"]
        logf = open(out / "logs" / f"detect_{i}.log", "a")
        procs.append((subprocess.Popen(cmd, env=env, stdout=logf, stderr=subprocess.STDOUT), logf))
        print(f"[detect] shard {i}/{len(slots)} on GPU {g}: {out / 'logs' / f'detect_{i}.log'}", flush=True)
    rc = 0
    for p, logf in procs:
        rc |= p.wait()
        logf.close()
    return rc
