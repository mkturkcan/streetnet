"""Per-panorama detection: one pass over a panorama with all three models.

GPU stage (one process per GPU)
    1. decode the equirectangular JPEG; render the detection views (4 x 1008 px by default)
    2. SAM 3 with all prompts of the vocabulary on all views, score > 0.25, lazy mask head
    3. duplicate flags: a mask overlapping a higher-scoring mask of the same class in the same view
       (IoU > 0.6) is flagged
    4. Mask2Former-Vistas: label map, Catch Basin panoptic segments
    5. MoGe-2 range of the views, fused into a 1024 x 512 equirectangular range map
CPU stage (a pool of worker processes)
    6. detection records (streetnet.instances) for SAM 3 masks and for the Mask2Former classes
       (Catch Basin segments, Curb Cut connected components, at least 40 px)
    7. routing: curb_ramp and storm_drain from Mask2Former, all other classes from SAM 3
    8. sign filter (streetnet.signs) with the label map fused to 2048 x 1024
    9. range: pole association, then the median of 5 range samples on a strip inside the mask directly above
       each contact pixel (column drng)
   10. write <out>/det/<pano_id>.parquet (last, so its presence marks the panorama as done)
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
from .geometry import View, detection_views, fuse_to_equirect
from .vocabulary import Vocabulary

FROM_MASK2FORMER = {9: "curb_ramp", 36: "storm_drain"}     # Curb Cut, Catch Basin -> class key
MIN_AREA = 40


# ----------------------------------------------------------------------------- run folder


def run_config(out: Path, n_views: int, vocabulary: Path, threshold: float) -> dict:
    """Read or create <out>/run.json, so that one run folder holds detections made with one setting."""
    out.mkdir(parents=True, exist_ok=True)
    f = out / "run.json"
    cfg = dict(views=dict(n=n_views, pitch=-10.0, fov=90.0, size=1008), threshold=threshold,
               vocabulary=json.loads(Path(vocabulary).read_text()), streetnet=__version__)
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


# ----------------------------------------------------------------------------- CPU stage


def _worker_init():
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import cv2
    cv2.setNumThreads(1)


def _write_parquet(df: pd.DataFrame, path: Path):
    tmp = path.with_name(f".{path.name}.tmp")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, path)


def finish(pid, packed, shape, meta, sem_v, conf_v, m2f_inst, R, views, out, keep_raw):
    """Records, routing, sign filter and range samples for one panorama; writes its files."""
    import cv2

    from .fusion import associate_poles, strip_median, strip_samples
    from .instances import empty_frame, mask_record, to_frame
    from .models.mask2former import CURB_CUT, VISTAS, components
    from .signs import apply_sign_filter

    out = Path(out)
    nv = len(views)
    rows, extra = [], []
    if len(meta):
        masks = np.unpackbits(packed, count=int(np.prod(shape))).reshape(shape).astype(bool)
        for m, md in zip(masks, meta):
            rec = mask_record(m, views[md["view"]], md["view"], n_views=nv)
            if rec is None:
                continue
            rec.update(pano_id=pid, model="sam3", label=md["label"], raw_label=md["raw_label"], score=md["score"])
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
    a = sam[~sam.label.isin(set(FROM_MASK2FORMER.values()))].assign(model="sam3")
    b = m2f[m2f.label.isin(set(FROM_MASK2FORMER.values()))].copy()
    b["mount"] = "surface"
    b["dup"] = False
    b["model"] = "m2f_vistas_pan"
    d = pd.concat([a, b], ignore_index=True)
    d = apply_sign_filter(d, sem, views).reset_index(drop=True)
    final = d.drop(columns=["rle"])
    q = final.copy()
    for c in ("cut_bottom", "cut_side", "owner"):
        q[c] = q[c].fillna(False).astype(bool)
    q["obs_id"] = np.arange(len(q))
    q = associate_poles(q)
    vals = strip_samples(R.astype(np.float32), q["view"].to_numpy(int), q["bot_px"].to_numpy(float),
                         q["bot_py"].to_numpy(float), (q["y1"] - q["y0"] + 1).to_numpy(float), views)
    final["drng"] = strip_median(vals) if len(q) else np.zeros(0)
    if keep_raw:
        _write_parquet(sam, out / "raw" / "sam3" / f"{pid}.parquet")
        _write_parquet(m2f, out / "raw" / "mask2former" / f"{pid}.parquet")
        p = out / "raw" / "labels" / f"{pid}.png"
        tmp = p.with_name(f".{pid}.tmp.png")
        cv2.imwrite(str(tmp), sem)
        os.replace(tmp, p)
    _write_parquet(final, out / "det" / f"{pid}.parquet")
    return pid, len(sam), len(m2f), len(final)


# ----------------------------------------------------------------------------- GPU stage


class Detector:
    """The three models on one GPU."""

    def __init__(self, views, vocabulary: Vocabulary, threshold: float = 0.25, compile: bool = False,
                 device: str = "cuda"):
        import torch

        from .models.mask2former import CATCH_BASIN, VistasSegmenter
        from .models.moge import RangeModel
        from .models.sam3 import Sam3Detector

        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        self.views = tuple(views)
        self.voc = vocabulary
        self.sam = Sam3Detector(device, threshold)
        if compile:
            self.sam.compile()
        self.seg = VistasSegmenter(device, instance_ids=(CATCH_BASIN,))
        self.rng = RangeModel(device)
        self.max_pairs = len(self.views) * len(vocabulary.prompts)
        self.bitw = torch.tensor([128, 64, 32, 16, 8, 4, 2, 1], dtype=torch.uint8, device=device)
        self.device = device

    def __call__(self, rgb: np.ndarray):
        """One panorama (H, W, 3) uint8 -> the payload of `finish` (without pid, views, out)."""
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
        return packed, shape, meta, seg["sem"], seg["conf"], seg["inst"], R


def run_shard(index: pd.DataFrame, images, out, order=None, shard: str = "0/1", n_views: int = 4,
              vocabulary=None, threshold: float = 0.25, compile: bool = False, workers: int = 8,
              keep_raw: bool = False, limit: int = 0, log=print):
    """Detect the panoramas of this shard that have no <out>/det/<pano_id>.parquet yet."""
    from .index import image_path, load_image
    from .vocabulary import DEFAULT_VOCABULARY

    out = Path(out)
    vocabulary = Path(vocabulary or DEFAULT_VOCABULARY)
    cfg = run_config(out, n_views, vocabulary, threshold)
    views = views_of(cfg)
    voc = Vocabulary.load(vocabulary)
    for sub in ["det"] + (["raw/sam3", "raw/mask2former", "raw/labels"] if keep_raw else []):
        (out / sub).mkdir(parents=True, exist_ok=True)
    ids = [p for p in (order or list(index.index)) if p in index.index]
    i, n = map(int, shard.split("/"))
    ids = ids[i::n]
    done = {f.stem for f in (out / "det").glob("*.parquet")}
    todo = [p for p in ids if p not in done]
    if limit:
        todo = todo[:limit]
    log(f"[detect] shard {shard}: {len(ids)} panoramas, {len(todo)} to do; {len(voc.prompts)} prompts x {len(views)} views")
    if not todo:
        return 0
    det = Detector(views, voc, threshold, compile)
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
                pid, ns, nm, nf = fut.result()
                n_done += 1
                if n_done % 25 == 0 or n_done < 5:
                    log(f"[detect] {n_done}/{len(todo)} {pid} sam3={ns} m2f={nm} kept={nf} gpu={tg:.2f}s "
                        f"avg={(time.time() - t0) / n_done:.3f}s/panorama")
            except Exception as e:  # noqa: BLE001
                n_fail += 1
                log(f"[detect] write failed: {e!r}\n{traceback.format_exc()}")

    for k in range(6):
        prefetch(k)
    for k, pid in enumerate(todo):
        prefetch(k + 6)
        try:
            rgb = futs.pop(k).result()
            tg = time.time()
            payload = det(rgb)
            tg = time.time() - tg
            pending.append((pool.submit(finish, pid, *payload, views, str(out), keep_raw), tg))
        except Exception as e:  # noqa: BLE001
            n_fail += 1
            log(f"[detect] failed {pid}: {e!r}\n{traceback.format_exc()}")
        drain(2 * workers)
    drain(0)
    pool.shutdown()
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
