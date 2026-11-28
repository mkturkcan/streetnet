"""Zero-shot multi-distractor SigLIP 2 candidate verifier.

For each fused candidate, renders clean tight (margin=0.45) and wide (margin=1.40) perspective crops
centered on its closest observing panorama detection (without bounding-box overlays) and scores the
cosine-similarity margin between class-specific positive prompts and hard negative urban distractors:
    sig2_comb = 0.65 * sig2_tight + 0.35 * sig2_wide
"""
from __future__ import annotations

import math
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from ..geometry import View, render_views

SIGLIP2_REPO = "google/siglip2-so400m-patch16-384"
TIGHT_MARGIN = 0.45
WIDE_MARGIN = 1.40
TIGHT_WEIGHT = 0.65
CROP_SIZE = 256

# Class-specific positive and hard-negative distractor prompt bank.
SIGLIP_PROMPTS: dict[str, tuple[list[str], list[str]]] = {
    "curb_ramp": (
        [
            "a photo of a pedestrian sidewalk curb ramp at a street corner crosswalk",
            "a photo of a wheelchair curb ramp with red or yellow tactile warning dome pad",
        ],
        [
            "a photo of a vehicle driveway entrance into a garage or parking lot",
            "a photo of a plain straight flat street curb and asphalt road",
            "a photo of a parked car next to a sidewalk",
            "a photo of a metal manhole cover or storm drain grate in the street",
        ],
    ),
    "fire_hydrant": (
        [
            "a photo of a street fire hydrant standing on a city sidewalk near the curb",
            "a photo of a cast iron fire hydrant with side nozzles on a sidewalk",
        ],
        [
            "a photo of a red fire alarm call box post on a street corner",
            "a photo of a plain metal or concrete sidewalk bollard post",
            "a photo of a public street trash can or garbage bin",
            "a photo of a fire department standpipe pipe attached to a building wall",
            "a photo of a parking meter kiosk on a sidewalk",
        ],
    ),
    "storm_drain": (
        [
            "a photo of a rectangular metal storm drain catch basin grate in a street gutter next to a curb",
            "a photo of a curbside street drainage inlet grate along the sidewalk edge",
        ],
        [
            "a photo of a round circular metal manhole cover in the middle of an asphalt road",
            "a photo of a small square utility valve box cover in the street",
            "a photo of plain asphalt street pavement or shadow next to a curb",
            "a photo of a sidewalk pedestrian curb ramp",
        ],
    ),
    "tree": (
        [
            "a photo of a street tree trunk growing in a city sidewalk tree pit",
            "a photo of a curbside urban street tree along a sidewalk",
        ],
        [
            "a photo of a metal street light pole or sign post",
            "a photo of a low bush, hedge, or potted planter box",
            "a photo of a building wall or scaffolding",
            "a photo of a parked car on a city street",
        ],
    ),
    "parking_meter": (
        [
            "a photo of a sidewalk multi-space parking pay station muni-meter kiosk with blue top",
            "a photo of a street curbside parking meter machine on a city sidewalk",
        ],
        [
            "a photo of a blue USPS street mailbox on a sidewalk",
            "a photo of a street trash can or litter basket",
            "a photo of a LinkNYC tall black digital Wi-Fi kiosk",
            "a photo of a metal street sign pole or traffic signal box",
            "a photo of a building storefront wall or doorway",
        ],
    ),
    "trash_can": (
        [
            "a photo of a public municipal wire mesh street litter basket on a sidewalk corner",
            "a photo of a public metal city sidewalk trash can",
        ],
        [
            "a photo of black plastic residential garbage bags piled on a sidewalk",
            "a photo of a private wheeled plastic household trash bin",
            "a photo of a newspaper vending box or mailbox",
            "a photo of a metal bollard or fire hydrant",
        ],
    ),
    "bus_shelter": (
        [
            "a photo of a public transit city bus stop shelter with glass panels, roof, and advertisement display",
            "a photo of a covered glass bus stop waiting shelter on a city sidewalk",
        ],
        [
            "a photo of a wooden outdoor restaurant dining shed cabin in the street",
            "a photo of sidewalk construction scaffolding and green shed",
            "a photo of a storefront glass awning or building entrance canopy",
            "a photo of a parked bus, van, or truck on the street",
        ],
    ),
    "kiosk": (
        [
            "a photo of a tall freestanding black and silver LinkNYC digital Wi-Fi sidewalk kiosk with screen",
            "a photo of a modern rectangular sidewalk digital information touchscreen kiosk",
        ],
        [
            "a photo of a glass bus stop shelter",
            "a photo of a short dark parking meter pay station",
            "a photo of a metal street sign pole or traffic signal pole",
            "a photo of a storefront doorway, ATM, or utility cabinet on a wall",
        ],
    ),
    "bike_rack": (
        [
            "a photo of a metal inverted-U city sidewalk bicycle parking rack",
            "a photo of a circular hoop sidewalk bike rack on concrete pavement",
        ],
        [
            "a photo of a Citi Bike blue bike-share docking station row",
            "a photo of a metal tree guard fence or building iron railing",
            "a photo of a sidewalk bollard or metal barricade",
            "a photo of a parked motorcycle, scooter, or car",
        ],
    ),
    "bench": (
        [
            "a photo of a public wooden or metal city sidewalk bench for sitting",
            "a photo of a street bench on a city sidewalk",
        ],
        [
            "a photo of building entrance stone steps or stoop",
            "a photo of a concrete planter wall or barrier",
            "a photo of outdoor restaurant tables and chairs",
            "a photo of a parked vehicle or car bumper",
        ],
    ),
    "fire_alarm_box": (
        [
            "a photo of a red cast iron street corner fire alarm call box post on a sidewalk",
            "a photo of a vintage red NYC FDNY street fire alarm telegraph pedestal box",
        ],
        [
            "a photo of a fire hydrant on a sidewalk",
            "a photo of a building wall water pipe or siamese standpipe connection",
            "a photo of a red traffic sign or mailbox",
            "a photo of a metal bollard or trash can",
        ],
    ),
    "traffic_sign": (
        [
            "a photo of an official metal street traffic, parking regulation, or street-name sign mounted on a pole",
            "a photo of a street parking rule sign, stop sign, or one way road sign",
        ],
        [
            "a photo of a traffic light signal head with red yellow green lights",
            "a photo of a commercial shop storefront business sign or restaurant awning",
            "a photo of an advertisement poster or billboard on a vehicle or bus shelter",
            "a photo of a bare building wall or window",
        ],
    ),
    "street_light": (
        [
            "a photo of a tall metal city street light lamp post with overhanging light arm",
            "a photo of a street lamp pole along a city sidewalk",
        ],
        [
            "a photo of a short thin street parking sign post",
            "a photo of a street tree trunk and branches",
            "a photo of a traffic signal mast pole without street lamp",
            "a photo of a building column or scaffolding pole",
        ],
    ),
    "sign_pole": (
        [
            "a photo of a thin metal street sign post standing on a sidewalk holding traffic signs",
            "a photo of a metal street parking sign pole on a curb",
        ],
        [
            "a photo of a tall thick street light lamp post",
            "a photo of a street tree trunk",
            "a photo of a sidewalk bollard or fence post",
            "a photo of a building facade or scaffolding pipe",
        ],
    ),
    "traffic_signal": (
        [
            "a photo of a traffic light signal head mounted on a street intersection pole",
            "a photo of a pedestrian walk signal light at a street crossing",
        ],
        [
            "a photo of a metal street parking regulation sign",
            "a photo of a street light lamp without traffic signal lights",
            "a photo of a storefront neon sign or window",
        ],
    ),
    "bus_stop_sign": (
        [
            "a photo of a blue and white city bus stop sign mounted on a sidewalk pole",
            "a photo of a transit bus route guide sign on a metal street pole",
        ],
        [
            "a photo of a no parking or street cleaning sign",
            "a photo of a commercial storefront sign or awning",
            "a photo of a glass bus stop shelter",
        ],
    ),
    "manhole": (
        [
            "a photo of a round circular metal utility manhole cover set flush in an asphalt road",
        ],
        [
            "a photo of a rectangular curbside storm drain catch basin grate",
            "a photo of a dark asphalt pavement patch or shadow",
        ],
    ),
    "bollard": (
        [
            "a photo of a short sturdy metal or concrete protective bollard post on a sidewalk",
        ],
        [
            "a photo of a fire hydrant on a sidewalk",
            "a photo of a tall street sign pole or street light",
            "a photo of a public street trash can",
        ],
    ),
    "barrier": (
        [
            "a photo of a street construction barricade, orange and white sawhorse, or concrete Jersey barrier",
        ],
        [
            "a photo of a parked car or delivery truck",
            "a photo of a permanent metal building fence or railing",
        ],
    ),
    "planter": (
        [
            "a photo of a freestanding sidewalk planter box or large pot containing plants or flowers",
        ],
        [
            "a photo of a street tree growing directly in a ground tree pit",
            "a photo of a street trash can or concrete barrier",
        ],
    ),
    "mailbox": (
        [
            "a photo of a freestanding blue postal collection mailbox standing on a city sidewalk",
        ],
        [
            "a photo of a sidewalk parking meter pay station",
            "a photo of a public street trash can or newspaper box",
        ],
    ),
    "bike_share_dock": (
        [
            "a photo of a row of public bike share docking stations with blue bicycles along a curb",
        ],
        [
            "a photo of an individual inverted-U sidewalk bicycle parking rack",
            "a photo of a parked car or motorcycle",
        ],
    ),
}


def crop_view(row, views: tuple[View, ...], size: int = CROP_SIZE, margin: float = TIGHT_MARGIN,
              min_fov: float = 4.0) -> View:
    """Pinhole view centered on a detection bounding box with relative padding `margin`."""
    src = views[int(row["view"])]
    cx = 0.5 * (float(row["x0"]) + float(row["x1"]))
    cy = 0.5 * (float(row["y0"]) + float(row["y1"]))
    d = src.pixel_to_ray(cx, cy)
    yaw = math.degrees(math.atan2(d[0], d[1]))
    pitch = math.degrees(math.asin(float(np.clip(d[2], -1.0, 1.0))))
    ext = max(float(row["x1"]) - float(row["x0"]) + 1.0, float(row["y1"]) - float(row["y0"]) + 1.0) * (1.0 + 2.0 * margin)
    fov = max(min_fov, math.degrees(2.0 * math.atan(ext / 2.0 / src.f)))
    return View(yaw=yaw, pitch=pitch, fov=min(fov, 100.0), w=size, h=size)


def render_crops_from_eq(eq, rows, views: tuple[View, ...], size: int = CROP_SIZE,
                         margin: float = TIGHT_MARGIN) -> list[np.ndarray]:
    """Render clean uint8 (size, size, 3) crops from a GPU equirectangular tensor (3, H, W) in [0, 1]."""
    rows = list(rows)
    if not rows:
        return []
    cviews = [crop_view(r, views, size=size, margin=margin) for r in rows]
    imgs = render_views(eq, cviews).clamp_(0, 1)
    return [(im * 255).byte().permute(1, 2, 0).cpu().numpy() for im in imgs]


class SigLIP2Verifier:
    """Multi-distractor zero-shot SigLIP 2 (SO400M) verifier."""

    def __init__(self, device: str = "cuda", repo: str = SIGLIP2_REPO,
                 prompts: dict[str, tuple[list[str], list[str]]] | None = None):
        import torch
        from transformers import AutoModel, AutoProcessor

        self.device = torch.device(device)
        self.repo = repo
        self.prompts = prompts or SIGLIP_PROMPTS
        self.proc = AutoProcessor.from_pretrained(repo)
        self.model = AutoModel.from_pretrained(repo, dtype=torch.float16).to(self.device).eval()
        self.txt_emb: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        with torch.inference_mode():
            for cls, (pos_list, neg_list) in self.prompts.items():
                all_t = pos_list + neg_list
                inp = self.proc(text=all_t, padding="max_length", max_length=64, return_tensors="pt").to(self.device)
                e = self.model.get_text_features(**inp)
                e = getattr(e, "pooler_output", e).float()
                e = torch.nn.functional.normalize(e, dim=-1).cpu().numpy()
                self.txt_emb[cls] = (e[:len(pos_list)], e[len(pos_list):])

    def _encode_images(self, images, batch_size: int = 64) -> np.ndarray:
        import torch

        embs = []
        with torch.inference_mode():
            for i in range(0, len(images), batch_size):
                x = self.proc(images=images[i:i + batch_size], return_tensors="pt")["pixel_values"].to(
                    self.device, torch.float16
                )
                e = self.model.get_image_features(pixel_values=x)
                e = getattr(e, "pooler_output", e).float()
                embs.append(torch.nn.functional.normalize(e, dim=-1).cpu().numpy())
        return np.concatenate(embs, axis=0) if embs else np.zeros((0, 1), np.float32)

    def _encode_tensors(self, tensors, batch_size: int = 128) -> np.ndarray:
        """Encode float/uint8 GPU or CPU tensors (N, 3, H, W) directly on GPU without PIL overhead."""
        import torch
        import torch.nn.functional as F

        if not len(tensors):
            return np.zeros((0, 1), np.float32)
        embs = []
        with torch.inference_mode():
            for i in range(0, len(tensors), batch_size):
                x = torch.stack(tensors[i:i + batch_size]).to(self.device, non_blocking=True)
                if x.dtype == torch.uint8:
                    x = x.float().div_(255.0)
                else:
                    x = x.float()
                if x.shape[-2:] != (384, 384):
                    x = F.interpolate(x, size=(384, 384), mode="bilinear", align_corners=False)
                x = ((x - 0.5) / 0.5).to(torch.float16)
                e = self.model.get_image_features(pixel_values=x)
                e = getattr(e, "pooler_output", e).float()
                embs.append(F.normalize(e, dim=-1).cpu().numpy())
        return np.concatenate(embs, axis=0) if embs else np.zeros((0, 1), np.float32)

    def score_crop_tensors(self, tight_tensors, wide_tensors, labels: list[str], batch_size: int = 128):
        """Return (sig2_tight, sig2_wide, sig2_comb) float32 arrays from (3, H, W) crop tensors."""
        et = self._encode_tensors(tight_tensors, batch_size=batch_size)
        ew = self._encode_tensors(wide_tensors, batch_size=batch_size)
        mt = np.zeros(len(labels), np.float32)
        mw = np.zeros(len(labels), np.float32)
        for i, cls in enumerate(labels):
            if cls not in self.txt_emb:
                continue
            ep, en = self.txt_emb[cls]
            mt[i] = float((et[i] @ ep.T).max() - (et[i] @ en.T).max())
            mw[i] = float((ew[i] @ ep.T).max() - (ew[i] @ en.T).max())
        comb = (TIGHT_WEIGHT * mt + (1.0 - TIGHT_WEIGHT) * mw).astype(np.float32)
        return mt, mw, comb

    def score_crops(self, tight_images, wide_images, labels: list[str], batch_size: int = 64):
        """Return (sig2_tight, sig2_wide, sig2_comb) float32 arrays aligned with `labels`."""
        et = self._encode_images(tight_images, batch_size=batch_size)
        ew = self._encode_images(wide_images, batch_size=batch_size)
        mt = np.zeros(len(labels), np.float32)
        mw = np.zeros(len(labels), np.float32)
        for i, cls in enumerate(labels):
            if cls not in self.txt_emb:
                continue
            ep, en = self.txt_emb[cls]
            mt[i] = float((et[i] @ ep.T).max() - (et[i] @ en.T).max())
            mw[i] = float((ew[i] @ ep.T).max() - (ew[i] @ en.T).max())
        comb = (TIGHT_WEIGHT * mt + (1.0 - TIGHT_WEIGHT) * mw).astype(np.float32)
        return mt, mw, comb


def score_candidates(
    assets: pd.DataFrame,
    index: pd.DataFrame,
    images,
    views: tuple[View, ...],
    device: str = "cuda",
    batch_size: int = 64,
    shard: str = "0/1",
    active_classes: set[str] | None = None,
    log=print,
) -> pd.DataFrame:
    """Compute SigLIP 2 tight, wide, and combined margins for candidates in `shard`.

    Requires `best_pano`, `label`, `view`, `x0`, `y0`, `x1`, `y1` columns in `assets`.
    When `active_classes` is given, only candidates whose `label` is in `active_classes` are scored.
    Returns a DataFrame with columns `asset_id`, `sig2_tight`, `sig2_wide`, `sig2_comb`.
    """
    import torch

    from ..index import image_path, load_image

    req = {"best_pano", "label", "view", "x0", "y0", "x1", "y1"}
    missing = req - set(assets.columns)
    if missing:
        raise ValueError(f"assets table missing required columns for SigLIP 2 verification: {sorted(missing)}")

    if active_classes is not None:
        assets = assets[assets["label"].astype(str).isin(active_classes)]
    if not len(assets):
        return pd.DataFrame(columns=["asset_id", "sig2_tight", "sig2_wide", "sig2_comb"])

    s_idx, n_shards = map(int, shard.split("/"))
    panos = sorted(p for p in assets["best_pano"].dropna().astype(str).unique() if p in index.index)
    my_panos = set(panos[s_idx::n_shards])
    sub = assets[assets["best_pano"].astype(str).isin(my_panos)].copy()
    if not len(sub):
        return pd.DataFrame(columns=["asset_id", "sig2_tight", "sig2_wide", "sig2_comb"])

    log(f"[verify] shard {shard}: {len(sub)} candidates across {len(my_panos)} panoramas")
    verifier = SigLIP2Verifier(device=device)
    pano_groups = list(sub.groupby("best_pano", sort=True))
    aids, mt_all, mw_all, mc_all = [], [], [], []
    buf_aid, buf_cls, buf_t, buf_w = [], [], [], []
    t0 = time.time()

    def flush():
        if not buf_aid:
            return
        mt, mw, mc = verifier.score_crop_tensors(buf_t, buf_w, buf_cls, batch_size=batch_size)
        aids.extend(buf_aid)
        mt_all.extend(mt.tolist())
        mw_all.extend(mw.tolist())
        mc_all.extend(mc.tolist())
        buf_aid.clear()
        buf_cls.clear()
        buf_t.clear()
        buf_w.clear()

    loader = ThreadPoolExecutor(6)
    futs = {}

    def prefetch(k):
        if k < len(pano_groups) and k not in futs:
            pid = str(pano_groups[k][0])
            futs[k] = loader.submit(lambda p: load_image(image_path(index, p, images)), pid)

    for k in range(min(8, len(pano_groups))):
        prefetch(k)
    for k, (_pid, grp) in enumerate(pano_groups):
        prefetch(k + 8)
        rgb = futs.pop(k).result()
        with torch.inference_mode():
            eq = torch.from_numpy(rgb).to(verifier.device, non_blocking=True).permute(2, 0, 1).float().div_(255)
            rows = [r for _, r in grp.iterrows()]
            vt = [crop_view(r, views, size=CROP_SIZE, margin=TIGHT_MARGIN) for r in rows]
            vw = [crop_view(r, views, size=CROP_SIZE, margin=WIDE_MARGIN) for r in rows]
            ct = (render_views(eq, vt).clamp_(0, 1) * 255).byte().cpu()
            cw = (render_views(eq, vw).clamp_(0, 1) * 255).byte().cpu()
            del eq
        for idx_r, r in enumerate(rows):
            buf_aid.append(int(r["asset_id"]) if "asset_id" in r else int(r.name))
            buf_cls.append(str(r["label"]))
            buf_t.append(ct[idx_r])
            buf_w.append(cw[idx_r])
        if len(buf_aid) >= 512:
            flush()
        if (k + 1) % 250 == 0 or (k + 1) == len(pano_groups):
            flush()
            el = time.time() - t0
            log(f"[verify] shard {shard}: {k + 1}/{len(pano_groups)} panos ({len(aids)} cands, {el:.1f}s)")
    loader.shutdown()
    return pd.DataFrame(dict(asset_id=np.asarray(aids, np.int64),
                             sig2_tight=np.asarray(mt_all, np.float32),
                             sig2_wide=np.asarray(mw_all, np.float32),
                             sig2_comb=np.asarray(mc_all, np.float32)))


def launch_verify(argv: list[str], gpus: list[str], work_dir: Path) -> int:
    """Run sharded SigLIP 2 verification across multiple GPUs, writing shard score files to `work_dir`."""
    work_dir.mkdir(parents=True, exist_ok=True)
    procs = []
    for i, g in enumerate(gpus):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=g)
        shard_out = work_dir / f"siglip_shard_{i}.parquet"
        cmd = [sys.executable, "-m", "streetnet", "verify", *argv,
               "--shard", f"{i}/{len(gpus)}", "--scores-out", str(shard_out)]
        logf = open(work_dir / f"verify_{i}.log", "a")
        procs.append((subprocess.Popen(cmd, env=env, stdout=logf, stderr=subprocess.STDOUT), logf))
        print(f"[verify] shard {i}/{len(gpus)} on GPU {g}: {work_dir / f'verify_{i}.log'}", flush=True)
    rc = 0
    for p, logf in procs:
        rc |= p.wait()
        logf.close()
    return rc
