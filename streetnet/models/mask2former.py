"""Mask2Former Swin-L trained on Mapillary Vistas v1.2 (panoptic), batched over the views of a panorama.

`predict(views)` returns per view:
    sem    native Vistas label map (uint8): panoptic segments where assigned, semantic argmax elsewhere
    conf   per-pixel semantic confidence x 255 (uint8)
    inst   panoptic segments of the instance classes in `instance_ids` as (mask, native id, score)
Post-processing re-implements the Hugging Face / detectron2 inference on the GPU:
    semantic  seg[c] = sum_q softmax(class_q)[c] * sigmoid(mask_q); label = argmax_c, conf = max / sum
    panoptic  queries with class != no-object and probability > 0.5; each pixel is assigned to the query with the
              largest probability x mask probability; a query keeps a segment when the pixels assigned to it number
              more than 0.8 of its mask area (pixels with mask >= 0.5); the segment is its assigned pixels with
              mask >= 0.5, and its score is the query's class probability
StreetNet uses this model for road-surface assets (Catch Basin panoptic segments and Curb Cut connected
components) and scene semantics (road-plane self-calibration, crosswalk priors, vehicle occlusion, manhole
exclusion, and traffic-sign consistency filtering).
"""
from __future__ import annotations

import cv2
import numpy as np
import torch
import torch.nn.functional as F

REPO = "facebook/mask2former-swin-large-mapillary-vistas-panoptic"
SIZE = 1024
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)

# Mapillary Vistas v1.2 classes in the model's id2label order.
VISTAS = [
    "Bird", "Ground Animal", "Curb", "Fence", "Guard Rail", "Barrier", "Wall", "Bike Lane", "Crosswalk - Plain",
    "Curb Cut", "Parking", "Pedestrian Area", "Rail Track", "Road", "Service Lane", "Sidewalk", "Bridge",
    "Building", "Tunnel", "Person", "Bicyclist", "Motorcyclist", "Other Rider", "Lane Marking - Crosswalk",
    "Lane Marking - General", "Mountain", "Sand", "Sky", "Snow", "Terrain", "Vegetation", "Water", "Banner",
    "Bench", "Bike Rack", "Billboard", "Catch Basin", "CCTV Camera", "Fire Hydrant", "Junction Box", "Mailbox",
    "Manhole", "Phone Booth", "Pothole", "Street Light", "Pole", "Traffic Sign Frame", "Utility Pole",
    "Traffic Light", "Traffic Sign (Back)", "Traffic Sign (Front)", "Trash Can", "Bicycle", "Boat", "Bus",
    "Car", "Caravan", "Motorcycle", "On Rails", "Other Vehicle", "Trailer", "Truck", "Wheeled Slow",
    "Car Mount", "Ego Vehicle",
]
CURB_CUT, CATCH_BASIN, MANHOLE = 9, 36, 41
TRAFFIC_LIGHT = {48}
TRAFFIC_SIGN = {46, 49, 50}
VEHICLE = {54, 55, 56, 57, 58, 59, 60, 61, 62}
ROAD_SURFACE = {7, 8, 13, 14, 23, 24}
CROSSWALK = {8, 23}
CURB_SIDEWALK = {2, 9, 15}
OCCLUDING_STRUCT = {3, 4, 5, 6, 17}
EGO_IGNORE = {63, 64, 255}
BAD_DEPTH = {27} | VEHICLE | EGO_IGNORE


def _lut(ids) -> np.ndarray:
    a = np.zeros(256, dtype=bool)
    a[list(ids)] = True
    return a


ROAD_LUT = _lut(ROAD_SURFACE)
VEHICLE_LUT = _lut(VEHICLE)
CROSSWALK_LUT = _lut(CROSSWALK)
CURB_SIDEWALK_LUT = _lut(CURB_SIDEWALK)
BAD_DEPTH_LUT = _lut(BAD_DEPTH)
OCCLUDING_LUT = _lut(VEHICLE | OCCLUDING_STRUCT | EGO_IGNORE)


class VistasSegmenter:
    def __init__(self, device: str = "cuda", instance_ids=(CATCH_BASIN,), dtype=torch.bfloat16,
                 pan_threshold: float = 0.5, overlap_threshold: float = 0.8):
        from transformers import Mask2FormerForUniversalSegmentation

        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        self.device = torch.device(device)
        self.dtype = dtype
        self.instance_ids = sorted(instance_ids)
        self.pan_threshold, self.overlap_threshold = pan_threshold, overlap_threshold
        self.model = Mask2FormerForUniversalSegmentation.from_pretrained(REPO)
        assert len(self.model.config.id2label) == len(VISTAS)
        self.num_classes = len(VISTAS)
        self.model.to(self.device).eval()
        self.mean = torch.tensor(MEAN, device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor(STD, device=self.device).view(1, 3, 1, 1)

    def _prep(self, views: torch.Tensor) -> torch.Tensor:
        x = views.to(self.device)
        if x.shape[-1] != SIZE or x.shape[-2] != SIZE:
            x = F.interpolate(x, (SIZE, SIZE), mode="bilinear", align_corners=False, antialias=x.shape[-1] > SIZE)
        return (x - self.mean) / self.std

    def _semantic(self, cls, msk, hw):
        prob = cls.softmax(-1)[..., :-1]                             # drop the no-object class
        seg = torch.einsum("qc,qhw->chw", prob, msk.sigmoid())
        seg = F.interpolate(seg[None], hw, mode="bilinear", align_corners=False)[0]
        mx, lab = seg.max(0)
        return lab, mx / seg.sum(0).clamp_min(1e-6)

    def _panoptic(self, cls, msk, hw):
        scores, labels = cls.softmax(-1).max(-1)
        keep = (labels != self.num_classes) & (scores > self.pan_threshold)
        mp = F.interpolate(msk[None], hw, mode="bilinear", align_corners=False)[0].sigmoid_()
        mid = ((scores * keep)[:, None, None] * mp).argmax(0)
        orig_area = (mp >= 0.5).flatten(1).sum(1).float()
        at_mid = torch.gather(mp, 0, mid[None])[0] >= 0.5
        del mp
        Q, flat = len(scores), mid.flatten()
        mid_area = torch.zeros(Q, device=self.device).scatter_add_(0, flat, torch.ones_like(flat, dtype=torch.float))
        fin_area = torch.zeros(Q, device=self.device).scatter_add_(0, flat, at_mid.flatten().float())
        ok = keep & (mid_area > 0) & (orig_area > 0) & (fin_area > 0) & (mid_area > self.overlap_threshold * orig_area)
        pix = at_mid & ok[mid]
        return torch.where(pix, labels[mid], -1), torch.where(pix, mid, -1).to(torch.int16), ok, labels, scores

    @torch.inference_mode()
    def predict(self, views: torch.Tensor) -> dict:
        """views: (N, 3, h, w) float in [0, 1] on the GPU. Returns numpy sem, conf (N, h, w) and inst."""
        hw = tuple(views.shape[-2:])
        x = self._prep(views)
        N = len(x)
        sem = torch.empty((N,) + hw, dtype=torch.uint8, device=self.device)
        conf = torch.empty((N,) + hw, dtype=torch.uint8, device=self.device)
        seg_ids = torch.empty((N,) + hw, dtype=torch.int16, device=self.device)
        with torch.autocast("cuda", dtype=self.dtype, enabled=self.dtype != torch.float32):
            out = self.model(pixel_values=x)
        cls, msk = out.class_queries_logits.float(), out.masks_queries_logits.float()
        oks, labs, scs = [], [], []
        for i in range(N):
            lab, cf = self._semantic(cls[i], msk[i], hw)
            seg, seg_ids[i], ok, lb, sc = self._panoptic(cls[i], msk[i], hw)
            lab = torch.where(seg >= 0, seg, lab)
            oks.append(ok), labs.append(lb), scs.append(sc)
            sem[i] = lab.to(torch.uint8)
            conf[i] = (cf * 255).round_().to(torch.uint8)
        seg_ids, oks = seg_ids.cpu().numpy(), torch.stack(oks).cpu().numpy()
        labs, scs = torch.stack(labs).cpu().numpy(), torch.stack(scs).cpu().numpy()
        want = np.isin(labs, self.instance_ids) & oks
        inst = [[(seg_ids[i] == k, int(labs[i, k]), float(scs[i, k])) for k in np.nonzero(want[i])[0]]
                for i in range(N)]
        return dict(sem=sem.cpu().numpy(), conf=conf.cpu().numpy(), inst=inst)


def components(sem: np.ndarray, conf: np.ndarray, class_ids, min_area: int = 40):
    """Connected components (8-connectivity) of each class in one view's label map, as
    (mask, native id, score = mean confidence over the component)."""
    out = []
    present = np.bincount(sem.ravel(), minlength=256)
    for c in class_ids:
        if present[c] < min_area:
            continue
        n, lab, stats, _ = cv2.connectedComponentsWithStats((sem == c).astype(np.uint8), connectivity=8)
        for j in range(1, n):
            if stats[j, cv2.CC_STAT_AREA] < min_area:
                continue
            x, y, w, h = stats[j, :4]
            sub = lab[y:y + h, x:x + w] == j
            mask = np.zeros(sem.shape, bool)
            mask[y:y + h, x:x + w] = sub
            out.append((mask, int(c), float(conf[y:y + h, x:x + w][sub].mean()) / 255.0))
    return out
