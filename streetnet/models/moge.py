"""MoGe-3 (ViT-G / ViT-L) and MoGe-2 metric range for the detection views, fused into one equirectangular map.

Supported models (`model_name`):
    moge3_g   Ruicheng/moge-3-vitg (default; MoGe-3 ViT-G with sparse volumetric refinement)
    moge3     Ruicheng/moge-3-vitl
    moge2     Ruicheng/moge-2-vitl-normal

Each model predicts a metric 3D point map per view with the true horizontal field of view in float16.
range = |point| (meters along the pixel ray); non-finite values, values <= 0, and values >= 1000 m become NaN.
`fuse_range` packs the per-view maps into a 1024 x 512 equirectangular map using the central-view lookup.
"""
from __future__ import annotations

import numpy as np
import torch

from ..geometry import equirect_lookup

DEPTH_REPOS = {
    "moge3_g": "Ruicheng/moge-3-vitg",
    "moge3": "Ruicheng/moge-3-vitl",
    "moge2": "Ruicheng/moge-2-vitl-normal",
}
DEFAULT_DEPTH_MODEL = "moge3_g"
RANGE_W, RANGE_H = 1024, 512
_FUSE_CACHE: dict = {}


class RangeModel:
    def __init__(self, device: str = "cuda", model_name: str = DEFAULT_DEPTH_MODEL,
                 resolution_level: int = 9, refine_steps: int = 3):
        if model_name not in DEPTH_REPOS:
            raise ValueError(f"unknown depth model {model_name!r}; expected one of {sorted(DEPTH_REPOS)}")
        self.device = torch.device(device)
        self.model_name = model_name
        self.repo = DEPTH_REPOS[model_name]
        self.resolution_level = resolution_level
        self.refine_steps = refine_steps
        self._is_v3 = model_name.startswith("moge3")
        if self._is_v3:
            from moge.model.v3 import MoGeModel
        else:
            from moge.model.v2 import MoGeModel
        self.model = MoGeModel.from_pretrained(self.repo).to(self.device).eval()

    @torch.inference_mode()
    def predict(self, views: torch.Tensor, fov_x: float) -> torch.Tensor:
        """views (N, 3, h, w) float [0, 1] -> range (N, h, w) float32, NaN where there is no estimate."""
        if self._is_v3:
            outs = []
            for i in range(views.shape[0]):
                out = self.model.infer(
                    views[i],
                    fov_x=float(fov_x),
                    resolution_level=self.resolution_level,
                    refine_steps=self.refine_steps,
                    use_fp16=True,
                )
                outs.append(out["points"].norm(dim=-1))
            r = torch.stack(outs).float()
        else:
            out = self.model.infer(views, fov_x=float(fov_x), resolution_level=self.resolution_level, use_fp16=True)
            r = out["points"].norm(dim=-1).float()
        return torch.where(torch.isfinite(r) & (r > 0) & (r < 1000.0), r, torch.full_like(r, float("nan")))


def fuse_range(per_view: torch.Tensor, views, W: int = RANGE_W, H: int = RANGE_H) -> np.ndarray:
    """(N, h, w) range maps on the GPU -> (H, W) float16 equirectangular map, NaN where no view covers."""
    key = (tuple(views), W, H, str(per_view.device))
    if key not in _FUSE_CACHE:
        vidx, PX, PY = equirect_lookup(tuple(views), W, H)
        ok = vidx >= 0
        h, w = views[0].h, views[0].w
        xi = np.clip(np.rint(PX).astype(np.int64), 0, w - 1)
        yi = np.clip(np.rint(PY).astype(np.int64), 0, h - 1)
        flat = np.where(ok, vidx.astype(np.int64) * h * w + yi * w + xi, 0)
        _FUSE_CACHE[key] = (torch.as_tensor(flat, device=per_view.device), torch.as_tensor(ok, device=per_view.device))
    flat, ok = _FUSE_CACHE[key]
    out = per_view.reshape(-1)[flat]
    return torch.where(ok, out, torch.full_like(out, float("nan"))).to(torch.float16).cpu().numpy()
