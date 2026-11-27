"""Batched SAM 3.1 / SAM 3 text-prompted instance segmentation over many views and many prompts.

One call segments all (view, prompt) pairs of a panorama:
    1. the image encoder runs once on the N views, the text encoder once on the prompts (cached);
    2. the fusion encoder and the DETR-style decoder run on all N x P pairs in one batch
       (chunks of at most `max_pairs`);
    3. query score = sigmoid(class logit) x sigmoid(decoder presence logit); queries with
       score > threshold are kept;
    4. with lazy masks (the default) the prompt-conditioned pixel decoder runs on the pairs that have a kept
       query, and mask logits are computed for the kept queries. On 9 panoramas (3,072 detections) the
       detections equal those of the full mask head and the masks agree with a mean IoU of 0.9993.
Inference runs under bfloat16 autocast. `compile()` applies torch.compile to the image backbone and the
fusion encoder, and to the decoder with CUDA graphs (fixed shapes: N x P must be divisible by max_pairs).
"""
from __future__ import annotations

import os

import numpy as np
import torch
import torch.nn.functional as F

RES = 1008                 # SAM 3 / 3.1 native input size
SAM31_REPO = "facebook/sam3.1"
SAM31_FILE = "sam3.1_multiplex.pt"


def sam31_checkpoint() -> str:
    """Download or resolve the cached SAM 3.1 multiplex checkpoint from Hugging Face."""
    from huggingface_hub import hf_hub_download
    return hf_hub_download(SAM31_REPO, SAM31_FILE)


class Sam3Detector:
    def __init__(self, device: str = "cuda", threshold: float = 0.22, checkpoint: str | None = "sam3.1"):
        import sam3
        from sam3 import build_sam3_image_model

        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        self.device = torch.device(device)
        self.threshold = threshold
        if checkpoint == "sam3.1":
            ckpt_path = sam31_checkpoint()
            self.model_tag = "sam31"
        elif checkpoint in (None, "", "sam3"):
            ckpt_path = None
            self.model_tag = "sam3"
        else:
            ckpt_path = str(checkpoint)
            self.model_tag = "sam31" if "3.1" in ckpt_path else "sam3"
        bpe = os.path.join(os.path.dirname(sam3.__file__), "assets", "bpe_simple_vocab_16e6.txt.gz")
        with torch.device(self.device):
            self.model = build_sam3_image_model(bpe_path=bpe, checkpoint_path=ckpt_path, device=str(self.device))
        self.model.eval()
        self._text_cache: dict[tuple, dict] = {}

    def compile(self):
        m = self.model
        m.backbone.vision_backbone = torch.compile(m.backbone.vision_backbone, dynamic=False)
        m.transformer.encoder = torch.compile(m.transformer.encoder, dynamic=False)
        m.transformer.decoder = torch.compile(m.transformer.decoder, mode="reduce-overhead", dynamic=False)
        return self

    def _text(self, prompts: list[str]):
        key = tuple(prompts)
        if key not in self._text_cache:
            if len(self._text_cache) > 8:
                self._text_cache.clear()
            self._text_cache[key] = self.model.backbone.forward_text(list(prompts), device=self.device)
        return self._text_cache[key]

    def _grounding_lazy(self, backbone_out, find, img_ids, thr):
        """Grounding with the mask head restricted to the kept queries. Returns (probs (Q, 200), boxes cxcywh
        (Q, 200, 4), kept query pair index, kept query index, mask logits (K, h, w) or None)."""
        m = self.model
        prompt, prompt_mask, backbone_out = m._encode_prompt(backbone_out, find, m._get_dummy_prompt(len(img_ids)))
        backbone_out, enc, _ = m._run_encoder(backbone_out, find, prompt, prompt_mask)
        out = {"encoder_hidden_states": enc["encoder_hidden_states"],
               "prev_encoder_out": {"encoder_out": enc, "backbone_out": backbone_out}}
        out, hs = m._run_decoder(memory=out["encoder_hidden_states"], pos_embed=enc["pos_embed"],
                                 src_mask=enc["padding_mask"], out=out, prompt=prompt,
                                 prompt_mask=prompt_mask, encoder_out=enc)
        probs = (out["pred_logits"].float().sigmoid()
                 * out["presence_logit_dec"].float().sigmoid().unsqueeze(1)).squeeze(-1)
        q_idx, d_idx = torch.nonzero(probs > thr, as_tuple=True)
        if not q_idx.numel():
            return probs, out["pred_boxes"], q_idx, d_idx, None
        seg = m.segmentation_head
        pairs, inv = torch.unique(q_idx, return_inverse=True)
        ehs = out["encoder_hidden_states"][:, pairs]                        # (L, b, C), sequence first
        pr, pm = prompt[:, pairs], prompt_mask[pairs]
        ehs = ehs + seg.cross_attend_prompt(query=seg.cross_attn_norm(ehs), key=pr, value=pr,
                                            key_padding_mask=pm)[0]
        pix = seg._embed_pixels(backbone_feats=backbone_out["backbone_fpn"], image_ids=img_ids[pairs],
                                encoder_hidden_states=ehs)
        inst = seg.instance_seg_head(pix)                                   # (b, C, h, w)
        emb = seg.mask_predictor.mask_embed(hs[-1][q_idx, d_idx])           # (K, C)
        logits = torch.empty((len(q_idx),) + inst.shape[-2:], device=inst.device, dtype=inst.dtype)
        for b in range(len(pairs)):
            k = torch.nonzero(inv == b, as_tuple=True)[0]
            logits[k] = torch.einsum("kc,chw->khw", emb[k].to(inst.dtype), inst[b])
        return probs, out["pred_boxes"], q_idx, d_idx, logits

    @torch.inference_mode()
    def detect(self, views: torch.Tensor, prompts: list[str], max_pairs: int = 100, lazy_masks: bool = True):
        """views: (N, 3, H, W) float in [0, 1] on the GPU. Returns results[view][prompt] = dict(
        masks (K, H, W) bool, boxes (K, 4) xyxy view pixels, scores (K,)), all on the GPU."""
        from sam3.model import box_ops
        from sam3.model.data_misc import FindStage

        thr = self.threshold
        N, _, H, W = views.shape
        results = [{p: None for p in prompts} for _ in range(N)]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            img = views.to(self.device, torch.float32)
            if (H, W) != (RES, RES):
                img = F.interpolate(img, (RES, RES), mode="bilinear", align_corners=False)
            img = (img - 0.5) / 0.5
            backbone_out = {"img_batch_all_stages": img}
            backbone_out.update(self.model.backbone.forward_image(img))
            backbone_out.update(self._text(prompts))
            pairs = [(i, j) for j in range(len(prompts)) for i in range(N)]
            scale = torch.tensor([W, H, W, H], device=self.device, dtype=torch.float32)
            for c0 in range(0, len(pairs), max_pairs):
                chunk = pairs[c0:c0 + max_pairs]
                ii = torch.tensor([p[0] for p in chunk], device=self.device, dtype=torch.long)
                jj = torch.tensor([p[1] for p in chunk], device=self.device, dtype=torch.long)
                find = FindStage(img_ids=ii, text_ids=jj, input_boxes=None, input_boxes_mask=None,
                                 input_boxes_label=None, input_points=None, input_points_mask=None)
                if lazy_masks:
                    probs, pboxes, q_idx, d_idx, mlog = self._grounding_lazy(backbone_out, find, ii, thr)
                else:
                    out = self.model.forward_grounding(backbone_out=backbone_out, find_input=find, find_target=None,
                                                       geometric_prompt=self.model._get_dummy_prompt(len(chunk)))
                    probs = (out["pred_logits"].float().sigmoid()
                             * out["presence_logit_dec"].float().sigmoid().unsqueeze(1)).squeeze(-1)
                    q_idx, d_idx = torch.nonzero(probs > thr, as_tuple=True)
                    pboxes = out["pred_boxes"]
                    mlog = out["pred_masks"][q_idx, d_idx] if q_idx.numel() else None
                    del out
                if q_idx.numel():
                    sc = probs[q_idx, d_idx]
                    bx = box_ops.box_cxcywh_to_xyxy(pboxes[q_idx, d_idx].float()) * scale
                    ms = F.interpolate(mlog.unsqueeze(1).float(), (H, W), mode="bilinear", align_corners=False)[:, 0] > 0
                else:
                    sc = bx = ms = None
                q_cpu = q_idx.cpu().numpy()
                for k, (i, j) in enumerate(chunk):
                    sel = np.nonzero(q_cpu == k)[0]
                    if len(sel) == 0:
                        results[i][prompts[j]] = dict(masks=torch.zeros((0, H, W), dtype=torch.bool, device=self.device),
                                                      boxes=torch.zeros((0, 4), device=self.device),
                                                      scores=torch.zeros((0,), device=self.device))
                        continue
                    s = torch.as_tensor(sel, device=self.device)
                    results[i][prompts[j]] = dict(masks=ms[s], boxes=bx[s], scores=sc[s])
        return results


def duplicate_flags(masks: torch.Tensor, scores: torch.Tensor, labels, view_ids, iou_thr: float = 0.6) -> np.ndarray:
    """True for a mask that overlaps a higher-scoring mask of the same class in the same view with mask
    IoU > iou_thr (IoU on a 2x downsampled grid). Several prompts of one class can find the same object.
    masks (K, H, W) bool, scores (K,), both on the GPU; labels and view_ids are lists of length K."""
    K = masks.shape[0]
    dup = np.zeros(K, bool)
    if K < 2:
        return dup
    m = masks[:, ::2, ::2].reshape(K, -1).float()
    area = m.sum(1)
    groups: dict = {}
    for k, key in enumerate(zip(view_ids, labels)):
        groups.setdefault(key, []).append(k)
    for idx in groups.values():
        if len(idx) < 2:
            continue
        ii = torch.as_tensor(idx, device=masks.device)
        mm = m[ii]
        inter = mm @ mm.T
        union = area[ii][:, None] + area[ii][None, :] - inter
        iou = (inter / union.clamp(min=1)).cpu().numpy()
        sc = scores[ii].float().cpu().numpy()
        n = len(idx)
        order = np.lexsort((np.arange(n), -sc))          # highest score first, ties in input order
        rank = np.empty(n, int)
        rank[order] = np.arange(n)
        for a in range(n):
            if ((rank < rank[a]) & (iou[a] > iou_thr)).any():
                dup[idx[a]] = True
    return dup
