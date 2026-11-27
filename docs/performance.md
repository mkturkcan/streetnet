# Performance

## Throughput

StreetNet decouples GPU-bound per-panorama detection from CPU-bound tiled mapping and lightweight per-candidate verification. Benchmarked on four NVIDIA RTX PRO 6000 Blackwell Max-Q GPUs (96 GB VRAM each), running one detection process per GPU paired with 8 CPU worker processes each:

- **Batched GPU detection:** `bfloat16` autocast, shared image-encoder pass across all views, cached text embeddings, fixed-shape `(view, prompt)` chunking (`max_pairs <= 100`), lazy prompt-conditioned mask head, and optional PyTorch 2 `torch.compile` (`--compile`).
- **Asynchronous I/O and CPU post-processing:** four prefetch threads decode incoming JPEGs ahead of the GPU while CPU worker processes execute mask record extraction, zero-GT road-plane self-calibration (`fit_road_plane_scale`), vehicle-masked RLE-mask range sampling, sign consistency filtering, crosswalk point extraction, 1-bit occlusion mask compression (~1 KiB/panorama), and atomic Parquet serialization.
- **Per-candidate verification (`streetnet verify`):** zero-shot multi-distractor SigLIP 2 runs once per fused physical asset rather than per raw image detection, adding minimal amortized cost across a citywide run and supporting multi-GPU sharding via `--gpus`.

## Inference optimizations

| Optimization | Impact on output |
|---|---|
| Shared image-encoder pass across all views, cached text embeddings, and fixed-shape `(view, prompt)` chunk decoding | Exact equivalence |
| Lazy mask head: pixel decoder runs only on pairs with passing queries; mask logits evaluated only for retained queries | Identical detections; 0.9993 mean mask IoU (9 panoramas, 3,072 detections) |
| `bfloat16` autocast for SAM 3.1 and Mask2Former; `float16` for MoGe-3 and SigLIP 2; TF32 matmul precision | Minor floating-point score perturbations |
| PyTorch 2 `torch.compile` on SAM 3.1 image backbone and fusion encoder; CUDA-graph compilation on decoder | Minor floating-point score perturbations |
| Inline self-calibrated range sampling and 128 x 256 packed occlusion masks (~1 KiB/panorama) instead of saving dense depth/label PNGs | Enables city-scale CPU mapping without multi-terabyte intermediate storage |

Mask2Former is intentionally kept eager (uncompiled): compiled versus eager Mask2Former diverged on `Traffic Light` pixels, which directly impacts the sign consistency filter.

## Reproducibility

`streetnet detect` writes one atomic Parquet record per panorama (last, after `cw/` and `occ/` sidecars), making multi-day citywide runs cleanly resumable after interruption. Because detection records are decoupled from fusion, `streetnet map` and `streetnet verify` can be re-executed with updated parameters or panorama subsets without re-running panorama detection.
