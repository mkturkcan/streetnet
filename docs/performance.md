# Performance

## Throughput

Measured in the New York City run on four NVIDIA RTX PRO 6000 Blackwell Max-Q GPUs (96 GB each), one detection
process per GPU with 8 CPU worker processes each:

| | |
|---|---|
| Time per panorama and GPU, end to end | 0.90 s (mean of the four processes over their first 11,000 panoramas) |
| Panoramas per hour on four GPUs | 16,000 |
| GPU memory per process | 33 to 37 GB |
| Detection file per panorama | 33 KiB |

GPU time per panorama by stage:

| Stage | Time |
|---|---:|
| SAM 3: 4 views x 25 prompts, lazy mask head, compiled | 0.45 s |
| Mask2Former Swin-L: 4 views at 1024 px | 0.27 s |
| MoGe-2: 4 views, resolution level 9, float16 | 0.14 s |

JPEG decoding runs in four threads ahead of the GPU, and the CPU stage (mask records, routing, sign filter, range
samples, writing) runs in worker processes while the GPU processes the next panorama. For comparison, the
configuration of the pilot study (8 views, 34 prompts, full mask head) took 1.9 s per panorama for SAM 3 alone.

## Optimizations

| Optimization | Effect on the output |
|---|---|
| One image-encoder pass for the four views and cached text embeddings; all 100 (view, prompt) pairs in one decoder batch | none |
| Lazy mask head: the pixel decoder runs for pairs with a query above the threshold, mask logits for those queries | same detections; mask IoU 0.9993 on average (9 panoramas, 3,072 detections) |
| bfloat16 autocast for SAM 3 and Mask2Former, TF32 matrix products | small score changes (below) |
| `torch.compile` of the SAM 3 image backbone and fusion encoder; decoder compiled with CUDA graphs | small score changes (below) |
| Four views (the pilot used eight) | half the SAM 3 work; coverage starts 3.55 degrees below the horizon where views meet |
| 25 prompts: the pilot vocabulary of 34 prompts minus 9 prompts of classes that the map leaves out or takes from Mask2Former | fewer classes |
| Mask2Former instance segments materialized for catch basins | none |
| MoGe-2 range sampled at detection time; no range maps written to disk | none |

Mask2Former runs uncompiled. On a test set of 8 panoramas, its Traffic Light pixels compiled and uncompiled agreed
with an IoU of 0.31, averaged over the panoramas containing that label, and the sign filter depends on that label.

Compiled and uncompiled SAM 3 differ in floating-point rounding. On 24 panoramas, 95.4% of the 3,158 detections of
the compiled run have a counterpart in an uncompiled run (same view and class, box IoU at least 0.5). Matched
detections differ in score by 0.0008 at the median, and at least 95% of them have the same range sample. The
detections without a counterpart score lower than the average detection (median 0.34 against 0.40).

The public code reproduces the New York City run. Run uncompiled on the same 24 panoramas, the research pipeline and
`streetnet detect` write identical detection files. `streetnet map --panos <list> --min-support-cap 1` rebuilds the
map of the run's 52,269 panoramas of plan level 18 and coarser with identical values in all columns: 675,862 assets
in 23 classes, of which the dataset export leaves out the 46,591 utility poles.
