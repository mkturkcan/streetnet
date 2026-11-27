# Data formats

## Panorama index

A Parquet or CSV table containing one row per panorama (`streetnet/index.py`):

| Column | Type | Required | Description |
|---|---|---|---|
| `pano_id` | string | yes | Unique panorama identifier |
| `lat`, `lon` | float | yes | Camera coordinates (WGS 84 / ITRF2020) |
| `heading` | float | yes | Compass bearing of the panorama center column (degrees clockwise from north) |
| `pitch`, `roll` | float | no | Capture metadata pitch and roll in degrees (`pitch = 0` is level) |
| `nx`, `ny`, `nz` | float | no | Ground-plane normal in camera frame (`x` right, `y` forward, `z` up); overrides `pitch` and `roll` |
| `h` | float | no | Camera height above ground in meters (defaults to `2.5`) |
| `image` | string | no | Path to panorama image (absolute or relative to `--images`; defaults to `<images>/<pano_id>.jpg`) |

Input images are equirectangular JPEGs (2:1 aspect ratio) whose center column points along `heading`. When `pitch`, `roll`, and ground normals are omitted, the camera is assumed level. Metric coordinates use the UTM zone containing the median camera position.

## Reference points

`streetnet eval --gt` accepts a Parquet or CSV table with one row per ground-truth reference asset:

| Column | Description |
|---|---|
| `gt_key` | Asset class key (`tree`, `fire_hydrant`, `storm_drain`, etc.; see `streetnet/configs/classes.json`) |
| `x`, `y` | Coordinates in the index's UTM metric frame (meters), or |
| `lon`, `lat` | Geographic coordinates (pass `--gt-datum nad83` for NAD83(2011) municipal GIS layers to align to ITRF2020 prior to UTM projection) |

Evaluation covers the intersection of classes present in both the asset map and the reference table. The evaluation region is defined either from panorama IDs in the index (`--region`, defaulting to all panoramas) or from an explicit table of disk centers with `lon` and `lat` (`--region-centers`); see [evaluation.md](evaluation.md) for the matching protocol.

## Run directory

```
runs/city/
├── run.json                    view count, score threshold, model tags, and vocabulary; prevents mismatched resumes
├── det/<pano_id>.parquet       per-panorama detection records; written atomically upon completion
├── cw/<pano_id>.npy            world-frame crosswalk ground points (K, 2) float32 for zero-GIS curb-ramp filtering
├── occ/<pano_id>.bin           zlib-compressed 128 x 256 1-bit occlusion mask (~1 KiB) for clear-view counting
├── raw/                        optional (--keep-raw): raw SAM 3.1 / Mask2Former records and label PNGs
├── logs/detect_<i>.log         per-GPU worker logs
└── map/
    ├── <name>_assets.parquet   fused asset inventory (schema below)
    ├── <name>_assets.geojson   GeoJSON point export (id, class, support, confidence)
    ├── <name>_obs.parquet      individual observations linked to fused assets
    ├── <name>_panos.txt        panorama IDs included in the map
    └── <name>_summary.json     asset counts and fusion settings
```

## Detections: `det/<pano_id>.parquet`

| Column | Description |
|---|---|
| `pano_id`, `model`, `label`, `raw_label`, `score` | Panorama ID, model (`sam31` or `m2f_vistas_pan`), class key, prompt or Vistas label, and confidence score |
| `view` | Perspective view index (`0` to `K - 1`) |
| `x0`, `y0`, `x1`, `y1`, `area_px` | Bounding box and mask area in view pixels (1008 x 1008) |
| `bot_px`, `bot_py`, `top_px`, `top_py` | Ground-contact pixel and top mask pixel |
| `bot_dx`, `bot_dy`, `bot_dz`; `top_*`; `cen_*` | Camera-frame unit rays through contact pixel, top pixel, and mask centroid |
| `bot_az`, `bot_el` | Azimuth (degrees clockwise from forward) and elevation of the contact ray |
| `cut_bottom`, `cut_side` | Boolean flags indicating whether the mask touches the bottom or side view borders |
| `owner` | True when the contact azimuth falls inside this view's central yaw sector |
| `mount` | Mount category: `ground`, `pole`, or `surface` |
| `dup` | True when suppressed by a higher-scoring same-class mask in the same view (`IoU > 0.6`) |
| `prompt_rank` | Index of the matched prompt within its class (`0` = primary prompt); `-1` for Mask2Former |
| `sign_filter` | `signal_relabel` when a sign detection was reclassified as a traffic signal; otherwise empty |
| `drng` | Road-plane self-calibrated MoGe-3 range at the lower-trunk contact strip (m), or NaN |
| `d_mask` | Self-calibrated vehicle-masked RLE-mask interior range (m) used when `base_veh >= 0.35` |
| `base_veh` | Panoptic vehicle-pixel fraction in the 7 x 3 band directly below the bounding box |
| `mh_frac`, `curb_near` | Manhole overlap and curb-sidewalk adjacency cues for `storm_drain` filtering |
| `road_alpha` | Per-panorama zero-GT road-plane range calibration factor applied to MoGe-3 |

## Map output: `map/<name>_assets.parquet`

`streetnet map` outputs all classes present in both the detections and the fusion parameter file: the 22 published classes plus `utility_pole` (used internally as a carrier for pole-mounted signs and excluded from the published dataset).

| Column | Description |
|---|---|
| `asset_id` | Zero-based integer asset identifier |
| `label` | Asset class key |
| `x`, `y`; `lon`, `lat` | Coordinates in UTM meters and WGS 84 degrees |
| `support`, `n_obs` | Number of distinct panoramas and total observations in the cluster |
| `score_mean`, `score_max`, `conf` | Mean/max detection score and aggregated confidence `1 - prod(1 - best_score_i)` |
| `rng_min`, `rng_med` | Minimum and median horizontal range of cluster observations (m) |
| `spread` | Median distance of cluster observations from the fused asset position (m) |
| `best_obs` | Tile-local ID of the closest high-score observation |
| `best_pano`, `best_row` | Panorama ID and row index of the best observation |
| `view`, `x0`, `y0`, `x1`, `y1` | Detection view index and bounding box in `best_pano` (used by `streetnet verify`) |
| `d_cw`, `mh_frac`, `curb_near` | Zero-GIS crosswalk distance (m), manhole overlap, and curb adjacency |
| `n_clear` | Number of nearby panoramas (within 15 m) with unobstructed 3D sightlines to the candidate |
| `sig2_tight`, `sig2_wide`, `sig2_comb`, `score_verified` | Zero-shot multi-distractor SigLIP 2 margins and combined verification score (added by `streetnet verify`) |
| `estimator` | Majority localization estimator in the cluster: `depth` or `ground` |
| `block`, `dense` | Owning 1 km tile ID and whether the tile used uncapped minimum support |
| `source` | Configured detector routing (`sam31`, `m2f`, or `ens`) |

## Configuration files

| File | Contents |
|---|---|
| `streetnet/configs/prompts.json` | SAM 3.1 natural-language prompts (34 prompts across 23 classes) and mount types |
| `streetnet/configs/fusion_params.json` | Per-class fusion, prior, occlusion-support, and SigLIP 2 verification parameters |
| `streetnet/configs/classes.json` | Display names, functional groups, and palette colors for published classes |

Custom vocabularies and fusion configurations can be supplied via `--vocabulary` and `--params`. Classes missing from the parameter file are skipped during mapping and logged.
