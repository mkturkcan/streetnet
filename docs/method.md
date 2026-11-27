# Method

StreetNet constructs georeferenced street asset inventories from street-level panoramas in three decoupled stages. The **detection stage** runs once per panorama on GPUs and writes compact per-instance geometric records alongside lightweight crosswalk and occlusion sidecars; the **mapping stage** operates solely on those records on CPUs, projecting each detection into a metric coordinate frame and fusing multi-view observations with zero-GIS spatial priors and occlusion-aware support; and the optional **verification stage** scores fused candidates once per physical asset with zero-shot multi-distractor SigLIP 2. This document specifies each stage in pseudocode alongside the constants used for New York City, referencing the corresponding modules under `streetnet/`.

## Classes

The published inventory spans 22 asset classes organized into eight functional groups. SAM 3.1 detects classes from 34 natural-language prompts in `streetnet/configs/prompts.json`, Mask2Former trained on Mapillary Vistas segments curb ramps and street-scene semantics, and catch basins ensemble both detectors. Display names, group assignments, and palette colors are stored in `streetnet/configs/classes.json`.

<!-- classes:start -->
| Class | SAM 3.1 prompts / source | Mount | Placement | Group |
|---|---|---|---|---|
| Street light | "street light pole" | ground | range | Lighting and poles |
| Sign pole | "sign pole" | ground | ground | Lighting and poles |
| Traffic signal | "traffic light", "pedestrian walk signal" | pole | range | Traffic control |
| Traffic sign | "traffic sign", "parking regulation sign", "street name sign" | pole | range | Traffic control |
| Bus stop sign | "bus stop sign" | pole | range | Traffic control |
| Manhole cover | "manhole cover" | surface | ground | Ground surface |
| Catch basin | "storm drain", "street drainage grate" + Mask2Former-Vistas: Catch Basin | surface | ground | Ground surface |
| Curb ramp | Mask2Former-Vistas: Curb Cut | surface | ground | Ground surface |
| Fire hydrant | "fire hydrant", "street fire hydrant" | ground | ground | Safety and emergency |
| Bollard | "bollard" | ground | ground | Safety and emergency |
| Barrier | "construction barrier" | ground | ground | Safety and emergency |
| Fire alarm box (experimental) | "fire alarm box", "red street fire call box" | ground | ground | Safety and emergency |
| Trash can | "trash can", "public litter basket" | ground | ground | Sanitation |
| Street tree | "tree trunk" | ground | range | Greenery |
| Planter (experimental) | "planter" | ground | ground | Greenery |
| Bus shelter | "bus stop shelter", "glass bus shelter" | ground | ground | Transit and mobility |
| Bike rack | "bike rack", "bicycle rack" | ground | range | Transit and mobility |
| Bike share dock (experimental) | "bike share docking station" | ground | ground | Transit and mobility |
| Parking meter (experimental) | "parking meter", "parking pay station" | ground | ground | Transit and mobility |
| Bench | "bench", "sidewalk bench" | ground | range | Amenities |
| Mailbox | "mailbox" | ground | ground | Amenities |
| LinkNYC kiosk | "tall black sidewalk digital kiosk", "LinkNYC kiosk" | ground | ground | Amenities |
<!-- classes:end -->

**Placement and class notes:** `range` classes are localized along the contact ray at their self-calibrated MoGe-3 metric range; `ground` classes intersect the leveled ground plane unless a parked vehicle occludes their base (`base_veh >= 0.35`), in which case localization switches automatically to vehicle-masked interior range (`d_mask`); pole-mounted classes inherit the contact point and range of their supporting pole. Detections from the pedestrian signal prompt are merged into `traffic_signal`, and utility poles (used internally as sign carriers) are omitted from the published export.

## Coordinate frames

```
cam    panorama sensor frame: x right, y forward (center column of the panorama), z up
       equirectangular pixel (u, v) of a W x H panorama, pixel centers at +0.5:
           lam = ((u + 0.5) / W - 0.5) * 2 pi          longitude, 0 at center column, positive right
           phi = (0.5 - (v + 0.5) / H) * pi            latitude, positive up
           ray = (cos phi sin lam, cos phi cos lam, sin phi)
lev    cam rotated by R_lev so that the local road plane lies at z = -h (camera height h = 2.5 m)
world  UTM meters in the zone containing the median camera (New York City: EPSG:32618)
       world <- lev: rotation about z by -heading; a lev ray of azimuth lam has compass bearing heading + lam
```

Road-plane leveling (`geometry.metadata_normal`, `geometry.leveling_from_normal`) derives a ground normal from capture pitch and roll metadata, calibrated in the Upper Manhattan pilot:

```
tilt  = -pitch;  roll = roll wrapped to [-180, 180)                                (degrees)
n     = normalize(sin(-0.7 roll), sin(tilt), 1)                                    ground normal in cam frame
n     = (0, 0, 1)  if |tilt| > 12 or |roll| > 8                                    fallback to level camera
R_lev = rotation mapping n to +z (Rodrigues formula)
```

## Stage 1: Detection (`detect.py`)

### Perspective views

```
views  = K pinhole views: yaw i * 360/K deg from forward; pitch -10 deg; 90 deg FOV;
         1008 x 1008 px (SAM 3.1 native resolution); focal length f = 504 / tan(45 deg) = 504 px
render:  for each view pixel, map cam-frame ray -> (lam, phi) -> bilinear sample from panorama,
         wrapping horizontally (geometry.render_views, on GPU)
```

With `K = 4` views, coverage spans 360 degrees in azimuth from 3.6 to 45.3 degrees below the horizon, capturing ground-contact points at horizontal distances from 2.5 m to 40 m for a 2.5 m camera height. To prevent cross-view duplicates within a panorama, each detection is assigned to the unique *owner view* whose central yaw sector contains its contact-ray azimuth (`geometry.owner_view`).

### SAM 3.1 (`models/sam3.py`)

```
input  views V (K, 3, 1008, 1008) in [0, 1]; prompts P (34, configs/prompts.json)
run image encoder(V) once; run text encoder(P) once (cached across panoramas)
pairs  = all K x P (view, prompt) pairs in fixed-shape chunks (max_pairs <= 100)
fusion encoder + decoder on each chunk -> 200 queries/pair with box, class logit, and presence logit
score(query) = sigmoid(class logit) * sigmoid(presence logit of pair)
retain queries with score > 0.22
lazy mask head: run prompt-conditioned pixel decoder only for pairs with retained queries,
                and evaluate mask logits only for retained queries;
                mask = (bilinear_upsample(logits, 1008) > 0)
```

Intra-view duplicates across prompts of the same class are flagged via mask NMS (`models/sam3.duplicate_flags`, IoU > 0.6 on 2x downsampled masks) and excluded during mapping.

### Mask2Former (`models/mask2former.py`)

Mask2Former (Swin-L backbone trained on Mapillary Vistas v1.2, 65 classes) runs on all views at 1024 px, providing panoptic Catch Basin segments (area >= 40 px), 8-connected Curb Cut semantic components (area >= 40 px), and a 2048 x 1024 stitched equirectangular semantic label map `L`. The label map supplies road pixels for depth self-calibration, crosswalk ground points (`cw/<pano_id>.npy`), a compressed 128 x 256 1-bit occlusion mask (`occ/<pano_id>.bin`), manhole and curb context for catch basins, base vehicle-occlusion fractions, and the sign consistency filter (`signs.py`).

### MoGe-3 metric geometry and road-plane self-calibration (`models/moge.py`, `geometry.fit_road_plane_scale`)

MoGe-3 ViT-G (`Ruicheng/moge-3-vitg`, resolution level 9, 3 sparse refinement steps, 90 degree FOV, `float16`) predicts a metric 3D point map per view fused into a 1024 x 512 equirectangular range map `R`. Without external depth ground truth, each panorama's range map is self-calibrated against unobstructed near-field roadway pixels (`ROAD_SURFACE` in `L`, leveled ray `d_lev.z < -0.15`, analytical horizontal distance in `[3, 15]` m):

```
r_road = h / (-d_lev.z)
alpha  = clip(median(r_road / R[road_pixels]), 0.6, 1.6)
R_cal  = alpha * R
```

For each detection, two range statistics are extracted from `R_cal`:
- `drng` (`d_strip`): median of five range samples on a vertical strip spanning 2% to 12% of box height above the ground-contact pixel (after pole association for pole-mounted detections).
- `d_mask`: 30th (or 45th) percentile of non-vehicle, non-sky range pixels inside the instance RLE mask (bottom 40% for tall poles; 15% to 92% height otherwise).

Alongside ranges, `base_veh` records the vehicle-pixel fraction in a 7 x 3 band directly below the bounding box, and `storm_drain` detections record panoptic manhole overlap (`mh_frac`) and curb-sidewalk adjacency (`curb_near`).

## Stage 2: Mapping (`mapping.py`, `fusion.py`)

### Pole association and single-panorama localization

Pole-mounted detections (`traffic_sign`, `bus_stop_sign`, `traffic_signal`) borrow the contact ray and strip range of the lowest-cost horizontally overlapping carrier (`sign_pole`, `street_light`, `utility_pole`) in the same view (`fusion.associate_poles`). Localization (`fusion.localize`) then computes:

```
filter      owner == True and dup == False
ray d       centroid ray for surface classes (manhole, catch basin, curb ramp); contact ray otherwise
d_lev       = R_lev * d
car_occ     = (base_veh >= 0.35)
r_use       = d_mask if (car_occ and finite(d_mask)) else drng
ground:     t = h / (-d_lev.z)  (requires d_lev.z < -0.001);   p = t * d_lev
range:      p = r_use * d_lev
selector:   use range for DEPTH_CLASSES or (car_occ and not surface) when r_use is finite; else ground
position    = camera_xy + world_from_lev(heading) * p.xy;      horizontal range rng = |p.xy|
```

### Cross-panorama clustering, priors, and adaptive support (`fusion.fuse`, `fusion.apply_candidate_filters`)

```
for each class c with parameters (source, R_max, r_c, method, pos, score_thr, min_support, conf_thr, ...):
    obs        = localized detections of c matching `source` with score >= score_thr,
                 1 m <= rng <= R_max, cut_bottom == False
    clustering:  bounded average-linkage (`cluster_bounded`, prevents single-linkage corridor chaining),
                 connected components (`cc`), or greedy (`greedy`) at radius r_c
    close_trim:  if cluster size >= 4, trim tail observations with rng > max(1.8 * r_min, r_min + 4 m)
    position:    weighted median (w = 1 / max(rng, 3 m)) refined by 3 IRLS iterations of anisotropic
                 Cauchy MAP (sigma_along = 0.5 + 0.12 rng, sigma_across = 0.5 + 0.03 rng)
    support    = count of distinct panoramas contributing to the cluster
    confidence = 1 - prod_i (1 - min(max_score_i, 0.999))
    priors:      curb_ramp requires distance d_cw <= cw_thr to self-projected crosswalk points (`cw/`);
                 storm_drain requires mh_frac < 0.15 and curb_near >= 0.02
    adaptive:    if use_occ and min_support >= 2, retain support == 1 candidates when 3D unobstructed
                 sightlines n_clear <= 1 (from `occ/` masks within 15 m) and confidence/margin pass strict floors
```

### Stage 3: Zero-shot multi-distractor SigLIP 2 verification (`models/siglip2.py`)

When `streetnet map --defer-verify` is followed by `streetnet verify`, each fused candidate renders a clean tight crop (`margin = 0.45`, `256 x 256`, no box overlay) and a wider context crop (`margin = 1.40`) from its closest observing panorama (`best_pano`). SigLIP 2 (`google/siglip2-so400m-patch16-384`) scores the cosine-similarity margin between positive class prompts and class-specific hard negative distractors:

```
sig2_comb = 0.65 * margin_tight + 0.35 * margin_wide
retain candidate iff (conf + w_sig * sig2_comb >= conf_thr) and (sig2_comb >= sig_thr)
```

### Per-class fusion and verification parameters

Parameters in `streetnet/configs/fusion_params.json` were selected by per-class grid search maximizing 3 m F1 against datum-aligned NYC Open Data on the 478-panorama Upper Manhattan development split (fire alarm boxes and unreferenced classes use conservative open-vocabulary defaults).

<!-- params:start -->
| Class | Source | R_max (m) | r_c (m) | Clustering | score_thr | min_support | conf_thr | w_sig | sig_thr | Tuned | Dev F1 at 3 m |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---|---:|
| Street light | sam31 | 20 | 3 | cc | 0.55 | 2+occ | 0.8 | 1.5 | -0.05 | yes | 0.279 |
| Sign pole | sam31 | 15 | 2 | bounded | 0.4 | 1 | 0.35 | 1.5 | 0 | yes | 0.153 |
| Traffic signal | sam31 | 15 | 2 | bounded | 0.4 | 2 | 0.6 | 0 |  | no |  |
| Traffic sign | sam31 | 20 | 3 | cc | 0.25 | 1 | 0.65 | 5 | -0.05 | yes | 0.301 |
| Bus stop sign | sam31 | 15 | 2 | bounded | 0.4 | 2 | 0.6 | 0 |  | no |  |
| Manhole cover | sam31 | 12 | 1.5 | bounded | 0.4 | 2 | 0.6 | 0 |  | no |  |
| Catch basin | ens | 16 | 2 | bounded | 0.55 | 1 | 0.65 | 2.5 | 0 | yes | 0.247 |
| Curb ramp | m2f | 16 | 2 | cc | 0.22 | 1 | 0.35 | 2.5 | 0 | yes | 0.700 |
| Fire hydrant | sam31 | 20 | 3 | bounded | 0.25 | 1 | 0.35 | 2.5 | 0.02 | yes | 0.496 |
| Bollard | sam31 | 15 | 2 | bounded | 0.4 | 2 | 0.6 | 0 |  | no |  |
| Barrier | sam31 | 15 | 2 | bounded | 0.4 | 2 | 0.6 | 0 |  | no |  |
| Fire alarm box | sam31 | 15 | 2 | bounded | 0.5 | 2 | 0.75 | 2 | 0.01 | no |  |
| Trash can | sam31 | 15 | 3 | bounded | 0.25 | 2+occ | 0.65 | 2.5 | 0.02 | yes | 0.181 |
| Street tree | sam31 | 15 | 3 | bounded | 0.4 | 1 | 0.35 | 0 |  | yes | 0.707 |
| Planter | sam31 | 15 | 2 | bounded | 0.4 | 2 | 0.6 | 0 |  | no |  |
| Bus shelter | sam31 | 15 | 1.5 | bounded | 0.25 | 1 | 0.35 | 2.5 | 0.02 | yes | 0.600 |
| Bike rack | sam31 | 20 | 1.5 | bounded | 0.25 | 1 | 0.5 | 5 | 0.02 | yes | 0.229 |
| Bike share dock | sam31 | 15 | 2 | bounded | 0.4 | 2 | 0.6 | 0 |  | no |  |
| Parking meter | sam31 | 20 | 1.5 | bounded | 0.25 | 1 | 0.35 | 2.5 | 0.02 | yes | 0.316 |
| Bench | sam31 | 20 | 2 | bounded | 0.55 | 2 | 0.35 | 0 |  | yes | 0.143 |
| Mailbox | sam31 | 15 | 2 | bounded | 0.4 | 2 | 0.6 | 0 |  | no |  |
| LinkNYC kiosk | sam31 | 20 | 3 | bounded | 0.25 | 1 | 0.35 | 2.5 | 0 | yes | 0.435 |
<!-- params:end -->

### City-scale tiled mapping (`mapping.build_map`)

Mapping partitions the UTM footprint into parallel 1000 m x 1000 m tiles padded with a 60 m halo (exceeding the maximum `R_max + r_c` of any class). Each tile reads detections, crosswalk points, and occlusion masks from panoramas in its halo, executes fusion and prior filtering independently, and retains only assets whose final coordinates lie inside the tile interior.
