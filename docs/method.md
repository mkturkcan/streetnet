# Method

StreetNet turns a set of street-level panoramas into a map of street assets in two stages. Detection runs once per
panorama and writes a table of detections with their geometry. Mapping reads the tables of many panoramas,
places each detection in a metric map frame and clusters detections of the same asset. This document gives both
stages as pseudocode, with the constants used for the New York City map. The code is in `streetnet/`; file and
function names are given with each step.

## Classes

The map has 22 classes in eight groups. SAM 3 finds 20 of them from the text prompts of
`streetnet/configs/prompts.json`; curb ramps and catch basins are classes of Mask2Former trained on Mapillary Vistas.
Display names, groups and colours are in `streetnet/configs/classes.json`.

<!-- classes:start -->
| Class | Prompts (SAM 3) or source | Mount | Placement | Group |
|---|---|---|---|---|
| Street light | "street light pole" | ground | range | Lighting and poles |
| Sign pole | "sign pole" | ground | ground | Lighting and poles |
| Traffic signal | "traffic light", "pedestrian walk signal" | pole | range | Traffic control |
| Traffic sign | "traffic sign", "parking regulation sign", "street name sign" | pole | range | Traffic control |
| Bus stop sign | "bus stop sign" | pole | range | Traffic control |
| Manhole cover | "manhole cover" | surface | ground | Ground surface |
| Catch basin | Mask2Former-Vistas: Catch Basin | surface | ground | Ground surface |
| Curb ramp | Mask2Former-Vistas: Curb Cut | surface | ground | Ground surface |
| Fire hydrant | "fire hydrant" | ground | ground | Safety and emergency |
| Bollard | "bollard" | ground | ground | Safety and emergency |
| Barrier | "construction barrier" | ground | ground | Safety and emergency |
| Fire alarm box (experimental) | "fire alarm box" | ground | ground | Safety and emergency |
| Trash can | "trash can" | ground | ground | Sanitation |
| Street tree | "tree trunk" | ground | range | Greenery |
| Planter (experimental) | "planter" | ground | ground | Greenery |
| Bus shelter | "bus stop shelter" | ground | ground | Transit and mobility |
| Bike rack | "bike rack" | ground | range | Transit and mobility |
| Bike share dock (experimental) | "bike share docking station" | ground | ground | Transit and mobility |
| Parking meter (experimental) | "parking pay station", "parking meter" | ground | ground | Transit and mobility |
| Bench | "bench" | ground | range | Amenities |
| Mailbox | "mailbox" | ground | ground | Amenities |
| LinkNYC kiosk | "LinkNYC kiosk" | ground | ground | Amenities |
<!-- classes:end -->

Placement: range classes are placed at the MoGe-2 range along the contact ray, ground classes where that ray meets
the ground plane; pole-mounted classes use the contact point of the pole below them. Experimental classes had a
precision below 0.67 in the pilot study's visual audit of up to 12 detections per class (fire alarm box 2 of 7,
parking meter 4 of 12, planter 7 of 12) or were not audited (bike share dock). The pedestrian signal prompt is
merged into traffic signals. `streetnet map` also writes utility poles, which serve as pole carriers for signs; the
published map leaves them out, because none of 12 audited utility-pole detections in the pilot was a utility pole.

## Frames

```
cam    panorama sensor frame: x right, y forward (centre column of the panorama), z up
       equirectangular pixel (u, v) of a W x H panorama, pixel centres at +0.5:
           lam = ((u + 0.5) / W - 0.5) * 2 pi          longitude, 0 at the centre column, positive to the right
           phi = (0.5 - (v + 0.5) / H) * pi            latitude, positive up
           ray = (cos phi sin lam, cos phi cos lam, sin phi)
lev    cam rotated by R_lev so that the local ground is the plane z = -h (camera height h = 2.5 m)
world  UTM metres of the zone containing the median camera (New York City: EPSG:32618)
       world <- lev: rotation about z by -heading; a lev ray of azimuth lam has compass bearing heading + lam
```

Leveling (`geometry.metadata_normal`, `geometry.leveling_from_normal`) uses the camera pitch and roll of the capture
metadata. The rule was calibrated on Street View metadata in the pilot study:

```
tilt = -pitch;  roll = roll wrapped to [-180, 180)                                  (degrees)
n    = normalise(sin(-0.7 roll), sin(tilt), 1)                                     ground normal, cam frame
n    = (0, 0, 1)  if |tilt| > 12 or |roll| > 8                                       treated as level
R_lev = the rotation taking n to +z (Rodrigues formula)
```

## Stage 1: detection (`detect.py`)

### Views

```
views = 4 pinhole views: yaw 0, 90, 180, 270 degrees from forward; pitch -10 degrees; 90 degree field of view;
        1008 x 1008 px (the SAM 3 input size); focal length f = 504 / tan(45 deg) = 504 px
render: for each view pixel, its cam-frame ray -> (lam, phi) -> bilinear sample of the panorama,
        wrapping horizontally (geometry.render_views, on the GPU)
```

The four views cover all azimuths from 3.6 to 45.3 degrees below the horizon, which contains the ground contact of
an object 2.5 m to 40 m from a camera 2.5 m above the ground. The narrowest coverage is where two views meet
(azimuths 45, 135, 225 and 315 degrees): there it starts 3.55 degrees below the horizon, and about 1 degree of the
horizon is missed; tall objects standing there can be cut at the view edge.

An object seen by two views is kept from one of them: the owner view of a detection is the view whose central
sector (yaw +- 45 degrees) contains the azimuth of its contact ray (`geometry.owner_view`).

### SAM 3 (`models/sam3.py`)

```
input  views V (4, 3, 1008, 1008) in [0, 1]; prompts P (25, configs/prompts.json)
image encoder(V) once; text encoder(P) once (cached across panoramas)
pairs  = all (view, prompt) pairs, 100 in one batch
fusion encoder and decoder on the pairs -> 200 queries per pair with a box, a class logit and a presence logit
score(query) = sigmoid(class logit) * sigmoid(presence logit of its pair)
keep queries with score > 0.25
mask head: the prompt-conditioned pixel decoder runs for pairs with a kept query, and mask logits are computed
           for kept queries (lazy mask head); mask = bilinear upsampling of the logits to 1008 px > 0
```

On 9 panoramas (4 views, 34 prompts, 3,072 detections) the full mask head and the lazy one gave the same
detections, with mask IoU 0.9993 on average and 0.977 at the minimum. Inference runs under bfloat16 autocast; with `--compile` the image backbone and the fusion encoder are compiled, and the
decoder is compiled with CUDA graphs.

Several prompts of one class can find the same object. Duplicate flags (`models/sam3.duplicate_flags`):

```
for each (view, class) group of masks, in order of decreasing score:
    dup(m) = exists m' in the group with score(m') > score(m) and IoU(m, m') > 0.6
IoU on masks downsampled by 2
```

Duplicates stay in the detection table with `dup = True` and are left out by the mapping stage.

### Mask2Former (`models/mask2former.py`)

Mask2Former Swin-L trained on Mapillary Vistas v1.2 (65 classes) runs on the four views at 1024 px:

```
semantic   seg[c] = sum over queries q of softmax(class_q)[c] * sigmoid(mask_q)
           label = argmax_c seg[c]; confidence = max_c seg[c] / sum_c seg[c]
panoptic   queries with a class other than no-object and probability > 0.5; each pixel is assigned to the query with
           the largest probability * mask probability; a query keeps a segment when the pixels assigned to it
           number more than 0.8 of its mask area (pixels with mask >= 0.5); the segment is its assigned pixels
           with mask >= 0.5
label map  panoptic segment class where assigned, semantic label elsewhere
catch basins   panoptic segments of class Catch Basin with at least 40 px; score = the query's class probability
curb ramps     8-connected components of label Curb Cut with at least 40 px; score = mean confidence
```

The label maps of the four views are fused into one 2048 x 1024 equirectangular label map: each panorama pixel takes
the nearest pixel of the most central view that covers it.

### Detection records (`instances.mask_record`)

```
for a mask with rows y0..y1 and columns x0..x1:
    k             = max(1, round(0.03 (y1 - y0 + 1)))
    contact pixel = (median column of the mask pixels in rows y1 - k .. y1, y1 + 0.5)     bottom edge of row y1
    top pixel     = (median column of the mask pixels in rows y0 .. y0 + k, y0 - 0.5)     top edge of row y0
    rays          = cam-frame rays through the contact pixel, the top pixel and the mask centroid
    cut_bottom    = y1 >= 1006 (the contact point is outside the view)
    cut_side      = x0 <= 1 or x1 >= 1006
    owner         = owner_view(azimuth of the contact ray) == this view
```

### Routing and sign filter (`detect.finish`, `signs.py`)

Curb ramps and catch basins are taken from Mask2Former (mount `surface`), all other classes from SAM 3. The sign
filter then examines each traffic sign and bus stop sign detection with the panorama's label map L:

```
f_light       = share of up to 400 mask pixels (seeded random sample) labelled Traffic Light in L
f_sign        = share labelled Traffic Sign (Front), Traffic Sign (Back) or Traffic Sign Frame
f_vehicle     = share labelled Bus, Car, Caravan, Motorcycle, On Rails, Other Vehicle, Trailer, Truck or Wheeled Slow
f_vehicle_adj = vehicle share of a 12 x 3 grid in the band below the box (rows y1 + 1 to y1 + max(2, 0.15 h))
overlap       = max over signal detections (score >= 0.3) in the same view of |box_sign and box_signal| / |box_sign|
overlap_win   = the same over signal detections that score higher than the sign
signal head   = f_light >= 0.4  or  (overlap_win >= 0.5 and f_sign < 0.3)
vehicle mount = f_vehicle >= 0.3  or  (f_vehicle_adj >= 0.7 and width / height >= 1.8 and top elevation < 2 deg)
vehicle mount                      -> removed
signal head and overlap >= 0.3     -> removed (a signal detection covers it)
signal head otherwise              -> relabelled traffic_signal
```

### Range samples (`models/moge.py`, `fusion.strip_samples`)

MoGe-2 (`Ruicheng/moge-2-vitl-normal`, resolution level 9, horizontal field of view 90 degrees) predicts a metric
point map per view in float16; range = |point|, set to NaN when non-finite, at most 0, or 1000 m or more. The four range maps
are fused into a 1024 x 512 equirectangular map R with the lookup of the label map. Then, per panorama:

```
apply pole association (below) so that pole-mounted detections sample at the contact point of their pole
for each detection with contact pixel (px, py) and box height bh in view v
    (for a pole-mounted detection, (px, py) is the contact pixel of its pole and bh its own box height):
    t      = 0.02, 0.045, 0.07, 0.095, 0.12
    pixels = (px, py - 0.5 - t * bh)                   a vertical strip inside the mask, above the contact pixel
    drng   = median over the five pixels of R at the nearest equirectangular pixel (NaN values ignored)
```

The detection table, `det/<pano_id>.parquet`, holds one row per detection with the record columns, `mount`, `dup`,
`prompt_rank`, `sign_filter` and `drng`. It is written last, so its presence marks the panorama as done.

## Stage 2: mapping (`mapping.py`, `fusion.py`)

### Pole association (`fusion.associate_poles`)

Traffic signs, bus stop signs and signals hang on poles; the bottom of their mask lies in the air. They take the
contact point of a carrier (sign pole, street light, utility pole) detected in the same panorama and view:

```
for a pole-mounted detection s and a carrier p in the same view:
    margin = max(0.15 width(s), 4 px)
    p qualifies when  p.x0 <= s.x1 + margin  and  p.x1 >= s.x0 - margin        the columns overlap
                      p.y1 > s.y1                                               the carrier reaches lower
                      p.y0 <= s.y1 + 0.35 height(p)                             and reaches up to the object
    cost(p) = |contact column of p - centre column of s| / width(s) + 0.001 (p.y1 - s.y1)
s takes the contact columns of the qualifying carrier with the lowest cost; without one it stays unlocalized
```

### Localization (`fusion.localize`)

```
selection   owner rows with dup = False
contact ray d: the centroid ray for surface classes (manholes, catch basins, curb ramps), the contact ray otherwise
d_lev = R_lev d
ground:   t = h / -d_lev.z  (d_lev.z < -0.001);  p = t d_lev              the ray meets the plane z = -h
range:    p = drng d_lev                                                   the contact point at the MoGe-2 range
estimator per class:
    range  for street trees, street lights, traffic signs, bus stop signs, traffic signals, benches, bike racks
           and utility poles, when drng is finite; ground otherwise
position  = camera position + world_from_lev(heading) p;   horizontal range rng = |p.xy|
```

On the pilot's development split (478 panoramas, 510 reference trees, the same fusion parameters for both), trees
placed at the MoGe-2 range reached F1 0.62 at 3 m, and trees placed on the ground plane 0.41. The ground estimate is
kept for small objects whose contact point is visible.

### Clustering and asset attributes (`fusion.fuse`, `fusion.fuse_with_params`)

```
per class c with parameters (R_max, r_c, method, pos, score_thr, min_support, conf_thr):
    obs = localized rows of c with score >= score_thr, 1 m <= rng <= R_max and a visible contact (cut_bottom = False)
    weight w = 1 / max(rng, 3 m)
    greedy:  observations in order of decreasing w join the nearest cluster centre within r_c that holds no
             observation of the same panorama, or start a cluster; centres are weighted means
    cc:      connected components of the graph that links observations closer than r_c
    position  = weighted median of x, weighted median of y
    aniso:    for clusters with support >= 2, refine the position by weighted least squares with per-observation
              Gaussians elongated along the viewing ray, sigma_along = 0.5 + 0.12 rng, sigma_across = 0.5 + 0.03 rng
              (metres), three iterations with Cauchy weights 1 / (1 + m^2 / 4)
    support    = number of distinct panoramas in the cluster
    confidence = 1 - prod over those panoramas of (1 - min(best score of the panorama, 0.999))
    keep the asset when support >= min_support and confidence >= conf_thr
```

Before clustering, pedestrian signals are relabelled traffic signals.

### Per-class parameters

The parameters (`streetnet/configs/fusion_params.json`) of 14 classes were chosen by a grid search per class that
maximized F1 at 3 m against NYC Open Data on the development split of the pilot study (478 of its 10,180 panoramas,
8 views, 34 prompts). The 8 classes without a reference layer (tuned = no) use manually set values: R_max 15 m, r_c 2 m,
greedy clustering, aniso position, score_thr 0.4, min_support 2, conf_thr 0.6, with R_max 12 m and r_c 1.5 m for
manhole covers. The file's `default` entry fills the keys a class leaves out (min_range 1 m, w_floor 3 m). Four tuned
classes rest on 11 or fewer reference points of the development split (bench 5, fire alarm box 7, bus shelter 10,
LinkNYC kiosk 11).

<!-- params:start -->
| Class | R_max (m) | r_c (m) | Clustering | Cluster position | score_thr | min_support | conf_thr | Tuned | Dev F1 at 3 m | Dev reference points |
|---|---:|---:|---|---|---:|---:|---:|---|---:|---:|
| Street light | 25 | 3 | cc | median | 0.5 | 2 | 0 | yes | 0.275 | 134 |
| Sign pole | 25 | 4 | cc | aniso | 0 | 2 | 0.7 | yes | 0.120 | 207 |
| Traffic signal | 15 | 2 | greedy | aniso | 0.4 | 2 | 0.6 | no |  |  |
| Traffic sign | 20 | 4 | cc | median | 0 | 1 | 0.5 | yes | 0.292 | 407 |
| Bus stop sign | 15 | 2 | greedy | aniso | 0.4 | 2 | 0.6 | no |  |  |
| Manhole cover | 12 | 1.5 | greedy | aniso | 0.4 | 2 | 0.6 | no |  |  |
| Catch basin | 15 | 1 | cc | median | 0 | 1 | 0.8 | yes | 0.222 | 98 |
| Curb ramp | 15 | 2 | cc | median | 0.5 | 1 | 0 | yes | 0.585 | 141 |
| Fire hydrant | 20 | 3 | cc | aniso | 0 | 1 | 0.6 | yes | 0.418 | 82 |
| Bollard | 15 | 2 | greedy | aniso | 0.4 | 2 | 0.6 | no |  |  |
| Barrier | 15 | 2 | greedy | aniso | 0.4 | 2 | 0.6 | no |  |  |
| Fire alarm box | 15 | 2 | greedy | median | 0 | 1 | 0.6 | yes | 0.182 | 7 |
| Trash can | 25 | 2 | cc | aniso | 0 | 1 | 0.7 | yes | 0.127 | 66 |
| Street tree | 25 | 3 | cc | aniso | 0 | 2 | 0.6 | yes | 0.633 | 510 |
| Planter | 15 | 2 | greedy | aniso | 0.4 | 2 | 0.6 | no |  |  |
| Bus shelter | 20 | 4 | greedy | median | 0 | 1 | 0.7 | yes | 0.476 | 10 |
| Bike rack | 20 | 3 | greedy | median | 0.5 | 1 | 0 | yes | 0.225 | 49 |
| Bike share dock | 15 | 2 | greedy | aniso | 0.4 | 2 | 0.6 | no |  |  |
| Parking meter | 15 | 1.5 | cc | median | 0.5 | 1 | 0.6 | yes | 0.170 | 30 |
| Bench | 20 | 1 | greedy | median | 0.5 | 2 | 0 | yes | 0.364 | 5 |
| Mailbox | 15 | 2 | greedy | aniso | 0.4 | 2 | 0.6 | no |  |  |
| LinkNYC kiosk | 10 | 2 | greedy | median | 0 | 1 | 0.8 | yes | 0.167 | 11 |
<!-- params:end -->

### City-scale processing (`mapping.build_map`)

```
blocks   squares of 1000 m in the UTM frame that contain at least one panorama
for each block, in parallel:
    panoramas  = those inside the block or within 60 m of it (the halo)
    run association, localization and clustering on their detections
    keep the assets whose position lies inside the block
asset ids are renumbered over all blocks
```

The 60 m halo is larger than R_max + r_c of any class (at most 29 m), so a greedy cluster near a block edge keeps
all of its observations; clusters of the connected-component method can chain beyond r_c and can then lose
observations at a block edge. Each asset is kept by one block.

### Order of the New York City run

The index of the New York City run has 1,170,687 panoramas of the five boroughs. They are processed on four GPUs in
order of plan level: for Web Mercator zoom levels z = 14 to 22, with cells of edge 40,075 km x cos(latitude) / 2^z
(1.85 km at z = 14, 116 m at z = 18, 29 m at z = 20, 7 m at z = 22), each cell that holds no panorama selected at a
coarser level contributes the panorama nearest its centre, which gets plan level z; the remaining panoramas get
level 23. The panoramas up to a complete level form an even sample of the city. Each published version of the map
states the panoramas it was built from.

Where panoramas are sparse, few assets are seen from two of them: in version v0.1 of the New York City map (plan
levels up to 18, one panorama per cell of about 116 m), 93.3% of the assets have support 1. `--min-support-cap k` caps the minimum
support of each class at k for such maps; `--dense-panos` lists panoramas whose blocks keep the uncapped values (a
block is dense when at least half of its own panoramas are listed). The dataset card of the New York City map states
the panoramas and the cap of each version.
