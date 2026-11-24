# Data formats

## Panorama index

A Parquet or CSV table with one row per panorama (`streetnet/index.py`):

| Column | Type | Required | Meaning |
|---|---|---|---|
| `pano_id` | string | yes | unique identifier |
| `lat`, `lon` | float | yes | camera position, WGS 84 |
| `heading` | float | yes | compass bearing of the centre column of the panorama, degrees clockwise from north |
| `pitch`, `roll` | float | no | camera pitch and roll from the capture metadata in degrees, pitch 0 = level |
| `nx`, `ny`, `nz` | float | no | ground up-normal in the camera frame (x right, y forward, z up); overrides pitch and roll |
| `h` | float | no | camera height above the ground in metres; 2.5 when missing |
| `image` | string | no | path of the panorama, absolute or relative to `--images`; `<images>/<pano_id>.jpg` when missing |

Panoramas are equirectangular JPEGs (width = 2 x height) whose centre column looks along `heading`. Without pitch,
roll or a normal the camera is treated as level. The map frame is the UTM zone that contains the median camera
position.

## Reference points

`streetnet eval --gt` reads a Parquet or CSV table with one row per reference asset:

| Column | Meaning |
|---|---|
| `gt_key` | class key of the map (`tree`, `fire_hydrant`, `storm_drain`, ...; see `streetnet/configs/classes.json`) |
| `x`, `y` | position in the map frame (UTM metres of the index's zone), or |
| `lon`, `lat` | position in WGS 84, converted to the map frame |

The classes present in both the map and the reference table are evaluated. The evaluation region is given by
panoramas of the index (`--region`, a text file of pano ids; all panoramas by default) or by a table of disk centres
with `lon` and `lat` (`--region-centers`); [evaluation.md](evaluation.md) gives the protocol.

## Run folder

```
runs/city/
├── run.json                    views, threshold and vocabulary of the run; a later call with other settings stops
├── det/<pano_id>.parquet       detections of one panorama (below); written last, marks the panorama as done
├── raw/                        with --keep-raw: SAM 3 and Mask2Former records, label maps (PNG, Vistas ids)
├── logs/detect_<i>.log         one log per GPU process
└── map/
    ├── <name>_assets.parquet   the map (below)
    ├── <name>_assets.geojson   the same as GeoJSON points (id, class, support, confidence)
    ├── <name>_obs.parquet      the observations of each asset
    ├── <name>_panos.txt        the panoramas the map was built from
    └── <name>_summary.json     counts and settings
```

A detection file takes 33 KiB per panorama on average in the New York City run.

## Detections: `det/<pano_id>.parquet`

| Column | Meaning |
|---|---|
| `pano_id`, `model`, `label`, `raw_label`, `score` | panorama; `sam3` or `m2f_vistas_pan`; class key; prompt or Vistas class name; detection score |
| `view` | index of the detection view (0 to 3, yaw 0, 90, 180, 270 degrees) |
| `x0`, `y0`, `x1`, `y1`, `area_px` | box and mask area in view pixels (1008 x 1008) |
| `bot_px`, `bot_py`, `top_px`, `top_py` | contact pixel and top pixel |
| `bot_dx`, `bot_dy`, `bot_dz`; `top_*`; `cen_*` | camera-frame unit rays through the contact pixel, the top pixel and the mask centroid |
| `bot_az`, `bot_el` | azimuth (degrees clockwise from forward) and elevation of the contact ray |
| `cut_bottom`, `cut_side` | the mask touches the bottom edge, a side edge of the view |
| `owner` | the contact azimuth lies in this view's central sector |
| `mount` | `ground`, `pole` or `surface` |
| `dup` | the mask overlaps a higher-scoring mask of the same class in the same view (IoU > 0.6) |
| `prompt_rank` | position of the prompt within its class (0 = first prompt); empty for Mask2Former rows |
| `sign_filter` | `signal_relabel` for sign detections relabelled traffic signal by the sign filter, else empty |
| `drng` | MoGe-2 range at the contact point in metres (median of five samples), NaN when unavailable |

## Map: `map/<name>_assets.parquet`

The map stage writes the classes of the parameter file found in the detections: the 22 classes of the published map
and utility poles, which serve as pole carriers for signs. The dataset export leaves utility poles out.

| Column | Meaning |
|---|---|
| `asset_id` | row number |
| `label` | class key |
| `x`, `y`; `lon`, `lat` | position in the UTM frame (metres) and in WGS 84 |
| `support`, `n_obs` | distinct panoramas and observations in the cluster |
| `score_mean`, `score_max`, `conf` | mean and maximum detection score; confidence 1 - prod(1 - best score per panorama) |
| `rng_min`, `rng_med` | smallest and median horizontal range of the cluster's observations (m) |
| `spread` | median distance of the observations from the asset position (m) |
| `best_obs` | the observation with the smallest range, as an id local to its block |
| `best_pano`, `best_row` | that observation's panorama and its row in the panorama's detection file |
| `estimator` | `depth` (MoGe-2 range) or `ground` (flat-ground geometry), the more frequent among the cluster's observations |
| `block`, `dense` | the 1 km block that kept the asset; whether the block used the uncapped parameters |
| `source` | the model the class is taken from |

`<name>_obs.parquet` lists the clustered observations: `pano_id`, `row` (row in the panorama's detection file),
`model`, `label`, `raw_label`, `score`, `view`, box (`x0`, `y0`, `x1`, `y1`), `X`, `Y` (position in metres), `rng`,
`bearing`, `asset`, `loc_src` (`self` or `pole`), `loc_est` (`depth` or `ground`), `cut_bottom`, `cut_side` and
`block`.

The published dataset renames and reduces these columns:

| Map column | Dataset column |
|---|---|
| `label` | `class` |
| `lon`, `lat` | `lon`, `lat` (and `geometry`) |
| `support` | `support` |
| `conf` | `confidence` |
| `estimator` (`depth`, `ground`) | `position_source` (`range`, `ground`) |
| `spread` | `spread_m` |
| `rng_min` | `range_min_m` |
| `best_pano` | `borough` of that panorama (the panorama id is not published) |

## Configuration

| File | Contents |
|---|---|
| `streetnet/configs/prompts.json` | the vocabulary: per class, the SAM 3 prompts and the mount type |
| `streetnet/configs/fusion_params.json` | per-class fusion parameters and the relabelling of pedestrian signals |
| `streetnet/configs/classes.json` | display names, groups and colours of the published classes |

A run can use another vocabulary (`--vocabulary`) and other fusion parameters (`--params`). The map stage leaves out
classes without an entry in the parameter file and logs their names. The classes placed by range
(`DEPTH_CLASSES`) and the pole carriers (`POLE_CARRIERS`) are set in `streetnet/fusion.py`. Parameters for a new
city or camera rig should be tuned on reference data of that place.
