# Evaluation

## Protocol

`streetnet eval` (`streetnet/evaluate.py`) compares a map with reference points of the same classes inside an
evaluation region:

```
region      union of disks of radius 15 m around the region centres (camera positions)
per class c and matching radius R (2, 3 and 5 m):
    candidate pairs   (prediction, reference point) of class c closer than R, with the reference anywhere
    matching          one to one: the largest number of pairs, and among those the smallest total distance,
                      within each connected group of candidate pairs (Hungarian assignment)
    TP  matched pairs whose reference point lies in the region
    FP  unmatched predictions in the region
    FN  unmatched reference points in the region
    matched pairs with the reference outside the region and unmatched predictions outside it are ignored
precision = TP / (TP + FP);  recall = TP / (TP + FN);  F1 = 2 precision recall / (precision + recall)
F1 is undefined (n/a) for a class without predictions or without reference points in the region, and 0 for a class
with both but no true positive
all classes: TP, FP and FN summed over the classes;  macro F1: mean F1 over the classes with reference points in the
region and a defined F1
median error: median distance of the pairs matched within 5 m
```

Matching against reference points outside the region keeps an asset near the region boundary from counting as a
false positive when its reference point lies outside the region.

## Reference data for New York City

The reference points come from NYC Open Data. Several layers are partial or dated; a detected asset that is
missing from its layer counts as a false positive.

| Class | Dataset | Agency | Id | Coverage |
|---|---|---|---|---|
| Street tree | 2015 Street Tree Census - Tree Data | Parks | `uvpi-gqnh` | census of 2015 |
| Curb ramp | Pedestrian Ramp Locations | DOT | `ufzp-rrqu` | survey of 2019 |
| Fire hydrant | Hydrants | DEP | `5bgh-vtsn` | inventory |
| Catch basin | NYCDEP Citywide Catch Basins | DEP | `2w2g-fk3i` | inventory |
| Traffic sign | Street Sign Work Orders, current records | DOT | `qt6m-xctn` | signs with a work order |
| Sign pole | Street Sign Work Orders, sign positions | DOT | `qt6m-xctn` | signs with a work order |
| Street light | Street Sign Work Orders; Mobile Telecommunications Franchise Pole Reservation Locations | DOT; OTI | `qt6m-xctn`; `tbgj-tdd6` | partial |
| Trash can | DSNY Litter Basket Inventory | DSNY | `8znf-7b2c` | public litter baskets |
| Bike rack | Bicycle Parking | DOT | `592z-n7dk` | city-installed racks |
| Bench | Seating Locations | DOT | `esmy-s8q5` | city-installed seating |
| Bus shelter | Bus Stop Shelters | DOT | `t4f2-8md7` | inventory |
| Parking meter | Parking Meters Locations and Status | DOT | `693u-uax6` | inventory |
| LinkNYC kiosk | LinkNYC Kiosk Locations | OTI | `s4kf-3yrf` | inventory |
| Fire alarm box | In-Service Alarm Box Locations | FDNY | `v57i-gtxb` | 2019 |

Manhole covers, bollards, barriers, planters, mailboxes, bus stop signs, traffic signals and bike share docks have no
point layer of this kind and are left out of the accuracy table.

## Region

The fusion parameters were tuned on the development split (478 panoramas) of a pilot study of 10,180 panoramas in
Upper Manhattan. The evaluation region of the New York City map lies in the pilot area and excludes the development
split:

```
centres = mapped panoramas within 20 m of a pilot panorama
          and at least 30 m from each panorama of the development split
          (two 15 m disks closer than 30 m could overlap)
```

The region centres are published with the dataset (`eval/region_centers.parquet`), together with the reference
points (`eval/ground_truth.parquet`).

## Results

The published version of the map, at a matching radius of 3 m:

<!-- accuracy:start -->
| Class | Ground truth | Predicted | Precision | Recall | F1 | Median error (m) |
|---|---:|---:|---:|---:|---:|---:|
| Street tree | 1,457 | 1,591 | 0.521 | 0.564 | 0.542 | 2.22 |
| Traffic sign | 913 | 634 | 0.146 | 0.101 | 0.119 | 2.26 |
| Sign pole | 485 | 306 | 0.039 | 0.025 | 0.030 | 2.91 |
| Street light | 351 | 781 | 0.097 | 0.217 | 0.134 | 1.58 |
| Curb ramp | 307 | 219 | 0.542 | 0.378 | 0.445 | 1.40 |
| Fire hydrant | 186 | 115 | 0.426 | 0.263 | 0.326 | 2.27 |
| Catch basin | 176 | 113 | 0.186 | 0.119 | 0.145 | 2.38 |
| Bike rack | 129 | 71 | 0.070 | 0.039 | 0.050 | 2.11 |
| Trash can | 51 | 314 | 0.029 | 0.176 | 0.049 | 1.97 |
| Parking meter | 29 | 16 | 0.125 | 0.069 | 0.089 | 2.49 |
| Fire alarm box | 23 | 2 | 0.000 | 0.000 | 0.000 | n/a |
| Bench | 19 | 182 | 0.055 | 0.526 | 0.100 | 2.55 |
| Bus shelter | 14 | 7 | 0.429 | 0.214 | 0.286 | 2.57 |
| LinkNYC kiosk | 13 | 0 | n/a | 0.000 | n/a | n/a |
| All classes | 4,153 | 4,351 | 0.281 | 0.293 | 0.287 | |
<!-- accuracy:end -->

The accuracy of a map depends on how densely its panoramas sample the streets. In version v0.1 (one panorama per
cell of about 116 m), 93.3% of the assets have support 1, so the minimum support is capped at 1, which raises recall
and lowers precision. The dataset card states the sampling of each version.

To recompute the table from the published files:

```bash
hf download mehmetkeremturkcan/streetnet-nyc --repo-type dataset --local-dir streetnet-nyc
cd streetnet-nyc
streetnet eval --map data/assets.parquet --gt eval/ground_truth.parquet \
    --region-centers eval/region_centers.parquet --out metrics.json
```
