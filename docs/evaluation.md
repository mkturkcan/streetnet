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
| Street tree | 9,768 | 12,432 | 0.500 | 0.636 | 0.560 | 2.17 |
| Traffic sign | 6,322 | 5,779 | 0.149 | 0.136 | 0.142 | 2.57 |
| Sign pole | 3,390 | 3,407 | 0.044 | 0.044 | 0.044 | 3.23 |
| Curb ramp | 2,962 | 2,254 | 0.633 | 0.479 | 0.546 | 1.41 |
| Street light | 2,409 | 7,464 | 0.084 | 0.261 | 0.127 | 1.84 |
| Catch basin | 1,582 | 1,147 | 0.305 | 0.221 | 0.256 | 1.82 |
| Fire hydrant | 1,454 | 1,106 | 0.375 | 0.284 | 0.323 | 2.29 |
| Bike rack | 817 | 691 | 0.041 | 0.034 | 0.037 | 2.10 |
| Trash can | 402 | 3,839 | 0.032 | 0.308 | 0.058 | 2.51 |
| Fire alarm box | 197 | 25 | 0.000 | 0.000 | 0.000 | n/a |
| Parking meter | 187 | 131 | 0.115 | 0.080 | 0.094 | 1.68 |
| LinkNYC kiosk | 109 | 4 | 0.000 | 0.000 | 0.000 | 4.09 |
| Bus shelter | 105 | 75 | 0.338 | 0.248 | 0.286 | 2.57 |
| Bench | 102 | 1,954 | 0.027 | 0.520 | 0.052 | 1.83 |
| All classes | 29,806 | 40,308 | 0.255 | 0.345 | 0.293 | |
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
