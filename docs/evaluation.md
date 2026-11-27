# Evaluation

## Matching protocol

`streetnet eval` (`streetnet/evaluate.py`) benchmarks a predicted asset map against municipal reference points inside a defined evaluation region:

```
region      union of 15 m disks around the evaluation region centers (camera positions)
for each class c and matching radius R in {2, 3, 5} m:
    candidate edges   all (prediction, reference) pairs of class c with distance < R (reference anywhere on map)
    assignment        maximum-cardinality, minimum-distance one-to-one matching within each connected
                      component of the candidate bipartite graph (Hungarian algorithm)
    TP  matched pairs whose reference point lies inside the region
    FP  unmatched predictions inside the region
    FN  unmatched reference points inside the region
    matched pairs whose reference lies outside the region and unmatched exterior predictions are ignored
precision = TP / (TP + FP);  recall = TP / (TP + FN);  F1 = 2 * precision * recall / (precision + recall)
F1 is n/a if a class has no predictions or no reference points in the region, and 0 if both exist with TP = 0
micro aggregate: sum TP, FP, and FN across classes;  macro F1: mean F1 over classes with defined F1
median error: median Euclidean distance over pairs matched within 5 m
```

Allowing candidate matches to reference points outside the region prevents boundary artifacts where an asset near the edge of the test region matches a reference point just outside it.

## Municipal reference layers and geodetic alignment

Reference layers are drawn from NYC Open Data. Several datasets are incomplete, dated, or derived from road-centerline work orders, so valid detections absent from a municipal layer count as false positives under raw registry matching. In addition, NYC GIS layers are surveyed in NAD83(2011) (EPSG:6318) whereas Street View poses lie in ITRF2020/WGS 84 (EPSG:9989), creating a systematic horizontal shift of `1.05 m` in Manhattan (`dx = -0.349 m, dy = +0.993 m`). Pass `--gt-datum nad83` to `streetnet eval` to apply the official PROJ NAD83(2011)-to-ITRF2020 transformation automatically; aligning this datum offset lifts tight 2 m F1 on the eight surveyed inventories by +14.6 points and halves near-field curb ranging error.

## Pilot benchmark and label-blind visual audit

On the 9,652-panorama Upper Manhattan pilot test region (31,562 datum-aligned reference points), StreetNet achieves:

- **Municipal registry F1 (datum-aligned):** `0.569` F1 at 3 m (`0.641` at 5 m) on the eight surveyed inventories (+13.1 points over the strongest localization and fusion baseline) and `0.364` F1 at 3 m across all 14 layers.
- **Large-N strictly label-blind visual audit (`N = 7,901` judgments, Qwen3.8-27B, 99.73% cross-class specificity, zero Qwen components inside StreetNet):** `85.1%` Rogan-Gladen error-adjusted macro precision across all 14 classes (`74.6%` raw blind agreement), with `94.0%` visible recall and `82.3%` real-world F1 across the seven curb-surveyed inventories. Genuine visible misses account for only `6.2%` of unmatched surveyed reference points, whereas most municipal false negatives arise from mis-geocoded or removed records and persistent parked-vehicle occlusion.

## NYC Open Data reference layers

| Class | Dataset | Agency | ID | Coverage |
|---|---|---|---|---|
| Street tree | 2015 Street Tree Census - Tree Data | Parks | `uvpi-gqnh` | 2015 census |
| Curb ramp | Pedestrian Ramp Locations | DOT | `ufzp-rrqu` | 2019 survey |
| Fire hydrant | Hydrants | DEP | `5bgh-vtsn` | Full inventory |
| Catch basin | NYCDEP Citywide Catch Basins | DEP | `2w2g-fk3i` | Full inventory |
| Traffic sign | Street Sign Work Orders (current records) | DOT | `qt6m-xctn` | Work-order records (partial) |
| Sign pole | Street Sign Work Orders (sign positions) | DOT | `qt6m-xctn` | Work-order records (partial) |
| Street light | Street Sign Work Orders; Mobile Telecom Franchise Pole Reservations | DOT; OTI | `qt6m-xctn`; `tbgj-tdd6` | Partial |
| Trash can | DSNY Litter Basket Inventory | DSNY | `8znf-7b2c` | City-managed litter baskets |
| Bike rack | Bicycle Parking | DOT | `592z-n7dk` | City-installed racks |
| Bench | Seating Locations | DOT | `esmy-s8q5` | City-installed seating |
| Bus shelter | Bus Stop Shelters | DOT | `t4f2-8md7` | Full inventory |
| Parking meter | Parking Meters Locations and Status | DOT | `693u-uax6` | Full inventory |
| LinkNYC kiosk | LinkNYC Kiosk Locations | OTI | `s4kf-3yrf` | Full inventory |
| Fire alarm box | In-Service Alarm Box Locations | FDNY | `v57i-gtxb` | 2019 inventory |

Eight classes (manhole covers, bollards, barriers, planters, mailboxes, bus stop signs, traffic signals, and bike-share docks) lack comparable municipal point layers and are omitted from the accuracy table.

## Evaluation region

Per-class fusion and verification parameters were selected on the 478-panorama development split of the 10,180-panorama Upper Manhattan pilot study. The evaluation region lies within the pilot footprint while remaining strictly disjoint from the development split:

```
centers = mapped panoramas within 20 m of a pilot panorama
          and at least 30 m from every development-split panorama
          (ensuring 15 m evaluation disks do not overlap the development split)
```

Both the evaluation region centers (`eval/region_centers.parquet`) and reference points (`eval/ground_truth.parquet`) are distributed with the dataset.

## Results

Accuracy of the published citywide map at a 3 m matching radius:

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

To reproduce evaluation directly from a published dataset:

```bash
hf download mehmetkeremturkcan/streetnet-nyc --repo-type dataset --local-dir streetnet-nyc
cd streetnet-nyc
streetnet eval --map data/assets.parquet --gt eval/ground_truth.parquet \
    --region-centers eval/region_centers.parquet --gt-datum nad83 --out metrics.json
```
