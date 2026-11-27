# Interactive map viewer

`docs/viewer/index.html` is a static browser viewer for StreetNet asset inventories. Built with MapLibre GL and an OpenFreeMap basemap, it streams vector tiles directly from a single [PMTiles](https://github.com/protomaps/PMTiles) archive via HTTP range requests. Assets can be filtered by class and confidence threshold, and clicking any marker inspects its class, confidence score, view support, localization estimator, borough, and coordinates.

Hosted deployment: Hugging Face Space [`mehmetkeremturkcan/streetnet-viewer`](https://huggingface.co/spaces/mehmetkeremturkcan/streetnet-viewer).

## Local deployment

```bash
pip install -e . && conda install -c conda-forge tippecanoe        # requires tippecanoe >= 2.17
hf download mehmetkeremturkcan/streetnet-nyc --repo-type dataset --local-dir streetnet-nyc
# Use the prebuilt archive (streetnet-nyc/tiles/streetnet-nyc.pmtiles),
# or rebuild from GeoParquet (~1 min for 4M assets, 87 MB):
python tools/viewer/build_tiles.py --assets streetnet-nyc/data/assets.parquet --out streetnet-nyc.pmtiles
python tools/viewer/serve.py --pmtiles streetnet-nyc/tiles/streetnet-nyc.pmtiles   # open http://127.0.0.1:8000/viewer/
```

`serve.py` serves `docs/` and the `.pmtiles` archive with HTTP range-request support (which Python's built-in `http.server` lacks). Any static web server supporting range requests can be used instead by opening `viewer/index.html?data=<pmtiles-url>`. To visualize a custom inventory, export it with `tools/export_nyc_dataset.py` (or any GeoParquet matching the schema in `docs/data.md`) and generate a PMTiles archive with `build_tiles.py`.

## Publishing

The Hugging Face Space contains `index.html` (configured to load its local tile archive), `assets/fonts/`, and `streetnet-nyc.pmtiles`. `release/work/update_release.sh` rebuilds the archive automatically with each dataset release.
