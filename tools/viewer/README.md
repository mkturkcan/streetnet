# Map viewer

`docs/viewer/index.html` is a static web map of the published asset map: MapLibre GL draws the assets from one
[PMTiles](https://github.com/protomaps/PMTiles) archive, read over HTTP range requests, on an OpenFreeMap basemap.
Assets can be filtered by class and confidence; a click shows class, confidence, support, position source and borough.

Online: the Hugging Face Space [`mehmetkeremturkcan/streetnet-viewer`](https://huggingface.co/spaces/mehmetkeremturkcan/streetnet-viewer).

## Run it locally

```bash
pip install -e . && conda install -c conda-forge tippecanoe        # tippecanoe 2.17 or later
hf download mehmetkeremturkcan/streetnet-nyc --repo-type dataset --local-dir streetnet-nyc
# either use the published tiles ...
#   streetnet-nyc/tiles/streetnet-nyc.pmtiles
# ... or build them from the GeoParquet (about one minute for 4 million assets; 87 MB):
python tools/viewer/build_tiles.py --assets streetnet-nyc/data/assets.parquet --out streetnet-nyc.pmtiles
python tools/viewer/serve.py --pmtiles streetnet-nyc/tiles/streetnet-nyc.pmtiles   # open http://127.0.0.1:8000/viewer/
```

`serve.py` serves `docs/` and the tile archive with range requests (Python's `http.server` does not support them).
Any web server with range support works as well; open `viewer/index.html?data=<url of the .pmtiles>`.
To view your own map, export it with `tools/export_nyc_dataset.py` (or write a GeoParquet with the columns of
`docs/data.md`) and build tiles from it.

## Publish

The Space holds `index.html` (with the tile URL set to its own copy of the archive), `assets/fonts/` and
`streetnet-nyc.pmtiles`. `release/work/update_release.sh` rebuilds the archive with each dataset version.
