"""The panorama index: one row per panorama with its position, heading and leveling.

Columns of the index file (Parquet or CSV):
    pano_id    str      unique id; the image is <images>/<pano_id>.jpg unless `image` is given
    lat, lon   float    WGS 84 / ITRF2020 position of the camera
    heading    float    compass bearing of the image's center column, degrees clockwise from north
    pitch      float    camera pitch from the capture metadata, degrees, 0 = level   (optional)
    roll       float    camera roll from the capture metadata, degrees               (optional)
    nx, ny, nz float    ground normal in the cam frame; overrides pitch and roll  (optional)
    h          float    camera height above the ground in meters, default 2.5        (optional)
    image      str      path of the equirectangular JPEG, absolute or relative to --images (optional)
Without pitch, roll or a normal the panorama is treated as level.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .geometry import leveling_from_normal, metadata_normal

DEFAULT_HEIGHT = 2.5
REQUIRED = ("pano_id", "lat", "lon", "heading")


def utm_epsg(lon: float, lat: float) -> int:
    """EPSG code of the UTM zone containing (lon, lat); New York City is in zone 18N (EPSG:32618)."""
    zone = int((lon + 180.0) // 6.0) + 1
    return (32600 if lat >= 0 else 32700) + zone


def transformers(epsg: int):
    from pyproj import Transformer
    return (Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True),
            Transformer.from_crs(f"EPSG:{epsg}", "EPSG:4326", always_xy=True))


def nad83_to_itrf2020(lon, lat):
    """Transform NAD83(2011) geographic coordinates (EPSG:6318) to ITRF2020/WGS 84 (EPSG:9989)."""
    from pyproj import Transformer
    t = Transformer.from_crs("EPSG:6318", "EPSG:9989", always_xy=True)
    x, y = t.transform(np.asarray(lon, np.float64), np.asarray(lat, np.float64))
    return np.asarray(x), np.asarray(y)


def ll_to_xy(lon, lat, epsg: int, datum: str = "wgs84"):
    """Project geographic coordinates into UTM meters. Use datum='nad83' for NAD83(2011) GIS layers."""
    if datum == "nad83":
        lon, lat = nad83_to_itrf2020(lon, lat)
    elif datum != "wgs84":
        raise ValueError(f"unknown datum {datum!r}; expected 'wgs84' or 'nad83'")
    x, y = transformers(epsg)[0].transform(np.asarray(lon, np.float64), np.asarray(lat, np.float64))
    return np.asarray(x), np.asarray(y)


def xy_to_ll(x, y, epsg: int):
    lon, lat = transformers(epsg)[1].transform(np.asarray(x, np.float64), np.asarray(y, np.float64))
    return np.asarray(lon), np.asarray(lat)


def read_index(path, epsg: int | None = None) -> pd.DataFrame:
    """Read and validate an index; returns it indexed by pano_id with x, y (UTM meters), nx, ny, nz, h.

    The UTM zone is the one containing the median camera position unless `epsg` is given; it is stored in
    df.attrs["epsg"]."""
    path = Path(path)
    df = pd.read_csv(path) if path.suffix.lower() in (".csv", ".txt") else pd.read_parquet(path)
    if "pano_id" not in df.columns and df.index.name == "pano_id":
        df = df.reset_index()
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing columns {missing}")
    if df["pano_id"].duplicated().any():
        raise ValueError(f"{path}: duplicate pano_id values")
    df = df.set_index("pano_id")
    df.index = df.index.astype(str)
    if epsg is None:
        epsg = utm_epsg(float(df["lon"].median()), float(df["lat"].median()))
    df["x"], df["y"] = ll_to_xy(df["lon"].to_numpy(), df["lat"].to_numpy(), epsg)
    if not all(c in df.columns for c in ("nx", "ny", "nz")):
        if "pitch" in df.columns and "roll" in df.columns:
            n = metadata_normal(df["pitch"].fillna(0.0).to_numpy(), df["roll"].fillna(0.0).to_numpy())
        else:
            n = np.tile([0.0, 0.0, 1.0], (len(df), 1))
        df["nx"], df["ny"], df["nz"] = n[:, 0], n[:, 1], n[:, 2]
    if "h" not in df.columns:
        df["h"] = DEFAULT_HEIGHT
    df["h"] = df["h"].fillna(DEFAULT_HEIGHT).astype(float)
    df.attrs["epsg"] = epsg
    return df


def leveling(index: pd.DataFrame, pano_ids) -> dict:
    """pano_id -> R_lev (lev <- cam) from the index normals."""
    n = index.loc[list(pano_ids), ["nx", "ny", "nz"]].to_numpy(float)
    return {p: leveling_from_normal(v) for p, v in zip(pano_ids, n)}


def image_path(index: pd.DataFrame, pano_id: str, images: Path | None) -> Path:
    if "image" in index.columns and isinstance(index.at[pano_id, "image"], str):
        p = Path(index.at[pano_id, "image"])
        return p if p.is_absolute() or images is None else Path(images) / p
    if images is None:
        raise ValueError("no image directory given and the index has no image column")
    return Path(images) / f"{pano_id}.jpg"


def load_image(path) -> np.ndarray:
    """RGB uint8 (H, W, 3) of an equirectangular JPEG at full resolution."""
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
    with Image.open(path) as im:
        return np.array(im.convert("RGB"))


def read_list(path) -> list[str]:
    """Panorama ids from a text file, one per line."""
    return [line.strip() for line in Path(path).read_text().splitlines() if line.strip()]
