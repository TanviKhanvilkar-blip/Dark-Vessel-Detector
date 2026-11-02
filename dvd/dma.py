"""Historical AIS from the Danish Maritime Authority (free daily files at http://aisdata.ais.dk).

Each day is one zipped CSV of every AIS message received around Denmark (~600 MB zipped, several GB of text).
We stream it in chunks and keep only the area and time window a pass needs, so memory stays small.
"""
from __future__ import annotations

import io
import zipfile

import pandas as pd

from . import store

S3 = "https://s3.eu-central-1.amazonaws.com/aisdata.ais.dk"   # path-style: the bucket name has dots, so the virtual-host cert fails

# DMA column names (matched case-insensitively, ignoring a leading '#')
COLS = {"timestamp": "time", "mmsi": "mmsi", "latitude": "lat", "longitude": "lon", "sog": "sog", "cog": "cog",
        "type of mobile": "mobile"}


def day_url(day: str) -> str:
    """day = 'YYYY-MM-DD'"""
    return f"{S3}/aisdk-{day}.zip"


def _open_csv(zip_path: str):
    z = zipfile.ZipFile(zip_path)
    name = next(n for n in z.namelist() if n.lower().endswith(".csv"))
    return io.TextIOWrapper(z.open(name), encoding="utf-8", errors="replace")


def read_window(zip_path: str, bbox, t_from: float, t_to: float, chunksize: int = 500_000) -> pd.DataFrame:
    """Messages from ships (class A/B) inside bbox = (west, south, east, north) between t_from and t_to (epoch s)."""
    west, south, east, north = bbox
    f = _open_csv(zip_path)
    header = f.readline().rstrip("\n").split(",")
    names = [h.strip().lstrip("#").strip().lower() for h in header]
    use = [i for i, n in enumerate(names) if n in COLS]
    missing = set(COLS) - {names[i] for i in use}
    if missing:
        raise ValueError(f"unexpected DMA file format; missing columns {missing}; header was {header[:12]}")
    out = []
    reader = pd.read_csv(f, header=None, usecols=use, names=range(len(names)), chunksize=chunksize, dtype=str)
    for chunk in reader:
        chunk.columns = [COLS[names[i]] for i in use]
        for c in ("lat", "lon", "sog", "cog", "mmsi"):
            chunk[c] = pd.to_numeric(chunk[c], errors="coerce")
        inside = chunk.lon.between(west, east) & chunk.lat.between(south, north)
        chunk = chunk[inside & chunk.mobile.str.startswith("Class", na=False)]
        if chunk.empty:
            continue
        t = pd.to_datetime(chunk["time"], format="%d/%m/%Y %H:%M:%S", errors="coerce", utc=True)
        chunk = chunk.assign(time=(t - pd.Timestamp(0, tz="UTC")).dt.total_seconds())
        chunk = chunk[chunk.time.between(t_from, t_to)]
        out.append(chunk[["mmsi", "time", "lon", "lat", "sog", "cog"]])
    df = pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=["mmsi", "time", "lon", "lat", "sog", "cog"])
    # the matcher needs speed and course; AIS uses 102.3 kn / 360 deg for "not available"
    df = df.dropna(subset=["mmsi", "time", "lon", "lat", "sog", "cog"])
    df = df[(df.sog < 102.3) & (df.cog < 360)]
    return df.astype({"mmsi": "int64"}).drop_duplicates().sort_values("time").reset_index(drop=True)


def import_window(zip_path: str, db_path: str, bbox, t_from: float, t_to: float) -> int:
    df = read_window(zip_path, bbox, t_from, t_to)
    con = store.connect(db_path)
    store.add_ais(con, df)
    con.close()
    return len(df)
