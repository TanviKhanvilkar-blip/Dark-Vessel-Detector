import zipfile

import pytest

from dvd import store
from dvd.dma import import_window, read_window

HEADER = ("# Timestamp,Type of mobile,MMSI,Latitude,Longitude,Navigational status,ROT,SOG,COG,Heading,IMO,Callsign,Name,"
          "Ship type,Cargo type,Width,Length,Type of position fixing device,Draught,Destination,ETA,Data source type,A,B,C,D")
ROWS = [
    "29/10/2026 17:05:00,Class A,219000001,55.35,11.00,Under way,0,12.3,45.0,44,,,,Cargo,,,,,,,,AIS,,,,",   # kept
    "29/10/2026 17:06:00,Class B,219000002,55.30,10.90,Unknown,,6.0,180.0,,,,,,,,,,,,,AIS,,,,",            # kept
    "29/10/2026 17:06:00,Base Station,2190000,55.30,10.90,Unknown,,,,,,,,,,,,,,,,AIS,,,,",                 # not a ship
    "29/10/2026 17:06:00,Class A,219000003,56.90,11.00,Under way,0,10.0,90.0,,,,,,,,,,,,,AIS,,,,",          # outside box
    "29/10/2026 19:30:00,Class A,219000004,55.35,11.00,Under way,0,10.0,90.0,,,,,,,,,,,,,AIS,,,,",          # outside window
    "29/10/2026 17:07:00,Class A,219000005,55.35,11.00,Moored,0,102.3,360.0,,,,,,,,,,,,,AIS,,,,",           # speed n/a
]


@pytest.fixture
def zpath(tmp_path):
    p = tmp_path / "aisdk-2026-10-29.zip"
    with zipfile.ZipFile(p, "w") as z:
        z.writestr("aisdk-2026-10-29.csv", "\n".join([HEADER] + ROWS) + "\n")
    return str(p)


T_PASS = 1793293727.0  # 2026-10-29T17:08:47Z


def test_read_window_filters_area_time_and_type(zpath):
    df = read_window(zpath, (10.85, 55.20, 11.25, 55.50), T_PASS - 3600, T_PASS + 3600)
    assert sorted(df.mmsi) == [219000001, 219000002]
    row = df[df.mmsi == 219000001].iloc[0]
    assert row.time == T_PASS - 227 and row.sog == 12.3 and row.cog == 45.0


def test_import_window_writes_to_db(zpath, tmp_path):
    db = str(tmp_path / "h.sqlite")
    assert import_window(zpath, db, (10.85, 55.20, 11.25, 55.50), T_PASS - 3600, T_PASS + 3600) == 2
    assert store.connect(db).execute("SELECT COUNT(*) FROM ais").fetchone()[0] == 2
