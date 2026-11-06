import json

import numpy as np
import pandas as pd

from dvd import config as C
from dvd import insight, store
from dvd.live import parse_ship_static
from dvd.pipeline import run_pass
from dvd.synth import make_ais, make_scene


def test_priors_share_of_types_by_length():
    ships = pd.DataFrame(dict(ship_type=["Cargo"] * 8 + ["Tanker"] * 2 + ["Undefined"] * 5 + ["Fishing"] * 3,
                              length_m=[200] * 10 + [200] * 5 + [30] * 3))
    p = insight.build_priors(ships)
    assert p["ships"] == 13
    assert insight.likely_types(180, p) == [["Cargo", 0.8], ["Tanker", 0.2]]    # Undefined is ignored
    assert insight.likely_types(20, p) == [["Fishing", 1.0]]


def test_nearest_ports_sorted_by_distance():
    ports = insight.nearest_ports("denmark", 12.62, 55.70)
    assert ports[0][0] == "Copenhagen" and ports[0][1] < ports[1][1]


def test_setting_from_surroundings():
    lon, lat = np.array([12.70, 12.701, 12.702, 12.703]), np.array([55.7] * 4)
    assert insight.setting(12.70, 55.7, 10, lon, lat, 20) == "Anchorage or waiting area"
    assert insight.setting(12.70, 55.7, 10, lon[:1], lat[:1], 20) == "Open water"
    assert insight.setting(12.70, 55.7, 2, lon[:1], lat[:1], 20) == "Coastal waters"
    assert insight.setting(12.70, 55.7, 2, lon, lat, 1) == "Port area"


def test_same_spot_note_is_dropped_only_in_crowded_areas():
    from dvd.site import shown_reason
    note = "no_ais_in_window,same_spot_in_another_pass"
    assert shown_reason(note, json.dumps({"setting": "Anchorage or waiting area"})) == "no_ais_in_window"
    assert shown_reason(note, json.dumps({"setting": "Port area"})) == "no_ais_in_window"
    assert shown_reason(note, json.dumps({"setting": "Open water"})) == note
    assert shown_reason("", "") == ""


def test_static_message_gives_ship_details():
    row = parse_ship_static({"MessageType": "ShipStaticData", "MetaData": {"MMSI": 219000001},
                             "Message": {"ShipStaticData": {"Name": "MAERSK TEST@@@", "Type": 71, "Destination": "AARHUS@@",
                                                            "Dimension": {"A": 150, "B": 50, "C": 15, "D": 15}}}})
    assert row == dict(mmsi=219000001, name="MAERSK TEST", ship_type="Cargo", length_m=200.0, destination="AARHUS")
    assert parse_ship_static({"MessageType": "PositionReport"}) is None


def test_ship_details_keep_known_values(tmp_path):
    con = store.connect(str(tmp_path / "s.sqlite"))
    store.add_ships(con, pd.DataFrame([dict(mmsi=1, name="A", ship_type="Cargo", length_m=100.0, destination="X")]))
    store.add_ships(con, pd.DataFrame([dict(mmsi=1, name=None, ship_type=None, length_m=None, destination="Y")]))
    assert store.get_ships(con, [1])[1] == dict(name="A", ship_type="Cargo", length_m=100.0, destination="Y")


def test_every_kept_target_gets_details_and_tracked_ships_get_ais(tmp_path):
    sc = make_scene(11, n_ships=20)
    con = store.connect(str(tmp_path / "t.sqlite"))
    ais = make_ais(sc, 11)
    store.add_ais(con, ais)
    store.add_ships(con, pd.DataFrame([dict(mmsi=int(m), name=f"SHIP {m}", ship_type="Cargo", length_m=120.0, destination="Y")
                                       for m in ais.mmsi.unique()]))
    _, det = run_pass(sc, con, "p", out_dir=str(tmp_path))
    kept = det[det.status != "filtered"]
    assert (kept.crop != "").all() and (kept["info"] != "").all()
    assert (tmp_path / kept.crop.iloc[0]).exists()
    m = json.loads(kept[kept.status == "matched"]["info"].iloc[0])
    assert m["ais"]["name"].startswith("SHIP") and m["ais"]["type"] == "Cargo" and len(m["ais"]["track"]) > 1
    c = json.loads(kept[kept.status == "candidate"]["info"].iloc[0])
    assert "ais" not in c and c["setting"]
