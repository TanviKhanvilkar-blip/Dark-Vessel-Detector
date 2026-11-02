import numpy as np
import pandas as pd
import torch

from dvd import cnn
from dvd.synth import make_scene


def test_architecture_matches_the_documented_table():
    rows = cnn.describe()
    assert [r["output"] for r in rows] == ["16 x 32 x 32", "32 x 16 x 16", "64 x 8 x 8", "64 x 8 x 8", "64 x 1 x 1", "32", "1"]
    assert sum(r["weights"] for r in rows) == sum(p.numel() for p in cnn.ShipCNN().parameters()) == 62689
    assert rows[-1]["activation"].startswith("Sigmoid")


def test_chip_is_64px_at_10m_whatever_the_latitude():
    sc = make_scene(3, n_ships=1)
    s = sc.ships[0]
    r, c = sc.grid.rowcol(s.lon, s.lat)
    chip = cnn.extract_chip(sc, r, c)
    assert chip.shape == (64, 64) and chip.dtype == np.float32
    assert chip[28:36, 28:36].max() > 0.5 > np.median(chip)          # the ship is bright and in the middle
    edge = cnn.extract_chip(sc, 2, 2)                                   # near the border: padded, not crashing
    assert edge.shape == (64, 64)


def test_group_split_keeps_each_location_on_one_side():
    df = pd.DataFrame(dict(area="a", lat=np.repeat(np.linspace(55, 55.5, 30), 2), lon=12.0, label=0,
                           chip=[np.zeros((64, 64), np.float32)] * 60))
    tr, va = cnn.group_split(df)
    key = lambda d: set(zip(d.lat.round(4), d.lon))
    assert len(va) and not key(tr) & key(va)


def test_metrics():
    m = cnn.metrics(np.array([1, 1, 0, 0]), np.array([0.9, 0.2, 0.1, 0.7]))
    assert (m["tp"], m["fn"], m["tn"], m["fp"]) == (1, 1, 1, 1) and m["accuracy"] == 0.5


def test_network_learns_ship_vs_pylon():
    # toy data: "ships" are elongated bright bars, "pylons" compact bright dots; a working network separates them
    rng = np.random.default_rng(0)

    def chip(ship):
        x = rng.normal(0, 0.05, (64, 64)).astype(np.float32)
        if ship:
            x[31:33, 20:44] = 1.0
        else:
            x[30:34, 30:34] = 1.0
        return x

    df = pd.DataFrame(dict(chip=[chip(i % 2 == 0) for i in range(160)], label=[int(i % 2 == 0) for i in range(160)],
                           area="t", lat=np.arange(160) * 0.1, lon=0.0, kind="toy"))
    tr, va = cnn.group_split(df)
    torch.set_num_threads(2)
    # tiny data = 2 updates per epoch; BatchNorm's running statistics need ~12 epochs to settle (see the loss curve)
    model, hist = cnn.train(tr, va, epochs=20, log=lambda *_: None)
    p = cnn.predict(model, np.stack(va.chip.values))
    assert cnn.metrics(va.label.values, p)["accuracy"] > 0.9
