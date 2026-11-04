"""ShipCNN: our own convolutional neural network that decides whether a radar target is a ship.

Two-stage detector ("CFAR + CNN discrimination", a standard SAR design):
  1. CFAR proposes every bright target on the sea (dvd.detect).
  2. ShipCNN looks at a 64 x 64 chip (640 m x 640 m at 10 m per pixel) around each target and outputs P(ship).

Labels come from AIS, not from people (the xView3 dark-vessel dataset was labelled the same way):
  ship (1)      a target matched to a ship broadcasting AIS at the image time
  not ship (0)  a target on a mapped point structure (bridge pylon, wind turbine, buoy, beacon, pile, rock...) with no
                AIS ship within 300 m, or a random patch of open sea (piers and quays are excluded: boats moor there)
  excluded      every other target: it could be a boat without AIS, so we refuse to guess its label
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
import torch
from scipy import ndimage
from torch import nn

from . import config as C
from .geo import haversine_m
from .match import add_ship_details, match, project_ais

CHIP = 64            # chip size in pixels
PX_M = 10.0          # chip pixel size in metres (Sentinel-1 GRD is ~10 m)
SEED = 7


# ----------------------------------------------------------------------------------------------- model
class ShipCNN(nn.Module):
    """4 convolution blocks + 2 fully connected layers, ~62k weights. Input 1 x 64 x 64, output P(ship)."""

    def __init__(self):
        super().__init__()

        def block(c_in, c_out, pool=True):
            layers = [nn.Conv2d(c_in, c_out, kernel_size=3, padding=1), nn.BatchNorm2d(c_out), nn.ReLU()]
            return nn.Sequential(*layers, nn.MaxPool2d(2)) if pool else nn.Sequential(*layers)

        self.features = nn.Sequential(
            block(1, 16),                 # 16 x 32 x 32  edges, bright points
            block(16, 32),                # 32 x 16 x 16  blob shapes
            block(32, 64),                # 64 x 8 x 8    elongated hulls vs compact pylons, wakes, sidelobes
            block(64, 64, pool=False),    # 64 x 8 x 8
            nn.AdaptiveAvgPool2d(1),      # 64 x 1 x 1    global average: position-independent summary
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(32, 1),             # logit; sigmoid gives P(ship)
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def describe(model: nn.Module | None = None) -> list[dict]:
    """Layer-by-layer table (name, output shape, activation, weights), computed by running a dummy chip through."""
    model = model or ShipCNN()
    rows, x = [], torch.zeros(1, 1, CHIP, CHIP)
    model.eval()
    with torch.no_grad():
        for i, blk in enumerate(model.features, start=1):
            x = blk(x)
            if isinstance(blk, nn.Sequential):
                conv = blk[0]
                rows.append(dict(layer=f"{i}. Conv {conv.kernel_size[0]}x{conv.kernel_size[1]}, {conv.out_channels} filters + BatchNorm"
                                 + (" + MaxPool 2x2" if len(blk) == 4 else ""),
                                 output=" x ".join(map(str, x.shape[1:])), activation="ReLU",
                                 weights=sum(p.numel() for p in blk.parameters())))
            else:
                rows.append(dict(layer=f"{i}. Global average pool", output=" x ".join(map(str, x.shape[1:])),
                                 activation="-", weights=0))
        x = x.flatten(1)
        fc = [m for m in model.classifier if isinstance(m, nn.Linear)]
        rows.append(dict(layer=f"{len(rows) + 1}. Fully connected + Dropout 0.3", output=str(fc[0].out_features),
                         activation="ReLU", weights=sum(p.numel() for p in fc[0].parameters())))
        rows.append(dict(layer=f"{len(rows) + 1}. Output", output="1", activation="Sigmoid (P(ship))",
                         weights=sum(p.numel() for p in fc[1].parameters())))
    return rows


# ----------------------------------------------------------------------------------------------- chips
def extract_chip(scene, row: float, col: float) -> np.ndarray:
    """64 x 64 chip centred on (row, col), resampled to 10 m square pixels so ships look alike at any latitude.
    Land (and no-data) is set to calm-sea level: otherwise the network learns "city around = ship in a harbour"
    from the surroundings instead of judging the target itself (seen in the first trained model)."""
    lat = scene.grid.north - row * scene.grid.res_deg
    mx, my = scene.grid.m_per_px(lat)
    hr, hc = int(np.ceil(CHIP / 2 * PX_M / my)) + 2, int(np.ceil(CHIP / 2 * PX_M / mx)) + 2
    r, c = int(round(row)), int(round(col))
    pad = max(hr, hc)
    r0, r1, c0, c1 = max(r - hr, 0), r + hr, max(c - hc, 0), c + hc
    sub = np.where(scene.land[r0:r1, c0:c1], 1.0, scene.image[r0:r1, c0:c1])   # only the neighbourhood: cheap
    img = np.full((r1 - r0 + 2 * pad, c1 - c0 + 2 * pad), 1.0, dtype=np.float32)  # 1.0 = calm sea level
    img[pad:pad + sub.shape[0], pad:pad + sub.shape[1]] = sub
    r, c = r - r0, c - c0
    win = img[r + pad - hr:r + pad + hr, c + pad - hc:c + pad + hc]
    z = ndimage.zoom(win, (my / PX_M, mx / PX_M), order=1)
    zr, zc = (z.shape[0] - CHIP) // 2, (z.shape[1] - CHIP) // 2
    return preprocess(z[zr:zr + CHIP, zc:zc + CHIP])


def preprocess(chip: np.ndarray) -> np.ndarray:
    """Backscatter spans orders of magnitude, so the network sees log10 intensity (sea ~0, ships 1..3), scaled by 1/2."""
    return (np.log10(np.clip(chip, 1e-2, 1e3)) / 2).astype(np.float32)


def is_point_structure(kind: str) -> bool:
    """Structures that can only be 'not a ship': bridge pylons, turbines, buoys, beacons, lights, piles, rocks, platforms.
    Piers, quays, breakwaters, groynes and moorings are excluded: small boats without AIS tie up there, so a bright
    target beside one may well be a boat, and calling it 'not ship' would teach the network the wrong answer."""
    if not kind:
        return False
    if kind in ("pier", "quay", "breakwater", "groyne", "mooring", "small_craft_facility", "marine_farm", "wall",
                "shoreline_construction", "structure"):
        return False
    return True


def label_scene(scene, ais: pd.DataFrame, structures, area: str, pass_id: str, rng: np.random.Generator,
                ships: dict | None = None) -> pd.DataFrame:
    """CFAR proposals in one scene, labelled from AIS and mapped structures; plus random open-sea chips.
    `ships` (mmsi -> reported length and type) lets matching refuse pairs of the wrong size, which keeps quay returns
    that a small moored boat happens to sit next to out of the "ship" class."""
    from .detect import detect_scene
    from .pipeline import shore_distance_km
    det = detect_scene(scene.image, scene.land, scene.grid)
    proj = add_ship_details(project_ais(ais, scene.time), ships or {})
    rows = []
    if len(det):
        det["shore_km"] = shore_distance_km(scene.land, scene.grid, det["row"], det["col"])
        eligible = det.index[(det.score >= C.MIN_SCORE) & (det.length_m >= C.MIN_LENGTH_M)]
        sub = match(det.loc[eligible].reset_index(drop=True), proj)
        matched = {int(eligible[k]) for k in sub}
        struct_kind = structures.kinds_near(det.lon.values, det.lat.values)
        for i, d in det.iterrows():
            ais_close = len(proj) and haversine_m(d.lon, d.lat, proj.lon.values, proj.lat.values).min() < 300
            if i in matched:
                label, kind = 1, "ship (AIS-matched)"
            elif is_point_structure(struct_kind[i]) and not ais_close:
                label, kind = 0, "fixed structure"
            else:
                continue                                   # unknown: might be a boat without AIS
            rows.append(dict(row=d.row, col=d.col, lon=d.lon, lat=d.lat, label=label, kind=kind, score=d.score,
                             length_m=d.length_m))
    n_pos = sum(r["label"] for r in rows)
    sea = np.argwhere(~ndimage.binary_dilation(scene.land, iterations=40))   # well away from land and nodata
    taken = 0
    while sea.size and taken < max(n_pos, 10):
        r, c = sea[rng.integers(len(sea))]
        lon, lat = scene.grid.lonlat(r, c)
        if len(det) and haversine_m(lon, lat, det.lon.values, det.lat.values).min() < 500:
            continue                                       # keep random sea chips free of any target
        rows.append(dict(row=float(r), col=float(c), lon=float(lon), lat=float(lat), label=0, kind="open sea",
                         score=np.nan, length_m=np.nan))
        taken += 1
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["chip"] = [extract_chip(scene, r.row, r.col) for r in df.itertuples()]
    df["area"], df["pass_id"] = area, pass_id
    return df


# ----------------------------------------------------------------------------------------------- training
def augment(x: torch.Tensor) -> torch.Tensor:
    """Ships can point any way: random 90-degree rotations, flips and a small shift."""
    k = int(torch.randint(0, 4, (1,)))
    x = torch.rot90(x, k, dims=(2, 3))
    if torch.rand(1) < 0.5:
        x = torch.flip(x, dims=(3,))
    s = torch.randint(-3, 4, (2,))
    return torch.roll(x, shifts=(int(s[0]), int(s[1])), dims=(2, 3))


def group_split(df: pd.DataFrame, val_frac: float = 0.2, seed: int = SEED) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Validation split by location, so the same pylon or anchored ship seen on two days never sits on both sides."""
    groups = df.area + "_" + (df.lat * 50).round().astype(str) + "_" + (df.lon * 50).round().astype(str)
    g = np.array(sorted(groups.unique()))
    rng = np.random.default_rng(seed)
    val_g = set(rng.choice(g, size=max(1, int(len(g) * val_frac)), replace=False))
    is_val = groups.isin(val_g).values
    return df[~is_val], df[is_val]


def to_tensors(df: pd.DataFrame):
    x = torch.from_numpy(np.stack(df.chip.values)[:, None])
    y = torch.from_numpy(df.label.values.astype(np.float32))[:, None]
    return x, y


LR = 3e-4            # 1e-3 made validation accuracy swing by +/-10 points between epochs


def train(train_df: pd.DataFrame, val_df: pd.DataFrame, epochs: int = 60, lr: float = LR, batch: int = 64,
          log=print) -> tuple[ShipCNN, list[dict]]:
    torch.manual_seed(SEED)
    model = ShipCNN()
    xt, yt = to_tensors(train_df)
    xv, yv = to_tensors(val_df)
    pos = float(yt.sum())
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([(len(yt) - pos) / max(pos, 1)]))   # balance the classes
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    history, best, best_state = [], float("inf"), None
    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(len(xt))
        tl = 0.0
        for i in range(0, len(xt), batch):
            idx = perm[i:i + batch]
            xb = torch.stack([augment(x[None])[0] for x in xt[idx]])
            opt.zero_grad()
            loss = loss_fn(model(xb), yt[idx])
            loss.backward()
            opt.step()
            tl += loss.item() * len(idx)
        model.eval()
        with torch.no_grad():
            lv = model(xv)
            vl = loss_fn(lv, yv).item()
            va = ((torch.sigmoid(lv) > 0.5).float() == yv).float().mean().item()
        with torch.no_grad():
            ta = ((torch.sigmoid(model(xt)) > 0.5).float() == yt).float().mean().item()
        history.append(dict(epoch=ep, train_loss=tl / len(xt), val_loss=vl, train_acc=ta, val_acc=va))
        if vl < best:
            best, best_state = vl, {k: v.clone() for k, v in model.state_dict().items()}
        log(f"epoch {ep:2d}  train loss {tl / len(xt):.3f} acc {ta:.3f} | val loss {vl:.3f} acc {va:.3f}")
    model.load_state_dict(best_state)       # keep the epoch with the lowest validation loss (early-stopping style)
    model.eval()
    return model, history


def predict(model: ShipCNN, chips: np.ndarray) -> np.ndarray:
    if len(chips) == 0:
        return np.zeros(0)
    with torch.no_grad():
        return torch.sigmoid(model(torch.from_numpy(np.asarray(chips, dtype=np.float32)[:, None]))).numpy().ravel()


def metrics(y: np.ndarray, p: np.ndarray, thr: float = 0.5) -> dict:
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
    yh = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yh, labels=[0, 1]).ravel()
    return dict(n=int(len(y)), accuracy=accuracy_score(y, yh), precision=precision_score(y, yh, zero_division=0),
                recall=recall_score(y, yh, zero_division=0), f1=f1_score(y, yh, zero_division=0),
                roc_auc=roc_auc_score(y, p) if len(set(y)) == 2 else float("nan"),
                tp=int(tp), fp=int(fp), tn=int(tn), fn=int(fn))


# ----------------------------------------------------------------------------------------------- persistence
MODEL_PATH = os.path.join("models", "ship_cnn.pt")


def save(model: ShipCNN, path: str = MODEL_PATH) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)


_loaded: ShipCNN | None = None


def load(path: str = MODEL_PATH) -> ShipCNN | None:
    global _loaded
    if _loaded is None and os.path.exists(path):
        m = ShipCNN()
        m.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
        m.eval()
        _loaded = m
    return _loaded


def threshold(report_path: str = os.path.join("models", "ship_cnn_report.json")) -> float:
    """The operating threshold chosen on validation data at training time (config.CNN_THRESHOLD if no report)."""
    try:
        return float(json.load(open(report_path))["threshold"])
    except Exception:
        return C.CNN_THRESHOLD


def save_report(report: dict, path: str = os.path.join("models", "ship_cnn_report.json")) -> None:
    with open(path, "w") as f:
        json.dump(report, f, indent=2, default=float)
