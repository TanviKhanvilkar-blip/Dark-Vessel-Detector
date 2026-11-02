"""Build the AIS-labelled chip dataset, train ShipCNN, evaluate on the held-out Oresund, and write the report + figures.

    python -m dvd train-cnn            (re-uses data/cnn/dataset.pkl if present; --rebuild to recreate it)
"""
from __future__ import annotations

import glob
import os
import pickle
import sqlite3

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import cnn
from . import config as C
from . import store
from .infra import Structures
from .live import recording_box
from .sar import load_scene

DATASET = os.path.join("data", "cnn", "dataset.pkl")
FIG_DIR = os.path.join("docs", "nn")
PAL = dict(panel="#0a1218", line="#16303a", ink="#d3ece9", muted="#6d8c91", green="#2ef2b0", red="#ff4d5e", cyan="#4fc3ff")


def _scene_set(folder: str, area: str, aoi, ais_db: str, rng) -> pd.DataFrame:
    tif = glob.glob(os.path.join(folder, "*-aoi.tiff"))[0]
    xml = glob.glob(os.path.join(folder, "calibration-*.xml"))[0]
    land = os.path.join("data", area, "coast_osm.geojson")
    scene = load_scene(tif, aoi=aoi, calibration_xml=xml, land_geojson=land if os.path.exists(land) else None)
    ais = store.ais_window(sqlite3.connect(ais_db), scene.time, C.MATCH_WINDOW_S, recording_box(aoi))
    structures = Structures.load(os.path.join("data", area, "infra_osm.geojson"))
    df = cnn.label_scene(scene, ais, structures, area, os.path.basename(folder), rng)
    counts = df.kind.value_counts().to_dict() if len(df) else {}
    print(f"  {os.path.basename(folder):28s} AIS ships {ais.mmsi.nunique():4d}  chips {counts}")
    return df


def build_dataset() -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(cnn.SEED)
    print("training scenes (Danish straits):")
    train = [
        _scene_set(f, os.path.basename(f).split("_")[0], C.TRAIN_AREAS[os.path.basename(f).split("_")[0]], "train_ais.sqlite", rng)
        for f in sorted(glob.glob(os.path.join("scenes", "train", "*")))]
    print("held-out test scenes (Oresund):")
    test = [_scene_set(f, "denmark", C.REGIONS["denmark"], "denmark_hist.sqlite", rng)
            for f in sorted(glob.glob(os.path.join("scenes", "202610*")))]
    train_df = pd.concat([d for d in train if len(d)], ignore_index=True)
    test_df = pd.concat([d for d in test if len(d)], ignore_index=True)
    os.makedirs(os.path.dirname(DATASET), exist_ok=True)
    with open(DATASET, "wb") as f:
        pickle.dump((train_df, test_df), f)
    return train_df, test_df


def cfar_baseline(df: pd.DataFrame) -> np.ndarray:
    """What the pipeline did without a neural network: every CFAR target passing the score and size rules is a ship.
    Random open-sea chips are never proposed by CFAR, so the baseline says 'not ship' for them."""
    return ((df.score >= C.MIN_SCORE) & (df.length_m >= C.MIN_LENGTH_M)).fillna(False).astype(float).values


def choose_threshold(y_val: np.ndarray, p_val: np.ndarray, min_recall: float = 0.9) -> float:
    """Missing a real dark ship costs more than a false alarm, so pick the highest threshold that still keeps
    `min_recall` of the validation ships. Chosen on validation data only, never on the test set."""
    for thr in np.arange(0.95, 0.0, -0.01):
        yh = p_val >= thr
        if yh[y_val == 1].mean() >= min_recall:
            return float(round(thr, 2))
    return 0.05


def accepted_by_cfar(df: pd.DataFrame, p: np.ndarray, thr: float = C.CNN_THRESHOLD) -> dict:
    """The fair comparison. Ship labels only exist for targets CFAR already accepts, so CFAR's recall is 1 by
    construction. The network's job is to clean up what CFAR accepts: count the non-ships it removes and ships it loses."""
    acc = cfar_baseline(df) == 1
    y, keep = df.label.values[acc], p[acc] >= thr
    ships, non = int(y.sum()), int((y == 0).sum())
    kept_ships, kept_non = int((keep & (y == 1)).sum()), int((keep & (y == 0)).sum())
    return dict(targets=int(acc.sum()), ships=ships, non_ships=non,
                ships_kept=kept_ships, ships_lost=ships - kept_ships,
                non_ships_removed=non - kept_non, non_ships_left=kept_non,
                precision_before=ships / max(ships + non, 1), precision_after=kept_ships / max(kept_ships + kept_non, 1))


def _style(ax):
    ax.set_facecolor(PAL["panel"])
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(PAL["line"])
    ax.tick_params(colors=PAL["muted"], labelsize=8)
    ax.grid(color=PAL["line"], linewidth=0.6)


def figures(history, test_df, p_test, report):
    os.makedirs(FIG_DIR, exist_ok=True)
    with plt.rc_context({"font.family": "monospace", "text.color": PAL["ink"]}):
        h = pd.DataFrame(history)
        fig, ax = plt.subplots(1, 2, figsize=(11, 3.4), facecolor=PAL["panel"])
        for a in ax:
            _style(a)
        ax[0].plot(h.epoch, h.train_loss, color=PAL["cyan"], label="train"); ax[0].plot(h.epoch, h.val_loss, color=PAL["green"], label="validation")
        ax[0].set_title("LOSS (BINARY CROSS-ENTROPY)", fontsize=9, loc="left")
        ax[1].plot(h.epoch, h.train_acc, color=PAL["cyan"], label="train"); ax[1].plot(h.epoch, h.val_acc, color=PAL["green"], label="validation")
        ax[1].set_ylim(0.5, 1.01); ax[1].set_title("ACCURACY", fontsize=9, loc="left")
        best = int(h.val_loss.idxmin()) + 1
        for a in ax:
            a.axvline(best, color=PAL["muted"], linestyle="--", linewidth=0.8)
            a.set_xlabel("epoch", color=PAL["muted"], fontsize=8)
            leg = a.legend(frameon=False, fontsize=8)
            for t in leg.get_texts():
                t.set_color(PAL["ink"])
        fig.tight_layout(); fig.savefig(os.path.join(FIG_DIR, "training.png"), dpi=110, facecolor=PAL["panel"]); plt.close(fig)

        fig, ax = plt.subplots(1, 2, figsize=(8, 3.4), facecolor=PAL["panel"])
        for a, key, title in zip(ax, ("cnn", "cfar"), (f"SHIPCNN (THRESHOLD {report.get('threshold', 0.5):g})", "CFAR ONLY (BASELINE)")):
            m = report["test"][key]
            cm = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
            a.imshow(cm, cmap="Greens", vmin=0, vmax=cm.max() * 1.4)
            for (i, j), v in np.ndenumerate(cm):
                a.text(j, i, str(v), ha="center", va="center", fontsize=14, color="#04110c", fontweight="bold")
            a.set_xticks([0, 1], ["pred: not ship", "pred: ship"], fontsize=8, color=PAL["muted"])
            a.set_yticks([0, 1], ["not ship", "ship"], fontsize=8, color=PAL["muted"])
            a.set_title(title, fontsize=9, loc="left")
        fig.tight_layout(); fig.savefig(os.path.join(FIG_DIR, "confusion.png"), dpi=110, facecolor=PAL["panel"]); plt.close(fig)

        # example chips: correct ships, correct structures, and the mistakes
        t = test_df.assign(p=p_test, pred=(p_test >= report["threshold"]).astype(int))
        groups = [("ship, CNN says ship", t[(t.label == 1) & (t.pred == 1)]),
                  ("structure, CNN says not ship", t[(t.kind == "fixed structure") & (t.pred == 0)]),
                  ("ship missed by CNN", t[(t.label == 1) & (t.pred == 0)]),
                  ("not ship, CNN says ship", t[(t.label == 0) & (t.pred == 1)])]
        fig, ax = plt.subplots(4, 6, figsize=(9, 6.6), facecolor=PAL["panel"])
        for r, (title, g) in enumerate(groups):
            g = g.sample(min(6, len(g)), random_state=1) if len(g) else g
            for c in range(6):
                a = ax[r, c]; a.axis("off")
                if c < len(g):
                    a.imshow(g.chip.iloc[c], cmap="gray", vmin=-0.3, vmax=1.2)
                    a.set_title(f"P={g.p.iloc[c]:.2f}", fontsize=7, color=PAL["ink"])
            ax[r, 0].text(-0.15, 0.5, title, transform=ax[r, 0].transAxes, ha="right", va="center", fontsize=8, color=PAL["ink"])
        fig.subplots_adjust(left=0.27, right=0.99, top=0.95, bottom=0.02, hspace=0.35)
        fig.savefig(os.path.join(FIG_DIR, "examples.png"), dpi=110, facecolor=PAL["panel"]); plt.close(fig)


def main(rebuild: bool = False, epochs: int = 60) -> dict:
    if rebuild or not os.path.exists(DATASET):
        train_df, test_df = build_dataset()
    else:
        with open(DATASET, "rb") as f:
            train_df, test_df = pickle.load(f)
    tr, va = cnn.group_split(train_df)
    print(f"\ntrain {len(tr)} chips ({int(tr.label.sum())} ships) | validation {len(va)} ({int(va.label.sum())} ships) | "
          f"test (Oresund) {len(test_df)} ({int(test_df.label.sum())} ships)\n")
    model, history = cnn.train(tr, va, epochs=epochs)
    cnn.save(model)

    p_val = cnn.predict(model, np.stack(va.chip.values))
    p_test = cnn.predict(model, np.stack(test_df.chip.values))
    thr = choose_threshold(va.label.values, p_val)
    report = dict(
        threshold=thr,
        architecture=cnn.describe(model), weights=int(sum(p.numel() for p in model.parameters())),
        training=dict(optimizer=f"Adam (learning rate {cnn.LR:g}, weight decay 1e-4)", loss="binary cross-entropy, positive class weighted",
                      batch=64, epochs=epochs, best_epoch=int(pd.DataFrame(history).val_loss.idxmin()) + 1,
                      augmentation="random 90-degree rotation, horizontal flip, +/-3 px shift"),
        data=dict(train=tr.kind.value_counts().to_dict(), validation=va.kind.value_counts().to_dict(),
                  test=test_df.kind.value_counts().to_dict(),
                  train_areas=sorted(train_df.area.unique().tolist()), test_area="Oresund (held out)"),
        validation=cnn.metrics(va.label.values, p_val, thr),
        test=dict(cnn=cnn.metrics(test_df.label.values, p_test, thr),
                  cnn_at_0_5=cnn.metrics(test_df.label.values, p_test, 0.5),
                  cfar=cnn.metrics(test_df.label.values, cfar_baseline(test_df))),
        where_it_matters=accepted_by_cfar(test_df, p_test, thr),
        where_it_matters_at_0_5=accepted_by_cfar(test_df, p_test, 0.5),
        history=history)
    cnn.save_report(report)
    figures(history, test_df, p_test, report)
    print(f"operating threshold chosen on validation (keep >= 90% of ships): {thr}")
    for k in ("cnn", "cnn_at_0_5", "cfar"):
        m = report["test"][k]
        print(f"TEST {k:10s}: accuracy {m['accuracy']:.3f} precision {m['precision']:.3f} recall {m['recall']:.3f} "
              f"F1 {m['f1']:.3f} ROC-AUC {m['roc_auc']:.3f}  (tp {m['tp']} fp {m['fp']} tn {m['tn']} fn {m['fn']})")
    w = report["where_it_matters"]
    print(f"ON TARGETS CFAR ACCEPTS: {w['ships']} ships + {w['non_ships']} non-ships; CNN removes {w['non_ships_removed']} "
          f"non-ships, loses {w['ships_lost']} ships; precision {w['precision_before']:.2f} -> {w['precision_after']:.2f}")
    return report
