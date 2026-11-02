import pandas as pd

from dvd.auto import SETTLE_S, select_passes

AOI = (12.55, 55.55, 12.95, 55.85)
T = pd.Timestamp("2026-11-02T16:52:00Z").timestamp()


def item(id_, when, lon0=12.0, lon1=13.5):
    return {"id": id_, "properties": {"datetime": when},
            "geometry": {"type": "Polygon", "coordinates": [[[lon0, 55.0], [lon1, 55.0], [lon1, 56.5], [lon0, 56.5], [lon0, 55.0]]]}}


def decisions(items, processed=(), rec_start=T - 86400, now=T + 4 * 3600):
    return {it["id"]: d for it, d in select_passes(items, AOI, set(processed), rec_start, now)}


def test_new_fully_covered_pass_is_processed():
    assert decisions([item("A", "2026-11-02T16:52:00Z")])["A"] == "process"


def test_skips_partial_coverage_done_early_and_too_recent():
    d = decisions([item("partial", "2026-11-02T16:52:00Z", lon0=12.8),          # covers ~38% of the sector
                   item("done", "2026-11-02T05:30:00Z"),
                   item("early", "2026-10-30T17:00:00Z"),                         # before the recorder started
                   item("fresh", "2026-11-02T16:52:00Z")],
                  processed={"done"}, rec_start=T - 12 * 3600, now=T + SETTLE_S - 60)
    assert d["partial"].startswith("covers only")
    assert d["done"] == "already processed"
    assert d["early"].startswith("taken before")
    assert d["fresh"].startswith("too recent")


def test_no_recording_means_no_processing():
    assert decisions([item("A", "2026-11-02T16:52:00Z")], rec_start=None)["A"].startswith("taken before")
