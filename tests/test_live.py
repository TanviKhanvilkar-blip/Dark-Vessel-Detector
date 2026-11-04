"""The AIS recorder against a fake local feed that drops, stalls and rejects keys like the real one can."""
import asyncio
import json
import time
from datetime import datetime, timezone

import pytest
import websockets

from dvd import config as C
from dvd import store
from dvd.live import AuthError, ais_status, record_ais


def report(mmsi, lon=72.8, lat=19.0):
    t = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S.123456789 +0000 UTC")
    return json.dumps({"MessageType": "PositionReport", "MetaData": {"MMSI": mmsi, "time_utc": t},
                       "Message": {"PositionReport": {"Latitude": lat, "Longitude": lon, "Sog": 5.0, "Cog": 90.0}}})


async def run_with_feed(script, db, stop_after, **kw):
    """`script(ws, n)` plays connection number n. Returns the subscriptions the server received."""
    subs, count = [], [0]

    async def handler(ws):
        subs.append(json.loads(await ws.recv()))
        count[0] += 1
        await script(ws, count[0])

    async with websockets.serve(handler, "127.0.0.1", 0) as server:
        port = server.sockets[0].getsockname()[1]
        stop = asyncio.Event()
        task = asyncio.create_task(record_ais("k", db, url=f"ws://127.0.0.1:{port}", stop=stop, quiet=True, **kw))
        try:
            await asyncio.wait_for(asyncio.shield(task), stop_after)
        except asyncio.TimeoutError:
            stop.set()
            await task
    return subs


def test_drop_and_stall_are_logged_and_reconnected(tmp_path):
    db = str(tmp_path / "ais.sqlite")

    async def script(ws, n):
        if n == 1:                      # three reports, then the server hangs up
            for m in (1, 2, 3):
                await ws.send(report(m))
        elif n == 2:                    # two reports and a non-position message, then silence (stall)
            await ws.send(report(4)); await ws.send(report(5))
            await ws.send(json.dumps({"MessageType": "ShipStaticData"}))
            await asyncio.sleep(3)      # longer than stall_s; the server waits for handlers on shutdown, so keep short
        else:                           # after the stall the recorder must come back
            await ws.send(report(6))
            await asyncio.sleep(2)

    subs = asyncio.run(run_with_feed(script, db, stop_after=6.0, gap_s=0.5, stall_s=1.5))
    con = store.connect(db)
    assert sorted(r[0] for r in con.execute("SELECT mmsi FROM ais")) == [1, 2, 3, 4, 5, 6]
    assert len(subs) >= 3                                         # dropped once, stalled once
    assert con.execute("SELECT COUNT(*) FROM ais_gaps").fetchone()[0] >= 2
    (s_lat, s_lon), (n_lat, n_lon) = subs[0]["BoundingBoxes"][0]  # aisstream wants [[lat, lon], [lat, lon]]
    w, s, e, n = C.LIVE_AOI
    assert (s_lat, s_lon, n_lat, n_lon) == pytest.approx((s - C.AIS_PAD_DEG, w - C.AIS_PAD_DEG, n + C.AIS_PAD_DEG, e + C.AIS_PAD_DEG))
    assert subs[0]["FilterMessageTypes"] == ["PositionReport", "ShipStaticData"]
    assert len(subs[0]["BoundingBoxes"]) == len(C.REGIONS)       # one connection records every region


def test_rejected_key_stops_instead_of_retrying(tmp_path):
    async def script(ws, n):
        await ws.send(json.dumps({"error": "Api Key Is Not Valid"}))
        await asyncio.sleep(1)

    with pytest.raises(AuthError):
        asyncio.run(run_with_feed(script, str(tmp_path / "a.sqlite"), stop_after=5.0))


def test_restart_after_downtime_logs_a_gap(tmp_path):
    db = str(tmp_path / "r.sqlite")
    con = store.connect(db)
    store.add_ais(con, __import__("pandas").DataFrame([dict(mmsi=9, time=time.time() - 3600, lon=72.8, lat=19.0, sog=1.0, cog=0.0)]))
    con.close()

    async def script(ws, n):
        await ws.send(report(10))
        await asyncio.sleep(1)

    asyncio.run(run_with_feed(script, db, stop_after=1.5))
    start, end = store.connect(db).execute("SELECT start, end FROM ais_gaps").fetchone()
    assert end - start == pytest.approx(3600, abs=30)            # the hour the recorder was down


def test_prune_keeps_recent_ais(tmp_path):
    import pandas as pd
    con = store.connect(str(tmp_path / "p.sqlite"))
    now = 1_790_000_000.0
    store.add_ais(con, pd.DataFrame([dict(mmsi=1, time=now - d * 86400, lon=72.8, lat=19.0, sog=1.0, cog=0.0) for d in (1, 4, 9)]))
    assert store.prune_ais(con, 5, now) == 1
    assert con.execute("SELECT COUNT(*) FROM ais").fetchone()[0] == 2


def test_ais_status_pass_readiness(tmp_path):
    import pandas as pd
    con = store.connect(str(tmp_path / "s.sqlite"))
    t = 1_790_000_000.0
    store.add_ais(con, pd.DataFrame([dict(mmsi=m, time=t + dt, lon=72.8, lat=19.0, sog=1.0, cog=0.0)
                                     for m in (1, 2) for dt in range(-4000, 4001, 60)]))
    assert ais_status(con, now=t + 4000, at=t)["pass_ready"]
    store.add_gap(con, t - 600, t + 600)                          # 20 min outage across the pass
    st = ais_status(con, now=t + 4000, at=t)
    assert not st["pass_ready"] and st["pass_gap_s"] == 1200
    assert not ais_status(con, now=t + 4000, at=t + 10_000)["pass_ready"]   # recording does not cover that time
