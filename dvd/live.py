"""Live data hooks: the always-on aisstream recorder and a Copernicus scene finder.

The recorder is tested against a local fake feed (tests/test_live.py); the real aisstream service needs a key.
Docs: https://aisstream.io/documentation and https://documentation.dataspace.copernicus.eu/APIs/OData.html
"""
from __future__ import annotations

import asyncio
import json
import time
from datetime import datetime, timezone

from . import config as C
from . import store

AISSTREAM_URL = "wss://stream.aisstream.io/v0/stream"


class AuthError(RuntimeError):
    pass


def parse_position_report(raw: str | bytes | dict) -> dict | None:
    """aisstream PositionReport -> row for the `ais` table (None for any other message)."""
    msg = json.loads(raw) if not isinstance(raw, dict) else raw
    if msg.get("MessageType") != "PositionReport":
        return None
    meta, rep = msg["MetaData"] if "MetaData" in msg else msg.get("Metadata", {}), msg["Message"]["PositionReport"]
    t = datetime.fromisoformat(meta["time_utc"].split(".")[0].replace(" +0000 UTC", "").replace(" ", "T")).replace(tzinfo=timezone.utc)
    sog, cog = rep.get("Sog"), rep.get("Cog")
    if sog is None or cog is None or sog >= 102.3 or cog >= 360:      # 102.3 kn / 360 deg = "not available" in AIS
        return None
    return dict(mmsi=int(meta["MMSI"]), time=t.timestamp(), lon=rep["Longitude"], lat=rep["Latitude"], sog=sog, cog=cog)


def recording_box(aoi, pad_deg: float = C.AIS_PAD_DEG):
    """Record a margin around the AOI: a ship 15 km outside at image time - 30 min can be inside at image time."""
    west, south, east, north = aoi
    return west - pad_deg, south - pad_deg, east + pad_deg, north + pad_deg


async def record_ais(api_key: str, db_path: str, aois=tuple(C.REGIONS.values()), url: str = AISSTREAM_URL,
                     gap_s: float = C.AIS_GAP_THRESHOLD_S, stall_s: float = C.AIS_STALL_S,
                     flush_s: float = 5.0, stop: asyncio.Event | None = None, quiet: bool = False) -> None:
    """Record AIS for one or more areas over a single connection until `stop` is set (or forever).

    Gap logging is based on silence, not on connection state: whenever the time since the last received message
    exceeds `gap_s`, that interval is written to `ais_gaps`. This one rule covers dropped connections, a feed that
    stays connected but goes quiet (stall), and the recorder itself being down (restart, PC asleep), because on
    start-up the last stored message time is used as the starting point.
    """
    import websockets

    boxes = []
    for aoi in aois:
        west, south, east, north = recording_box(aoi)
        boxes.append([[south, west], [north, east]])          # aisstream wants [[lat, lon], [lat, lon]]
    sub = json.dumps({"APIKey": api_key, "BoundingBoxes": boxes, "FilterMessageTypes": ["PositionReport"]})
    con = store.connect(db_path)
    last_rx = store.last_ais_time(con)          # None on a fresh database
    buf: list[dict] = []
    last_flush = last_report = time.time()
    n_since_report, backoff, empty_closes = 0, 1.0, 0

    def log(msg):
        if not quiet:
            print(datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ"), msg, flush=True)

    def flush():
        nonlocal last_flush
        if buf:
            con.executemany("INSERT INTO ais VALUES (:mmsi,:time,:lon,:lat,:sog,:cog)", buf)
            con.commit()
            buf.clear()
        last_flush = time.time()

    async def wait_or_stop(seconds):
        if stop is None:
            await asyncio.sleep(seconds)
        else:
            try:
                await asyncio.wait_for(stop.wait(), seconds)
            except asyncio.TimeoutError:
                pass

    try:
        while not (stop and stop.is_set()):
            try:
                async with websockets.connect(url, open_timeout=30, ping_interval=20, ping_timeout=20) as ws:
                    await ws.send(sub)
                    connected_at = time.time()
                    got_message = False
                    log("connected")
                    while not (stop and stop.is_set()):
                        try:
                            raw = await asyncio.wait_for(ws.recv(), timeout=min(stall_s, 1.0))
                        except asyncio.TimeoutError:
                            # silence is measured from this connection's start or its last message, whichever is later
                            if time.time() - max(last_rx or 0.0, connected_at) > stall_s:
                                raise TimeoutError(f"no messages for {stall_s:.0f}s (stalled feed)")
                            continue
                        msg = json.loads(raw)
                        got_message, empty_closes = True, 0
                        if "error" in msg:
                            err = str(msg["error"])
                            if "api" in err.lower() or "key" in err.lower():
                                raise AuthError(err)
                            raise RuntimeError(err)
                        now = time.time()
                        if last_rx is not None and now - last_rx > gap_s:
                            store.add_gap(con, last_rx, now)
                            log(f"gap of {now - last_rx:.0f}s recorded")
                        last_rx, backoff = now, 1.0
                        row = parse_position_report(msg)
                        if row:
                            buf.append(row)
                            n_since_report += 1
                        if len(buf) >= 500 or now - last_flush > flush_s:
                            flush()
                        if now - last_report > 60:
                            log(f"{n_since_report} position reports in the last {now - last_report:.0f}s")
                            n_since_report, last_report = 0, now
            except AuthError as exc:
                log(f"aisstream rejected the API key: {exc}. Stopping.")
                raise
            except Exception as exc:                   # beta service: expect drops
                flush()
                if stop and stop.is_set():
                    break
                log(f"feed dropped ({exc!r}); reconnecting in {backoff:.0f}s")
                if not locals().get("got_message", True):
                    empty_closes += 1
                    if empty_closes == 3:
                        log(f"aisstream keeps closing before sending anything: usually a wrong API key "
                            f"(the key in use is {len(api_key)} characters long). Stop with Ctrl+C and set it again.")
                await wait_or_stop(backoff)
                backoff = min(backoff * 2, 60)
    finally:
        flush()
        con.close()


def ais_status(con, now: float | None = None, at: float | None = None, aoi=None) -> dict:
    """Health of the recording (within `aoi`'s recording box if given), and (with `at`) whether a pass at that time
    has usable AIS. Feed gaps are global: one connection serves every area."""
    now = now or time.time()
    where, args = "1=1", ()
    if aoi is not None:
        w, s, e, n_ = recording_box(aoi)
        where, args = "lon BETWEEN ? AND ? AND lat BETWEEN ? AND ?", (w, e, s, n_)
    n, n_ship, t0, t1 = con.execute(f"SELECT COUNT(*), COUNT(DISTINCT mmsi), MIN(time), MAX(time) FROM ais WHERE {where}", args).fetchone()
    last_hour = con.execute(f"SELECT COUNT(*) FROM ais WHERE time > ? AND {where}", (now - 3600, *args)).fetchone()[0]
    gaps = con.execute("SELECT COUNT(*), COALESCE(SUM(MIN(end, ?) - MAX(start, ?)), 0) FROM ais_gaps WHERE end > ?",
                       (now, now - 86400, now - 86400)).fetchone()
    out = dict(messages=n, ships=n_ship, first=t0, last=t1, minutes_since_last=(now - t1) / 60 if t1 else None,
               messages_last_hour=last_hour, gaps_last_24h=gaps[0], gap_minutes_last_24h=gaps[1] / 60)
    if at is not None:
        w = C.MATCH_WINDOW_S
        in_win = con.execute(f"SELECT COUNT(*), COUNT(DISTINCT mmsi) FROM ais WHERE time BETWEEN ? AND ? AND {where}",
                             (at - w, at + w, *args)).fetchone()
        gap = store.gap_seconds(con, at, w)
        out.update(pass_messages=in_win[0], pass_ships=in_win[1], pass_gap_s=gap,
                   pass_ready=in_win[0] > 0 and gap <= C.MAX_AIS_GAP_S and (t0 or 1e18) <= at - w and (t1 or 0) >= at + w)
    return out


def find_scenes(days: int = 7, aoi=C.LIVE_AOI) -> list[dict]:
    """List recent Sentinel-1 GRD scenes intersecting the AOI via the Copernicus Data Space OData catalogue (no login needed)."""
    import requests

    west, south, east, north = aoi
    poly = f"POLYGON(({west} {south},{east} {south},{east} {north},{west} {north},{west} {south}))"
    since = datetime.fromtimestamp(time.time() - days * 86400, timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z")
    flt = (f"Collection/Name eq 'SENTINEL-1' and contains(Name,'GRD') and OData.CSC.Intersects(area=geography'SRID=4326;{poly}')"
           f" and ContentDate/Start gt {since}")
    r = requests.get("https://catalogue.dataspace.copernicus.eu/odata/v1/Products",
                     params={"$filter": flt, "$orderby": "ContentDate/Start desc", "$top": 20}, timeout=30)
    r.raise_for_status()
    return [{"name": p["Name"], "start": p["ContentDate"]["Start"], "id": p["Id"]} for p in r.json()["value"]]
