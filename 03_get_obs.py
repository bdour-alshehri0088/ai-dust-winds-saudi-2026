"""Station observations from the IEM METAR archive (or NCM files if present).

Usage:  python 03_get_obs.py events     # event days (+/-1 day)
        python 03_get_obs.py clim       # 2016-2025 climatology for thresholds
"""
import sys
import time
from datetime import datetime, timedelta
from io import StringIO

import pandas as pd
import requests

import config as C

IEM = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"
FIELDS = ["sknt", "drct", "gust", "vsby", "wxcodes", "mslp", "alti"]  # alti = QNH (inHg)
PAUSE_S = 6  # IEM limits request rate; wait between calls


def iem(stations, start, end):
    params = [("station", s) for s in stations] + [("data", f) for f in FIELDS] + [
        ("year1", start.year), ("month1", start.month), ("day1", start.day),
        ("year2", end.year), ("month2", end.month), ("day2", end.day),
        ("tz", "Etc/UTC"), ("format", "onlycomma"), ("latlon", "yes"),
        ("missing", "empty"), ("trace", "empty"),
        ("report_type", "3"), ("report_type", "4")]
    for attempt in range(6):
        r = requests.get(IEM, params=params, timeout=600)
        if r.status_code == 429:
            wait = 30 * (attempt + 1)
            print(f"  rate-limited, waiting {wait}s")
            time.sleep(wait)
            continue
        r.raise_for_status()
        time.sleep(PAUSE_S)
        return pd.read_csv(StringIO(r.text))
    raise RuntimeError("IEM kept refusing requests; try again in a few minutes")


def has_alti(path):
    """True if the file exists and already holds the QNH column (re-download older files)."""
    return path.exists() and "alti" in open(path).readline()


def events():
    for ev, info in C.EVENTS.items():
        day = datetime.strptime(info["date"], "%Y-%m-%d")
        todo = [s for s in info["stations"]
                if not (C.OBS / f"ncm_{s}.csv").exists() and not has_alti(C.OBS / f"event{ev}_{s}.csv")]
        if not todo:
            continue
        df = iem(todo, day - timedelta(days=2), day + timedelta(days=3))  # one request, all stations
        for st in todo:
            part = df[df["station"] == st]
            part.to_csv(C.OBS / f"event{ev}_{st}.csv", index=False)
            print(f"event{ev}_{st}.csv: {len(part)} reports")


def clim():
    stations = sorted({s for e in C.EVENTS.values() for s in e["stations"]})
    y0, y1 = C.CLIM_YEARS
    for st in stations:
        out = C.OBS / f"clim_{st}.csv"
        if out.exists():
            continue
        df = iem([st], datetime(y0, 1, 1), datetime(y1 + 1, 1, 1))  # one request per station
        df.to_csv(out, index=False)
        print(f"{out.name}: {len(df)} reports")


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "events"
    {"events": events, "clim": clim}[what]()
