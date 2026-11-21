"""Shared settings for the AI dust-wind evaluation.

Edit station lists or the domain here; every script reads this file.
"""
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
RAW = DATA / "raw"          # downloaded GRIB / partial files
FC = DATA / "forecasts"     # cropped, standardised forecast NetCDF
OBS = DATA / "obs"          # station observations
OUT = ROOT / "results"      # tables and figures
for p in (RAW, FC, OBS, OUT):
    p.mkdir(parents=True, exist_ok=True)

# Verification domain (degrees). Longitudes in -180..180.
LAT_MIN, LAT_MAX = 20.0, 34.0
LON_MIN, LON_MAX = 34.0, 52.0

# Events: the day of the storm (UTC) and the stations to verify.
EVENTS = {
    "A": {"date": "2026-01-13",
          "stations": ["OERR", "OERF", "OETR", "OESK", "OEPA", "OEHL", "OEGS", "OERK"]},
    "B": {"date": "2026-05-03",
          "stations": ["OERR", "OERF", "OETR", "OESK", "OETB", "OEHL"]},
}

# Initialisations: 00 UTC on D-5 ... D0
INIT_OFFSETS_DAYS = [5, 4, 3, 2, 1, 0]
INIT_HOUR = 0
# Valid times verified: 6-hourly from D-1 00 UTC to D+2 00 UTC (pre-event, event, post-event),
# so that strong-wind hours without dust also enter the false-alarm statistics
VALID_HOURS = list(range(-24, 49, 6))

# Models. "src" decides which downloader handles them.
MODELS = {
    "IFS":    {"src": "ecmwf", "model": "ifs"},
    "AIFS":   {"src": "ecmwf", "model": "aifs-single"},
    "GraphCast": {"src": "noaa", "prefix": "GRAP_v100_IFS"},
    "Aurora":    {"src": "noaa", "prefix": "AURO_v100_IFS"},
    "Pangu":     {"src": "noaa", "prefix": "PANG_v100_IFS"},
}

# Verification settings
MATCH_TOL_MIN = 30          # forecast-observation time tolerance
DUST_WINDOW_H = 3           # +/- hours for a dust report to count as a hit
DIR_MIN_SPEED = 3.0         # m/s, minimum observed speed for direction error
FIXED_THRESHOLD = 6.5       # m/s, sensitivity test (Kurosaki and Mikami, 2005)
DEFAULT_THRESHOLD = 7.0     # m/s, used when a station climatology is too short
CLIM_YEARS = (2016, 2025)   # years for station-threshold climatology
KT2MS = 0.514444


def event_times(event_key):
    """Return (init datetimes, valid datetimes) for an event."""
    day = datetime.strptime(EVENTS[event_key]["date"], "%Y-%m-%d")
    inits = [day - timedelta(days=d) + timedelta(hours=INIT_HOUR) for d in INIT_OFFSETS_DAYS]
    valids = [day + timedelta(hours=h) for h in VALID_HOURS]
    return inits, valids


def steps_for(init, valids):
    """Forecast steps (hours) from an init to each valid time, keeping steps >= 0."""
    return [int((v - init).total_seconds() // 3600) for v in valids if v >= init]


def fc_path(model, event_key, init):
    return FC / f"{model}_{event_key}_{init:%Y%m%d%H}.nc"
