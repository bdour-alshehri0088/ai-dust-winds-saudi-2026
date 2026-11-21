"""Download IFS HRES and AIFS Single forecasts from ECMWF open data on AWS.

Polite downloader: reads each step's .index file, then fetches only the
needed GRIB messages one at a time with byte-range requests, pausing between
requests and backing off when AWS answers 503 (SlowDown). Cropped NetCDF
files go to data/forecasts/. Re-running skips finished files.

Usage:  python 01_get_ecmwf.py            # both events, both models
        python 01_get_ecmwf.py A AIFS     # one event, one model
"""
import json
import sys
import time

import cfgrib
import requests
import xarray as xr

import config as C
from common import standardise

BASE = "https://ecmwf-forecasts.s3.eu-central-1.amazonaws.com"
PARAMS = {"10u", "10v", "msl", "100u", "100v", "10fg"}  # 10fg (gust) only where it exists
PAUSE_S = 1.0
session = requests.Session()


def get(url, headers=None, tries=12):
    wait = 15
    for i in range(tries):
        try:
            r = session.get(url, headers=headers, timeout=120)
            if r.status_code in (200, 206):
                time.sleep(PAUSE_S)
                return r
            if r.status_code == 404:
                return None
            print(f"    HTTP {r.status_code}, waiting {wait}s")
        except requests.RequestException as e:
            print(f"    {type(e).__name__}, waiting {wait}s")
        time.sleep(wait)
        wait = min(wait * 2, 300)
    raise RuntimeError(f"Gave up on {url}")


def step_url(model, init, step):
    return (f"{BASE}/{init:%Y%m%d}/{init:%H}z/{model}/0p25/oper/"
            f"{init:%Y%m%d%H}0000-{step}h-oper-fc")


def fetch_step(model, init, step, fh):
    base = step_url(model, init, step)
    r = get(base + ".index")
    if r is None:
        print(f"    no file for step {step}")
        return 0
    n = 0
    for line in r.text.splitlines():
        rec = json.loads(line)
        if rec.get("param") not in PARAMS or rec.get("levtype", "sfc") != "sfc":
            continue
        a, b = rec["_offset"], rec["_offset"] + rec["_length"] - 1
        msg = get(base + ".grib2", headers={"Range": f"bytes={a}-{b}"})
        fh.write(msg.content)
        n += 1
    return n


def to_netcdf(grib, init, out):
    parts = cfgrib.open_datasets(str(grib), backend_kwargs={"indexpath": ""})
    merged = []
    for p in parts:
        p = p.drop_vars([v for v in ("heightAboveGround", "meanSea", "surface", "number")
                         if v in p.coords], errors="ignore")
        if "step" in p.dims:
            p = p.swap_dims({"step": "valid_time"}).drop_vars("step", errors="ignore")
        elif "valid_time" in p.coords and "valid_time" not in p.dims:
            p = p.expand_dims("valid_time")
        merged.append(p)
    ds = xr.merge(merged, compat="override", join="outer")
    std = standardise(ds, init)
    std.to_netcdf(out)
    return std


def fetch(model_key, event_key):
    model = C.MODELS[model_key]["model"]
    inits, valids = C.event_times(event_key)
    for init in inits:
        out = C.fc_path(model_key, event_key, init)
        if out.exists():
            print(f"skip {out.name}")
            continue
        grib = C.RAW / f"{model_key}_{event_key}_{init:%Y%m%d%H}.grib2"
        with open(grib, "wb") as fh:
            for step in C.steps_for(init, valids):
                n = fetch_step(model, init, step, fh)
                print(f"  {model_key} {init:%Y-%m-%d %H}z step {step}h: {n} fields")
        std = to_netcdf(grib, init, out)
        grib.unlink(missing_ok=True)
        print(f"saved {out.name}  vars={list(std.data_vars)}  times={std.sizes.get('valid_time')}")


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore", category=FutureWarning)
    events = [sys.argv[1]] if len(sys.argv) > 1 else list(C.EVENTS)
    models = [sys.argv[2]] if len(sys.argv) > 2 else [m for m, v in C.MODELS.items() if v["src"] == "ecmwf"]
    for ev in events:
        for m in models:
            print(f"== {m} event {ev}")
            fetch(m, ev)
