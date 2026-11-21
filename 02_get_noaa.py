"""Read GraphCast, Aurora and Pangu-Weather (IFS-initialised) from the NOAA AIWP archive.

Files are ~3-4 GB each, so they are opened remotely and only the needed
variables, times and domain are read (anonymous S3, no account needed).

Usage:  python 02_get_noaa.py            # all events and NOAA models
        python 02_get_noaa.py A Aurora
        python 02_get_noaa.py --inspect  # print the variable names of one file
"""
import sys

import pandas as pd
import s3fs
import xarray as xr

import config as C
from common import NAME_MAP, standardise

BUCKET = "noaa-oar-mlwp-data"
fs = s3fs.S3FileSystem(anon=True)


def key_for(prefix, init):
    return (f"{BUCKET}/{prefix}/{init:%Y}/{init:%m%d}/"
            f"{prefix}_{init:%Y%m%d%H}_f000_f240_06.nc")


def open_remote(key):
    f = fs.open(key, mode="rb", block_size=8 * 2**20, cache_type="readahead")
    return xr.open_dataset(f, engine="h5netcdf", chunks={})


def fetch(model_key, event_key):
    prefix = C.MODELS[model_key]["prefix"]
    inits, valids = C.event_times(event_key)
    for init in inits:
        out = C.fc_path(model_key, event_key, init)
        if out.exists():
            print(f"skip {out.name}")
            continue
        key = key_for(prefix, init)
        if not fs.exists(key):
            print(f"  MISSING in archive: {key}")
            continue
        ds = open_remote(key)
        keep = [n for cands in NAME_MAP.values() for n in cands if n in ds.data_vars]
        ds = ds[keep]
        # Time axis: valid time or lead time; convert to valid_time
        tname = "time" if "time" in ds.dims else [d for d in ds.dims if "time" in d or "step" in d][0]
        t = ds[tname]
        if pd.api.types.is_timedelta64_dtype(t):
            ds = ds.assign_coords({tname: pd.Timestamp(init) + pd.to_timedelta(t.values)})
        ds = ds.rename({tname: "valid_time"})
        ds = ds.sel(valid_time=[pd.Timestamp(v) for v in valids if v >= init], method="nearest")
        std = standardise(ds, init).load()
        std.to_netcdf(out)
        print(f"saved {out.name}  vars={list(std.data_vars)}  times={std.sizes['valid_time']}")


def inspect():
    init = C.event_times("A")[0][0]
    ds = open_remote(key_for(C.MODELS["GraphCast"]["prefix"], init))
    print(ds)


if __name__ == "__main__":
    if "--inspect" in sys.argv:
        inspect()
        sys.exit()
    events = [sys.argv[1]] if len(sys.argv) > 1 else list(C.EVENTS)
    models = [sys.argv[2]] if len(sys.argv) > 2 else [m for m, v in C.MODELS.items() if v["src"] == "noaa"]
    for ev in events:
        for m in models:
            print(f"== {m} event {ev}")
            fetch(m, ev)
