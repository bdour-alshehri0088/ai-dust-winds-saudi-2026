"""ERA5 without the CDS queue: read it from the public ARCO-ERA5 copy on Google Cloud.

Same outputs as 08_get_era5.py, so 09_era5_diagnostics.py works with either:
  data/era5/era5_sfc_event{A,B}.nc    10 m wind and MSLP, 6-hourly, D-1 to D+2, 15-40N 25-60E
  data/era5/era5_swvl1_event{A,B}.nc  ERA5 top-layer soil moisture, 00 UTC on each initialisation day
                                      (used in place of ERA5-Land for the Fecan adjustment)
No account or key is needed. Files already downloaded are skipped.

Usage:  pip install gcsfs zarr && python 08b_get_era5_arco.py
"""
from datetime import datetime, timedelta

import pandas as pd
import xarray as xr

import config as C

STORE = "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"
OUT = C.DATA / "era5"
OUT.mkdir(parents=True, exist_ok=True)
LAT, LON = slice(40, 15), slice(25, 60)
RENAME = {"10m_u_component_of_wind": "u10", "10m_v_component_of_wind": "v10",
          "mean_sea_level_pressure": "msl", "volumetric_soil_water_layer_1": "swvl1"}


def save(sub, out):
    """Rename to short names, refuse empty placeholders, write netCDF."""
    sub = sub.rename({k: v for k, v in RENAME.items() if k in sub})
    if "time" in sub.dims and "swvl1" not in sub:
        sub = sub.rename({"time": "valid_time"})
    frac = float(sum(sub[v].isnull().mean() for v in sub.data_vars) / len(sub.data_vars))
    if frac > 0.5:
        print(f"{out.name}: not yet filled in ARCO-ERA5 ({frac:.0%} missing), skipped")
        return
    sub.to_netcdf(out)
    print(f"saved {out.name}")


def main():
    print("opening ARCO-ERA5 (public, read-only) ...")
    ds = xr.open_zarr(STORE, chunks=None, storage_options={"token": "anon"})
    for ev, info in C.EVENTS.items():
        d0 = datetime.strptime(info["date"], "%Y-%m-%d")
        out = OUT / f"era5_sfc_event{ev}.nc"
        if out.exists():
            print(f"skip {out.name}")
        else:
            times = pd.date_range(d0 - timedelta(days=1), d0 + timedelta(days=2, hours=18), freq="6h")
            sub = ds[["10m_u_component_of_wind", "10m_v_component_of_wind", "mean_sea_level_pressure"]] \
                .sel(time=times, latitude=LAT, longitude=LON).load()
            save(sub, out)
        out = OUT / f"era5_swvl1_event{ev}.nc"
        if not out.exists():
            inits = [d0 - timedelta(days=k) for k in C.INIT_OFFSETS_DAYS]
            sub = ds[["volumetric_soil_water_layer_1"]].sel(time=inits, latitude=LAT, longitude=LON).load()
            save(sub, out)


if __name__ == "__main__":
    main()
