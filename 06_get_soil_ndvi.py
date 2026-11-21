"""Download the surface inputs of the susceptibility index.

  soil : SoilGrids 2.0 sand, silt and clay, 0-5 cm (Poggio et al., 2021), ISRIC WCS,
         resampled by the server to 0.025 deg over the study domain + 1 deg margin
  ndvi : NOAA CDR VIIRS daily NDVI (0.05 deg, public AWS bucket, no login), combined into a
         maximum-value composite over the 16 days before each storm

Outputs in data/susceptibility/:  sand_0-5cm.tif, silt_0-5cm.tif, clay_0-5cm.tif,
                                  ndvi_eventA.nc, ndvi_eventB.nc

Usage:  python 06_get_soil_ndvi.py          # both
        python 06_get_soil_ndvi.py soil     # or only one part
        python 06_get_soil_ndvi.py ndvi
"""
import sys
import time
from datetime import datetime, timedelta

import requests
import s3fs
import xarray as xr

import config as C

OUT = C.DATA / "susceptibility"
OUT.mkdir(parents=True, exist_ok=True)
PAD = 1.0
W, S, E, N = C.LON_MIN - PAD, C.LAT_MIN - PAD, C.LON_MAX + PAD, C.LAT_MAX + PAD
SOIL_RES = 0.025
WCS = "https://maps.isric.org/mapserv"
NDVI_DAYS = 16
EPSG4326 = "http://www.opengis.net/def/crs/EPSG/0/4326"


# ---------------------------------------------------------------- soil
def soil_params(var, scaled):
    p = [("map", f"/map/{var}.map"), ("SERVICE", "WCS"), ("VERSION", "2.0.1"),
         ("REQUEST", "GetCoverage"), ("COVERAGEID", f"{var}_0-5cm_mean"), ("FORMAT", "image/tiff"),
         ("SUBSET", f"long({W},{E})"), ("SUBSET", f"lat({S},{N})"),
         ("SUBSETTINGCRS", EPSG4326), ("OUTPUTCRS", EPSG4326)]
    if scaled:
        p.append(("SCALESIZE", f"long({round((E - W) / SOIL_RES)}),lat({round((N - S) / SOIL_RES)})"))
    return p


def soil():
    for var in ("sand", "silt", "clay"):
        out = OUT / f"{var}_0-5cm.tif"
        if out.exists() and out.stat().st_size > 10_000:
            print(f"skip {out.name}")
            continue
        scaled = True
        for attempt in range(8):
            try:
                r = requests.get(WCS, params=soil_params(var, scaled), timeout=600)
            except requests.RequestException as e:
                print(f"  {var}: {type(e).__name__}, waiting {20 * (attempt + 1)} s")
                time.sleep(20 * (attempt + 1))
                continue
            if r.status_code == 200 and r.content[:2] in (b"II", b"MM"):   # a real GeoTIFF
                out.write_bytes(r.content)
                print(f"saved {out.name} ({len(r.content) / 1e6:.1f} MB)")
                break
            if 400 <= r.status_code < 500 and scaled:
                print(f"  {var}: server refused resampling, asking for native resolution instead")
                scaled = False
                continue
            print(f"  {var}: HTTP {r.status_code}, waiting {20 * (attempt + 1)} s")
            time.sleep(20 * (attempt + 1))
        else:
            raise RuntimeError(f"SoilGrids kept failing for {var}; try again later")


# ---------------------------------------------------------------- ndvi
def ndvi():
    fs = s3fs.S3FileSystem(anon=True)
    for ev, info in C.EVENTS.items():
        out = OUT / f"ndvi_event{ev}.nc"
        if out.exists():
            print(f"skip {out.name}")
            continue
        day0 = datetime.strptime(info["date"], "%Y-%m-%d")
        stack = []
        for k in range(1, NDVI_DAYS + 1):
            d = day0 - timedelta(days=k)
            keys = fs.glob(f"noaa-cdr-ndvi-pds/data/{d:%Y}/*_{d:%Y%m%d}_*.nc")
            if not keys:
                print(f"  {d:%Y-%m-%d}: no file")
                continue
            with fs.open(keys[0], "rb", block_size=8 * 2**20) as f:
                ds = xr.open_dataset(f, engine="h5netcdf")
                lat_desc = float(ds.latitude[0]) > float(ds.latitude[-1])
                lat = slice(N, S) if lat_desc else slice(S, N)
                v = ds["NDVI"].sel(latitude=lat, longitude=slice(W, E)).load().squeeze(drop=True)
            stack.append(v.where((v > -0.2) & (v <= 1.0)))
            print(f"  {d:%Y-%m-%d}: {keys[0].split('/')[-1]}")
        if not stack:
            print(f"event {ev}: no NDVI files found")
            continue
        comp = xr.concat(stack, "time").max("time")   # maximum-value composite limits cloud effects
        comp.attrs = {"long_name": f"{NDVI_DAYS}-day maximum NDVI before event {ev}",
                      "source": "NOAA CDR VIIRS NDVI (noaa-cdr-ndvi-pds)", "days_used": len(stack)}
        comp.to_dataset(name="ndvi").sortby("latitude").to_netcdf(out)
        print(f"saved {out.name} from {len(stack)} days")


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    if what in ("soil", "all"):
        soil()
    if what in ("ndvi", "all"):
        ndvi()
