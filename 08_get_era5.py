"""Download ERA5 and ERA5-Land from the Copernicus Climate Data Store (CDS).

  era5      : 10 m wind, MSLP (single levels) and 850 hPa wind, 6-hourly, from the day before to
              two days after each storm, 15-40N 25-60E. Used for the synoptic description
              (Section 3), MSLP maps (Figure 3), activity ratio and wind spectra (Section 5.6).
  era5land  : top-layer soil moisture (swvl1) at 00 UTC on each initialisation day, for the
              Fecan et al. (1999) moisture-adjusted threshold (Section 4.7).

Needs ~/.cdsapirc with your CDS key (never put the key in this folder), and the ERA5 and
ERA5-Land licences accepted once on the CDS website.
Outputs: data/era5/era5_sfc_event{A,B}.nc, era5_850_event{A,B}.nc, era5land_swvl1_event{A,B}.nc

Usage:  python 08_get_era5.py            # all
        python 08_get_era5.py era5 | era5land
        python 08_get_era5.py sfc   # skip the slower 850 hPa requests
"""
import sys
import zipfile
from datetime import datetime, timedelta

import cdsapi

import config as C

OUT = C.DATA / "era5"
OUT.mkdir(parents=True, exist_ok=True)
AREA = [40, 25, 15, 60]                      # N, W, S, E
TIMES = ["00:00", "06:00", "12:00", "18:00"]


def days(start, n):
    return [start + timedelta(days=i) for i in range(n)]


def ymd(dates):
    return {"year": sorted({f"{d:%Y}" for d in dates}), "month": sorted({f"{d:%m}" for d in dates}),
            "day": sorted({f"{d:%d}" for d in dates})}


def fetch(c, dataset, request, out):
    if out.exists():
        print(f"skip {out.name}")
        return
    tmp = out.with_suffix(".download")
    c.retrieve(dataset, {**request, "data_format": "netcdf", "download_format": "unarchived"}, str(tmp))
    if zipfile.is_zipfile(tmp):                  # CDS sometimes zips netcdf output
        with zipfile.ZipFile(tmp) as z:
            name = z.namelist()[0]
            out.write_bytes(z.read(name))
        tmp.unlink()
    else:
        tmp.rename(out)
    print(f"saved {out.name}")


def era5(c, levels=True):
    for ev, info in C.EVENTS.items():
        d0 = datetime.strptime(info["date"], "%Y-%m-%d")
        dates = days(d0 - timedelta(days=1), 4)            # D-1 ... D+2
        fetch(c, "reanalysis-era5-single-levels",
              {"product_type": ["reanalysis"], **ymd(dates), "time": TIMES, "area": AREA,
               "variable": ["10m_u_component_of_wind", "10m_v_component_of_wind",
                            "mean_sea_level_pressure"]},
              OUT / f"era5_sfc_event{ev}.nc")
        if not levels:
            continue
        fetch(c, "reanalysis-era5-pressure-levels",
              {"product_type": ["reanalysis"], **ymd(dates), "time": TIMES, "area": AREA,
               "pressure_level": ["850"],
               "variable": ["u_component_of_wind", "v_component_of_wind", "geopotential"]},
              OUT / f"era5_850_event{ev}.nc")


def era5land(c):
    for ev, info in C.EVENTS.items():
        d0 = datetime.strptime(info["date"], "%Y-%m-%d")
        dates = days(d0 - timedelta(days=max(C.INIT_OFFSETS_DAYS)), max(C.INIT_OFFSETS_DAYS) + 1)
        fetch(c, "reanalysis-era5-land",
              {**ymd(dates), "time": ["00:00"], "area": AREA,
               "variable": ["volumetric_soil_water_layer_1"]},
              OUT / f"era5land_swvl1_event{ev}.nc")


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    c = cdsapi.Client()
    if what in ("era5", "all", "sfc"):
        era5(c, levels=(what != "sfc"))
    if what in ("era5land", "all", "sfc"):
        era5land(c)
