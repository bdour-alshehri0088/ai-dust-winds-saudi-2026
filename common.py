"""Helpers to put every model's output into one standard form.

Standard dataset: variables u10, v10 (m/s), msl (hPa) and, when present,
u100, v100, gust; coordinates valid_time, latitude (ascending), longitude (-180..180).
"""
import numpy as np
import xarray as xr

import config as C

# Candidate names seen across ECMWF GRIB (cfgrib) and NOAA AIWP NetCDF files
NAME_MAP = {
    "u10": ["u10", "10u", "u10m", "U10", "u_10m", "UGRD_10maboveground"],
    "v10": ["v10", "10v", "v10m", "V10", "v_10m", "VGRD_10maboveground"],
    "msl": ["msl", "mslp", "MSL", "prmsl", "mean_sea_level_pressure", "PRMSL_meansealevel"],
    "u100": ["u100", "100u", "u100m"],
    "v100": ["v100", "100v", "v100m"],
    "gust": ["fg10", "i10fg", "10fg", "gust"],
}
LAT_NAMES = ["latitude", "lat"]
LON_NAMES = ["longitude", "lon"]
TIME_NAMES = ["valid_time", "time"]


def _find(ds, names):
    for n in names:
        if n in ds.variables:
            return n
    return None


def standardise(ds, init=None):
    """Rename variables/coords, crop to the domain, convert units."""
    lat = _find(ds, LAT_NAMES)
    lon = _find(ds, LON_NAMES)
    if lat is None or lon is None:
        raise ValueError(f"No lat/lon in dataset: {list(ds.variables)}")
    ds = ds.rename({lat: "latitude", lon: "longitude"})

    # Longitudes to -180..180, latitudes ascending
    if float(ds.longitude.max()) > 180:
        ds = ds.assign_coords(longitude=(((ds.longitude + 180) % 360) - 180))
    ds = ds.sortby("longitude").sortby("latitude")
    ds = ds.sel(latitude=slice(C.LAT_MIN, C.LAT_MAX), longitude=slice(C.LON_MIN, C.LON_MAX))

    out = {}
    for std, cands in NAME_MAP.items():
        n = _find(ds, cands)
        if n is not None:
            da = ds[n]
            # Drop singleton height/level dims
            for d in list(da.dims):
                if d not in ("latitude", "longitude") and da.sizes[d] == 1:
                    da = da.isel({d: 0}, drop=True)
            out[std] = da
    if not {"u10", "v10"} <= out.keys():
        raise ValueError(f"10 m wind not found. Variables: {list(ds.data_vars)}")

    std_ds = xr.Dataset(out)
    if "msl" in std_ds and float(std_ds["msl"].max()) > 2000:  # Pa -> hPa
        std_ds["msl"] = std_ds["msl"] / 100.0
    if init is not None:
        std_ds.attrs["init"] = str(init)
    return std_ds


def wind_speed(u, v):
    return np.hypot(u, v)


def wind_dir(u, v):
    """Meteorological direction the wind blows FROM, degrees 0-360."""
    return (np.degrees(np.arctan2(-u, -v)) + 360.0) % 360.0


def dir_error(df, do):
    """Smallest signed angle forecast minus observed, degrees in [-180, 180)."""
    return ((df - do + 180.0) % 360.0) - 180.0


def dup(u, ut):
    """Dust uplift potential (Marsham et al., 2011); zero below threshold."""
    u = np.asarray(u, dtype=float)
    ut = np.broadcast_to(np.asarray(ut, dtype=float), u.shape)
    with np.errstate(divide="ignore", invalid="ignore"):
        val = u**3 * (1 + ut / u) * (1 - ut**2 / u**2)
    return np.where(u > ut, val, 0.0)
