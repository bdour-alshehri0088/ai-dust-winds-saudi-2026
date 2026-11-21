"""Surface dust susceptibility E = S x G x B on the 0.25 deg model grid (Section 4.7).

  S  topographic source function (Ginoux et al., 2001) from the 90 m SRTM DEM:
     S = ((z_max - z) / (z_max - z_min))^5 within a 10 x 10 deg window
  G  soil weight: sand + silt fraction of SoilGrids 0-5 cm, rescaled to 0-1 over the domain
  B  bare fraction: 1 - f_v, with f_v = (NDVI - 0.05) / (0.80 - 0.05) clipped to 0-1,
     from the 16-day maximum NDVI before each storm
  E  S x G x B, divided by its domain maximum (0-1)

Grid: cell centres on multiples of 0.25 deg, the same points as the forecast fields.
Inputs: the DEM (argument or default path), data/susceptibility/{sand,silt,clay}_0-5cm.tif,
        data/susceptibility/ndvi_event{A,B}.nc  (from 06_get_soil_ndvi.py)
Outputs: data/susceptibility/surface_event{A,B}.nc, results/fig_surface.png

Usage:  python 05_susceptibility.py [path/to/dem.tif]
"""
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import xarray as xr
from rasterio.transform import from_origin
from rasterio.warp import Resampling, reproject
from rasterio.windows import Window
from scipy.ndimage import maximum_filter, minimum_filter

import config as C

DEM = Path(sys.argv[1]) if len(sys.argv) > 1 else \
    Path.home() / "Downloads" / "merged_dem_middle_east_fixed.tif"
SUS = C.DATA / "susceptibility"
RES = 0.25
WINDOW_DEG = 10.0
MIN_LAND = 0.5
NDVI_BARE, NDVI_FULL = 0.05, 0.80
warnings.filterwarnings("ignore", category=RuntimeWarning)

LAT = np.arange(C.LAT_MIN, C.LAT_MAX + RES / 2, RES)
LON = np.arange(C.LON_MIN, C.LON_MAX + RES / 2, RES)
DST = from_origin(C.LON_MIN - RES / 2, C.LAT_MAX + RES / 2, RES, RES)   # rows run north to south
STATIONS = {"OERR": (30.91, 41.14), "OERF": (29.63, 43.49), "OETR": (31.69, 38.73),
            "OESK": (29.79, 40.10), "OETB": (28.37, 36.62), "OEPA": (28.34, 46.13),
            "OEHL": (27.44, 41.69), "OEGS": (26.30, 43.77), "OERK": (24.96, 46.70)}


def dem_cells(path):
    """Mean elevation of 0.25 deg cells centred on multiples of 0.25 deg, over the whole DEM."""
    with rasterio.open(path) as r:
        n = int(round(RES / r.res[0]))
        off = n // 2                                   # shift so cells are centred on x.00, x.25 ...
        H, W = (r.height - off) // n, (r.width - off) // n
        z = np.full((H, W), np.nan, dtype="float32")
        for i in range(H):
            a = r.read(1, window=Window(off, off + i * n, W * n, n)).astype("float32")
            a[a == 0] = np.nan                         # sea
            a = a.reshape(n, W, n)
            row = np.nanmean(a, axis=(0, 2))
            row[np.isfinite(a).mean(axis=(0, 2)) < MIN_LAND] = np.nan
            z[i] = row
        lat = r.bounds.top - (off + (np.arange(H) + 0.5) * n) * r.res[1]
        lon = r.bounds.left + (off + (np.arange(W) + 0.5) * n) * r.res[0]
    return xr.DataArray(z, coords={"latitude": np.round(lat, 4), "longitude": np.round(lon, 4)},
                        dims=("latitude", "longitude"))


def ginoux_S(z):
    k = int(round(WINDOW_DEG / RES)) + 1
    v = z.values
    zmax = maximum_filter(np.where(np.isnan(v), -np.inf, v), size=k, mode="constant", cval=-np.inf)
    zmin = minimum_filter(np.where(np.isnan(v), np.inf, v), size=k, mode="constant", cval=np.inf)
    S = np.where(np.isnan(v), np.nan, ((zmax - v) / (zmax - zmin)) ** 5)
    return z.copy(data=S.astype("float32"))


def to_grid(path, nodata=None):
    """Average a GeoTIFF onto the model grid."""
    out = np.full((len(LAT), len(LON)), np.nan, dtype="float32")
    with rasterio.open(path) as r:
        reproject(rasterio.band(r, 1), out, src_nodata=nodata, dst_transform=DST, dst_crs="EPSG:4326",
                  dst_nodata=np.nan, resampling=Resampling.average)
    return xr.DataArray(out[::-1], coords={"latitude": LAT, "longitude": LON}, dims=("latitude", "longitude"))


def ndvi_grid(ev):
    v = xr.open_dataset(SUS / f"ndvi_event{ev}.nc")["ndvi"]
    return v.coarsen(latitude=5, longitude=5, boundary="trim").mean() \
            .interp(latitude=LAT, longitude=LON, method="nearest")


def main():
    z = dem_cells(DEM).sortby("latitude")
    S = ginoux_S(z).sel(latitude=LAT, longitude=LON, method="nearest")
    S = S.assign_coords(latitude=LAT, longitude=LON)
    sea = np.isnan(S)

    sand, silt, clay = (to_grid(SUS / f"{v}_0-5cm.tif", nodata=0) / 10 for v in ("sand", "silt", "clay"))
    fine = sand + silt                                  # percent
    lo, hi = float(fine.where(~sea).quantile(0.01)), float(fine.where(~sea).quantile(0.99))
    G = ((fine - lo) / (hi - lo)).clip(0, 1).where(~sea)

    for ev in C.EVENTS:
        ndvi = ndvi_grid(ev)
        fv = ((ndvi - NDVI_BARE) / (NDVI_FULL - NDVI_BARE)).clip(0, 1)
        B = (1 - fv).where(~sea)
        E = S * G * B
        E = E / float(E.max())
        ds = xr.Dataset({"S": S, "G": G, "B": B, "E": E, "ndvi": ndvi.where(~sea),
                         "sand": sand, "silt": silt, "clay": clay})
        ds.attrs = {"event": ev, "ndvi_bare": NDVI_BARE, "ndvi_full": NDVI_FULL,
                    "G_range_percent": f"{lo:.1f}-{hi:.1f}", "dem": DEM.name}
        ds.to_netcdf(SUS / f"surface_event{ev}.nc")
        land = ~sea
        print(f"event {ev}: saved surface_event{ev}.nc | land mean S {float(S.where(land).mean()):.2f}, "
              f"G {float(G.mean()):.2f}, B {float(B.mean()):.2f}, E {float(E.mean()):.2f}; "
              f"E > 0.5 in {float((E > 0.5).sum() / land.sum()):.1%} of land")
        st = {k: round(float(E.sel(latitude=la, longitude=lo_, method='nearest')), 2)
              for k, (la, lo_) in STATIONS.items()}
        print(f"  E at stations: {st}")
        if ev == "A":
            panels = {"S (topography)": S, "G (sand + silt)": G}
        panels[f"B (bare), event {ev}"] = B
        panels[f"E = S×G×B, event {ev}"] = E

    fig, axs = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    order = ["S (topography)", "G (sand + silt)", "B (bare), event A",
             "E = S×G×B, event A", "B (bare), event B", "E = S×G×B, event B"]
    for ax, key in zip(axs.flat, order):
        m = ax.pcolormesh(LON, LAT, panels[key], cmap="YlOrBr", vmin=0, vmax=1, shading="auto")
        ax.contour(LON, LAT, (~sea).astype(int), levels=[0.5], colors="k", linewidths=0.5)
        for code, (la, lo_) in STATIONS.items():
            ax.plot(lo_, la, "k^", ms=4)
        ax.set_title(key, fontsize=10)
        ax.set_aspect("equal")
    fig.colorbar(m, ax=axs, shrink=0.7, label="0–1")
    fig.savefig(C.OUT / "fig_surface.png", dpi=180)
    print("saved results/fig_surface.png")


if __name__ == "__main__":
    main()
