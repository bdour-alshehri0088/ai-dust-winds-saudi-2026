"""Identify the source of the local 90 m DEM.

Downloads one Copernicus GLO-90 tile (30-31N, 40-41E, northern Saudi Arabia) from the public
AWS bucket and compares it with the same area of the local DEM tile.

Usage:  python dem_check.py
"""
import urllib.request

import numpy as np
import rasterio
from rasterio.warp import Resampling, reproject

LOCAL = "/Users/budouralshehri/Downloads/DEM files 90m/dem_30_35.0_40_45.0.tif"
URL = ("https://copernicus-dem-90m.s3.amazonaws.com/Copernicus_DSM_COG_30_N30_00_E040_00_DEM/"
       "Copernicus_DSM_COG_30_N30_00_E040_00_DEM.tif")
TILE = "glo90_N30E040.tif"

print("downloading one GLO-90 tile ...")
urllib.request.urlretrieve(URL, TILE)

with rasterio.open(TILE) as g, rasterio.open(LOCAL) as loc_src:
    ref = g.read(1).astype("float32")
    loc = np.zeros_like(ref)
    reproject(rasterio.band(loc_src, 1), loc, dst_transform=g.transform, dst_crs=g.crs,
              resampling=Resampling.nearest)

ok = (loc != 0) & np.isfinite(ref)
d = loc[ok] - ref[ok]
r = np.corrcoef(loc[ok], ref[ok])[0, 1]
print(f"pixels compared: {ok.sum():,}")
print(f"mean difference (local - GLO-90): {d.mean():.2f} m")
print(f"mean absolute difference:         {np.abs(d).mean():.2f} m")
print(f"share identical within 1 m:       {(np.abs(d) <= 1).mean():.1%}")
print(f"correlation:                      {r:.5f}")
if np.abs(d).mean() < 2 and (np.abs(d) <= 1).mean() > 0.8:
    print("VERDICT: the local DEM matches Copernicus GLO-90")
else:
    print("VERDICT: not GLO-90 (most likely SRTM 3 arc-second); send me this output")
