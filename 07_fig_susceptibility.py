"""Figures for the surface susceptibility index (Section 5.7).

  fig_susceptibility.png/.pdf   (a) map of E with stations, (b) E at each station
  fig_surface_components.png/.pdf  S, G and B, each with its own scale (supplementary)

Country borders: Natural Earth 1:110m (data/basemap). Requires pyshp.
Usage:  python 07_fig_susceptibility.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import shapefile
import xarray as xr
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import config as C

PALETTE = {"red_strong": "#B64342", "neutral": "#CFCECE", "gray_dark": "#4D4D4D",
           "gray_ink": "#272727", "blue_main": "#0F4D92"}
plt.rcParams.update({"font.family": ["Arial", "Helvetica", "DejaVu Sans"], "font.size": 12,
                     "axes.spines.right": False, "axes.spines.top": False, "axes.linewidth": 1.2,
                     "legend.frameon": False, "pdf.fonttype": 42, "svg.fonttype": "none"})

STATIONS = {  # name: (ICAO, lat, lon)
    "Turaif": ("OETR", 31.69, 38.73), "Arar": ("OERR", 30.91, 41.14),
    "Al-Jawf": ("OESK", 29.79, 40.10), "Rafha": ("OERF", 29.63, 43.49),
    "Tabuk": ("OETB", 28.37, 36.62), "Al-Qaisumah": ("OEPA", 28.34, 46.13),
    "Ha'il": ("OEHL", 27.44, 41.69), "Al-Qassim": ("OEGS", 26.30, 43.77),
    "Riyadh": ("OERK", 24.96, 46.70)}
DUST_STORM = {"Turaif", "Arar", "Al-Jawf", "Rafha"}   # dust-storm hours observed (Table 10)
LABEL_OFFSET = {"Rafha": (6, -12), "Al-Jawf": (-8, -14), "Al-Qaisumah": (-30, 8)}
REGIONS = {"Mesopotamian\nlowlands": (45.6, 32.6), "Gulf coast": (49.6, 25.3),
           "Northern\nplateau": (39.6, 30.4), "Red Sea\ncoast": (37.1, 23.6)}


def borders(ax):
    sf = shapefile.Reader(str(C.DATA / "basemap" / "naturalearth_lowres.shp"))
    for shp in sf.shapes():
        pts = np.array(shp.points)
        parts = list(shp.parts) + [len(pts)]
        for a, b in zip(parts[:-1], parts[1:]):
            ax.plot(pts[a:b, 0], pts[a:b, 1], color=PALETTE["gray_dark"], lw=0.6)


def frame(ax):
    ax.set_xlim(C.LON_MIN, C.LON_MAX)
    ax.set_ylim(C.LAT_MIN, C.LAT_MAX)
    ax.set_aspect("equal")
    ax.set_xticks(range(36, 53, 4), [f"{x}°E" for x in range(36, 53, 4)])
    ax.set_yticks(range(20, 35, 4), [f"{y}°N" for y in range(20, 35, 4)])
    for s in ("top", "right"):
        ax.spines[s].set_visible(True)


def main():
    ds = xr.open_dataset(C.DATA / "susceptibility" / "surface_eventB.nc")
    E = ds["E"]

    # ---------------- main figure
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(14, 5.6), gridspec_kw={"width_ratios": [1.45, 1]})
    m = ax.pcolormesh(E.longitude, E.latitude, E, cmap="YlOrBr", vmin=0, vmax=1, shading="nearest")
    borders(ax)
    for name, (code, la, lo) in STATIONS.items():
        storm = name in DUST_STORM
        ax.plot(lo, la, "^", ms=9, mec="k", mew=1.2,
                mfc=PALETTE["red_strong"] if storm else "white", zorder=5)
        ax.annotate(name, (lo, la), xytext=LABEL_OFFSET.get(name, (6, 4)), textcoords="offset points",
                    fontsize=9.5, color=PALETTE["gray_ink"], zorder=6)
    for txt, (lo, la) in REGIONS.items():
        ax.text(lo, la, txt, fontsize=9, style="italic", color=PALETTE["gray_dark"], ha="center")
    frame(ax)
    cb = fig.colorbar(m, ax=ax, fraction=0.035, pad=0.02)
    cb.set_label("Surface susceptibility E (0 = none, 1 = highest)")
    ax.legend(handles=[Line2D([], [], marker="^", ls="", ms=9, mec="k", mfc=PALETTE["red_strong"],
                              label="Dust-storm hours observed"),
                       Line2D([], [], marker="^", ls="", ms=9, mec="k", mfc="white",
                              label="No dust-storm hours")],
              loc="lower left", fontsize=9.5, frameon=True, facecolor="white", edgecolor="none")
    ax.set_title("(a) Where the land surface favours dust emission", loc="left", fontsize=12.5)

    vals = {n: float(E.sel(latitude=la, longitude=lo, method="nearest")) for n, (_, la, lo) in STATIONS.items()}
    order = sorted(vals, key=vals.get)
    y = np.arange(len(order))
    colors = [PALETTE["red_strong"] if n in DUST_STORM else PALETTE["neutral"] for n in order]
    bx.barh(y, [vals[n] for n in order], color=colors, edgecolor="k", lw=1.0, height=0.65)
    for i, n in enumerate(order):
        bx.text(vals[n] + 0.008, i, f"{vals[n]:.2f}", va="center", fontsize=10)
    land_median = float(E.median())
    bx.axvline(land_median, color=PALETTE["blue_main"], ls="--", lw=1.4)
    bx.text(land_median + 0.006, len(order) - 0.45, f"domain median {land_median:.2f}",
            color=PALETTE["blue_main"], fontsize=9.5)
    bx.set_yticks(y, order)
    bx.set_xlim(0, max(0.4, max(vals.values()) * 1.25))
    bx.set_xlabel("E at the station grid point")
    bx.legend(handles=[Patch(fc=PALETTE["red_strong"], ec="k", label="Dust-storm hours observed"),
                       Patch(fc=PALETTE["neutral"], ec="k", label="No dust-storm hours")],
              loc="lower right", fontsize=9.5)
    bx.set_title("(b) Dust-storm stations score no higher than the others", loc="left", fontsize=12.5)
    fig.tight_layout(pad=1.2)
    for ext in ("png", "pdf"):
        fig.savefig(C.OUT / f"fig_susceptibility.{ext}", dpi=300, facecolor="white")
    plt.close(fig)

    # ---------------- supplementary: components
    comps = [("S", "Topographic lows (S)", "Low ground within a 10° region", "YlOrBr"),
             ("G", "Fine loose soil (G)", "Sand + silt share, rescaled", "copper_r"),
             ("B", "Bare ground (B)", "1 − vegetation fraction, spring storm", "Greens_r")]
    fig, axs = plt.subplots(1, 3, figsize=(16, 4.6))
    for a, (var, title, sub, cmap) in zip(axs, comps):
        mm = a.pcolormesh(ds.longitude, ds.latitude, ds[var], cmap=cmap, vmin=0, vmax=1, shading="nearest")
        borders(a)
        for name, (code, la, lo) in STATIONS.items():
            a.plot(lo, la, "^", ms=6, mec="k", mew=0.8,
                   mfc=PALETTE["red_strong"] if name in DUST_STORM else "white")
        frame(a)
        a.set_title(f"{title}\n{sub}", loc="left", fontsize=11)
        fig.colorbar(mm, ax=a, fraction=0.04, pad=0.02)
    fig.tight_layout(pad=1.0)
    for ext in ("png", "pdf"):
        fig.savefig(C.OUT / f"fig_surface_components.{ext}", dpi=300, facecolor="white")
    plt.close(fig)
    print("saved results/fig_susceptibility.png and results/fig_surface_components.png")
    print({n: round(v, 2) for n, v in vals.items()}, "domain median", round(land_median, 2))


if __name__ == "__main__":
    main()
