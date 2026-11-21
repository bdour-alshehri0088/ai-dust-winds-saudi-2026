"""ERA5 diagnostics (Sections 3 and 5.6).

  1. Activity ratio: spatial standard deviation of forecast 10 m wind speed over land in the
     domain, divided by that of ERA5 at the same time (< 1 = smoother than ERA5).
  2. Small-scale power ratio: zonal kinetic-energy spectra of 10 m wind along each latitude row;
     forecast power at wavelengths below 400 km divided by ERA5 power at the same wavelengths.
  3. Figure 3: ERA5 MSLP and 10 m wind at the strongest storm-day time of each event.

Inputs : data/era5/era5_sfc_event{A,B}.nc (08_get_era5.py), forecasts, surface land mask
Outputs: results/smoothing_by_lead.csv, fig_activity.png, fig_spectra.png, fig_synoptic.png
Usage  : python 09_era5_diagnostics.py
"""
import glob
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shapefile
import xarray as xr

import config as C

ERA = C.DATA / "era5"
SMALL_KM = 400.0
COLORS = {"IFS": "k", "AIFS": "#0F4D92", "GraphCast": "#42949E", "Aurora": "#9A4D8E", "Pangu": "#B64342"}
COMMON_MISSING = {("B", "2026043000"), ("B", "2026050200")}   # not available for NOAA models
plt.rcParams.update({"font.family": ["Arial", "Helvetica", "DejaVu Sans"], "font.size": 12,
                     "axes.spines.right": False, "axes.spines.top": False, "legend.frameon": False,
                     "pdf.fonttype": 42})


def load_era5(ev):
    f = ERA / f"era5_sfc_event{ev}.nc"
    if not f.exists():
        return None
    ds = xr.open_dataset(f).sortby("latitude")
    if "valid_time" not in ds.dims and "time" in ds.dims:
        ds = ds.rename({"time": "valid_time"})
    ds["msl"] = ds["msl"] / 100.0
    return ds


def land_mask(ev):
    s = xr.open_dataset(C.DATA / "susceptibility" / f"surface_event{ev}.nc")["S"]
    return s.notnull()


def ke_spectrum(u, v, dx_km):
    """Mean zonal KE spectrum over rows (latitude) of 2-D u, v arrays (lat, lon)."""
    n = u.shape[1]
    w = np.hanning(n)
    spec = 0
    for comp in (u, v):
        a = comp - comp.mean(axis=1, keepdims=True)
        a = np.nan_to_num(a) * w
        spec = spec + np.abs(np.fft.rfft(a, axis=1)) ** 2
    k = np.fft.rfftfreq(n, d=dx_km)
    return k[1:], spec[:, 1:].mean(axis=0)


def borders(ax):
    sf = shapefile.Reader(str(C.DATA / "basemap" / "naturalearth_lowres.shp"))
    for shp in sf.shapes():
        pts = np.array(shp.points)
        parts = list(shp.parts) + [len(pts)]
        for a, b in zip(parts[:-1], parts[1:]):
            ax.plot(pts[a:b, 0], pts[a:b, 1], color="#4D4D4D", lw=0.6)


def main():
    rows, spectra = [], []
    for f in sorted(glob.glob(str(C.FC / "*.nc"))):
        model, ev, init = re.match(r".*/(.+)_([AB])_(\d{10})\.nc", f).groups()
        if (ev, init) in COMMON_MISSING:
            continue
        era = load_era5(ev)
        if era is None:
            continue
        land = land_mask(ev)
        fc = xr.open_dataset(f)
        t0 = pd.to_datetime(init, format="%Y%m%d%H")
        lat_mid = float(fc.latitude.mean())
        dx = 0.25 * 111.2 * np.cos(np.radians(lat_mid))
        for t in fc.valid_time.values:
            lead_h = int((pd.Timestamp(t) - t0).total_seconds() // 3600)
            if lead_h <= 0 or t not in era.valid_time.values:
                continue
            e = era.sel(valid_time=t).sel(latitude=fc.latitude, longitude=fc.longitude)
            p = fc.sel(valid_time=t)
            sf_, se = np.hypot(p.u10, p.v10), np.hypot(e.u10, e.v10)
            ratio = float(sf_.where(land).std() / se.where(land).std())
            k, Pf = ke_spectrum(p.u10.values, p.v10.values, dx)
            _, Pe = ke_spectrum(e.u10.values, e.v10.values, dx)
            small = (1 / k) < SMALL_KM
            rows.append({"model": model, "event": ev, "init": init, "lead_day": int(np.ceil(lead_h / 24)),
                         "activity_ratio": ratio, "small_scale_power_ratio": Pf[small].sum() / Pe[small].sum()})
            spectra.append({"model": model, "lead_day": int(np.ceil(lead_h / 24)), "k": k, "P": Pf, "Pera": Pe})
    if not rows:
        raise SystemExit("No ERA5 files yet: run 08_get_era5.py")
    df = pd.DataFrame(rows)
    df[df.lead_day <= 6].groupby(["event", "model", "lead_day"])[["activity_ratio", "small_scale_power_ratio"]].median().reset_index().to_csv(C.OUT / "smoothing_by_event_lead.csv", index=False)
    tab = df[df.lead_day <= 6].groupby(["model", "lead_day"])[["activity_ratio", "small_scale_power_ratio"]] \
        .median().reset_index()
    tab.to_csv(C.OUT / "smoothing_by_lead.csv", index=False)
    print("events with ERA5:", sorted(df.event.unique()))
    print(tab.pivot(index="lead_day", columns="model", values="activity_ratio").round(2))
    print(tab.pivot(index="lead_day", columns="model", values="small_scale_power_ratio").round(2))

    # activity and small-scale power against lead
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.4))
    for ax, col, lab in zip(axs, ["activity_ratio", "small_scale_power_ratio"],
                            ["Activity ratio (forecast / ERA5)", f"Power below {SMALL_KM:.0f} km (forecast / ERA5)"]):
        for m, x in tab.groupby("model"):
            ax.plot(x.lead_day, x[col], marker="o", lw=2.2, color=COLORS.get(m), ls="--" if m == "IFS" else "-", label=m)
        ax.axhline(1, color="grey", lw=0.8)
        ax.set_xlabel("Forecast day")
        ax.set_ylabel(lab)
    axs[0].legend(fontsize=10)
    axs[0].set_title("(a) Spatial variability of 10 m wind", loc="left", fontsize=12)
    axs[1].set_title("(b) Small-scale wind energy", loc="left", fontsize=12)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(C.OUT / f"fig_activity.{ext}", dpi=300, facecolor="white")
    plt.close(fig)

    # spectra at day 1 and day 5
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.4), sharey=True)
    for ax, day in zip(axs, (1, 5)):
        sel = [s for s in spectra if s["lead_day"] == day]
        k = sel[0]["k"]
        ax.loglog(1 / k, np.mean([s["Pera"] for s in sel if s["model"] == "IFS"], axis=0), color="grey", lw=3,
                  label="ERA5")
        for m in COLORS:
            ps = [s["P"] for s in sel if s["model"] == m]
            if ps:
                ax.loglog(1 / k, np.mean(ps, axis=0), color=COLORS[m], lw=2, ls="--" if m == "IFS" else "-", label=m)
        ax.invert_xaxis()
        ax.axvline(SMALL_KM, color="grey", lw=0.8, ls=":")
        ax.set_xlabel("Wavelength (km)")
        ax.set_title(f"({'ab'[day == 5]}) Forecast day {day}", loc="left", fontsize=12)
    axs[0].set_ylabel("10 m kinetic energy (arbitrary units)")
    axs[0].legend(fontsize=10)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(C.OUT / f"fig_spectra.{ext}", dpi=300, facecolor="white")
    plt.close(fig)

    # Figure 3: synoptic situation
    evs = [ev for ev in C.EVENTS if load_era5(ev) is not None]
    fig, axs = plt.subplots(1, len(evs), figsize=(7.2 * len(evs), 5.6), squeeze=False)
    for ax, ev in zip(axs[0], evs):
        era = load_era5(ev)
        day = pd.Timestamp(C.EVENTS[ev]["date"])
        cand = era.sel(valid_time=slice(day, day + pd.Timedelta(hours=18)))
        dom = cand.sel(latitude=slice(C.LAT_MIN, C.LAT_MAX), longitude=slice(C.LON_MIN, C.LON_MAX))
        t = dom.valid_time.values[int(np.hypot(dom.u10, dom.v10).mean(("latitude", "longitude")).argmax())]
        e = era.sel(valid_time=t)
        spd = np.hypot(e.u10, e.v10)
        m = ax.pcolormesh(e.longitude, e.latitude, spd, cmap="YlOrBr", vmin=0, vmax=16, shading="nearest")
        cs = ax.contour(e.longitude, e.latitude, e.msl, levels=np.arange(990, 1040, 2), colors="k", linewidths=0.7)
        ax.clabel(cs, fmt="%d", fontsize=8)
        q = 6
        ax.quiver(e.longitude[::q], e.latitude[::q], e.u10[::q, ::q], e.v10[::q, ::q], scale=250, width=0.0025)
        borders(ax)
        ax.plot([C.LON_MIN, C.LON_MAX, C.LON_MAX, C.LON_MIN, C.LON_MIN],
                [C.LAT_MIN, C.LAT_MIN, C.LAT_MAX, C.LAT_MAX, C.LAT_MIN], color="#0F4D92", lw=1.5, ls="--")
        ax.set_xlim(25, 60)
        ax.set_ylim(15, 40)
        ax.set_aspect("equal")
        ax.set_title(f"({'ab'[evs.index(ev)]}) Event {ev}: {pd.Timestamp(t):%d %b %Y %H} UTC", loc="left", fontsize=12)
        fig.colorbar(m, ax=ax, fraction=0.035, pad=0.02, label="10 m wind speed (m s⁻¹)")
        # numbers for the text
        lo = e.msl.where(e.msl == e.msl.min(), drop=True)
        hi = e.msl.where(e.msl == e.msl.max(), drop=True)
        print(f"event {ev} at {pd.Timestamp(t):%Y-%m-%d %H}Z: MSLP min {float(e.msl.min()):.0f} hPa at "
              f"{float(lo.latitude[0]):.1f}N {float(lo.longitude[0]):.1f}E; max {float(e.msl.max()):.0f} hPa at "
              f"{float(hi.latitude[0]):.1f}N {float(hi.longitude[0]):.1f}E; domain max wind {float(spd.sel(latitude=slice(C.LAT_MIN, C.LAT_MAX), longitude=slice(C.LON_MIN, C.LON_MAX)).max()):.1f} m/s")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(C.OUT / f"fig_synoptic.{ext}", dpi=300, facecolor="white")
    from springer_style import to_springer      # journal artwork: results/springer/Fig3.tif
    to_springer(fig, 3)
    plt.close(fig)
    print("saved fig_activity, fig_spectra, fig_synoptic")


if __name__ == "__main__":
    main()
