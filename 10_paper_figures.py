"""Publication figures for the paper (figures4papers style; credit: github.com/ChenLiu-1996/figures4papers).

One colour per system in every figure: IFS black dashed (physics reference), AIFS blue,
GraphCast teal, Aurora violet, Pangu red. Panels are lettered; units are in every axis label.
Outputs (PNG 300 dpi + PDF) in results/figures/:
  fig01_study_area   fig04_wind_scores   fig05_direction   fig06_storm_timeseries
  fig07_threshold_skill   fig08_dup_eventB   fig09_smoothing
Usage:  python 10_paper_figures.py
"""
import importlib.util
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd
import shapefile
import xarray as xr
from matplotlib.lines import Line2D

import config as C

FIG = C.OUT / "figures"
FIG.mkdir(parents=True, exist_ok=True)
MODELS = ["IFS", "AIFS", "GraphCast", "Aurora", "Pangu"]
COLORS = {"IFS": "#272727", "AIFS": "#0F4D92", "GraphCast": "#42949E", "Aurora": "#9A4D8E", "Pangu": "#B64342"}
LS = {m: ("--" if m == "IFS" else "-") for m in MODELS}
STATIONS = {"OETR": ("Turaif", 31.69, 38.73), "OERR": ("Arar", 30.91, 41.14), "OESK": ("Al-Jawf", 29.79, 40.10),
            "OERF": ("Rafha", 29.63, 43.49), "OETB": ("Tabuk", 28.37, 36.62), "OEPA": ("Al-Qaisumah", 28.34, 46.13),
            "OEHL": ("Ha'il", 27.44, 41.69), "OEGS": ("Al-Qassim", 26.30, 43.77), "OERK": ("Riyadh", 24.96, 46.70)}
plt.rcParams.update({"font.family": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"], "font.size": 13,
                     "axes.spines.right": False, "axes.spines.top": False, "axes.linewidth": 1.6,
                     "xtick.major.width": 1.4, "ytick.major.width": 1.4, "legend.frameon": False,
                     "svg.fonttype": "none", "pdf.fonttype": 42})


def module(name):
    spec = importlib.util.spec_from_file_location(name.replace(".py", "").lstrip("0123456789_"), name)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def save(fig, name):
    fig.tight_layout(pad=1.2)
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"{name}.{ext}", dpi=300, facecolor="white")
    plt.close(fig)
    print("saved", name)


def borders(ax, lw=0.7):
    sf = shapefile.Reader(str(C.DATA / "basemap" / "naturalearth_lowres.shp"))
    for shp in sf.shapes():
        pts = np.array(shp.points)
        parts = list(shp.parts) + [len(pts)]
        for a, b in zip(parts[:-1], parts[1:]):
            ax.plot(pts[a:b, 0], pts[a:b, 1], color="#4D4D4D", lw=lw)


def model_legend(ax, **kw):
    h = [Line2D([], [], color=COLORS[m], ls=LS[m], lw=2.6, marker="o", ms=5, label=m) for m in MODELS]
    ax.legend(handles=h, **kw)


def lead_lines(ax, tab, col, ylabel, ref=None):
    for m in MODELS:
        x = tab[tab.model == m].sort_values("lead_day")
        ax.plot(x.lead_day, x[col], color=COLORS[m], ls=LS[m], lw=2.6, marker="o", ms=5)
    if ref is not None:
        ax.axhline(ref, color="#767676", lw=1)
    ax.set_xticks(range(1, 7))
    ax.set_xlabel("Forecast day")
    ax.set_ylabel(ylabel)


# ------------------------------------------------------------------ figures
def fig01():
    sus = module("05_susceptibility.py")
    z = sus.dem_cells(sus.DEM).sortby("latitude").sel(latitude=slice(15, 38), longitude=slice(30, 56))
    fig, ax = plt.subplots(figsize=(9.5, 7.2))
    land_cmap = ListedColormap(plt.get_cmap("terrain")(np.linspace(0.25, 1.0, 256)))   # no blue: sea is white
    m = ax.pcolormesh(z.longitude, z.latitude, z.clip(min=0), cmap=land_cmap, vmin=0, vmax=2400, shading="nearest")
    borders(ax)
    ax.plot([C.LON_MIN, C.LON_MAX, C.LON_MAX, C.LON_MIN, C.LON_MIN],
            [C.LAT_MIN, C.LAT_MIN, C.LAT_MAX, C.LAT_MAX, C.LAT_MIN], color="#0F4D92", lw=2, ls="--")
    evA = set(C.EVENTS["A"]["stations"]); evB = set(C.EVENTS["B"]["stations"])
    for code, (name, la, lo) in STATIONS.items():
        mk = "o" if (code in evA and code in evB) else ("s" if code in evA else "D")
        ax.plot(lo, la, mk, ms=9, mfc="white", mec="k", mew=1.6, zorder=5)
        ax.annotate(f"{name}\n({code})", (lo, la), xytext=(7, 2), textcoords="offset points", fontsize=9.5,
                    zorder=6, bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.7))
    for txt, (lo, la) in {"Iraq": (44.2, 33.2), "Jordan": (36.6, 30.9), "Syria": (38.5, 35.2),
"Saudi Arabia": (44.5, 23.3), "Red Sea": (37.6, 21.0),
                          "Arabian Gulf": (51.2, 27.3)}.items():
        ax.text(lo, la, txt, fontsize=10, style="italic", color="#272727", ha="center")
    ax.set_xlim(30, 56); ax.set_ylim(15, 38); ax.set_aspect("equal")
    ax.set_xlabel("Longitude (°E)"); ax.set_ylabel("Latitude (°N)")
    for s in ("top", "right"):
        ax.spines[s].set_visible(True)
    fig.colorbar(m, ax=ax, fraction=0.035, pad=0.02, label="Elevation (m), SRTM averaged to 0.25°")
    ax.legend(handles=[Line2D([], [], marker="o", ls="", ms=9, mfc="white", mec="k", label="Both storms"),
                       Line2D([], [], marker="s", ls="", ms=9, mfc="white", mec="k", label="January storm only"),
                       Line2D([], [], marker="D", ls="", ms=8, mfc="white", mec="k", label="May storm only"),
                       Line2D([], [], color="#0F4D92", ls="--", lw=2, label="Verification domain")],
              loc="lower left", fontsize=10, frameon=True, facecolor="white", edgecolor="none")
    save(fig, "fig01_study_area")


def fig04(tab, boot):
    fig, axs = plt.subplots(1, 3, figsize=(17, 4.8))
    lead_lines(axs[0], tab, "rmse", "Wind-speed RMSE (m s$^{-1}$)")
    lead_lines(axs[1], tab, "bias", "Wind-speed bias (m s$^{-1}$)", ref=0)
    lead_lines(axs[2], tab, "vec_rmse", "Vector-wind RMSE (m s$^{-1}$)")
    for ax, t in zip(axs, ["(a) Speed error", "(b) Speed bias (forecast − observed)", "(c) Speed and direction error"]):
        ax.set_title(t, loc="left", fontsize=13)
    a = boot[(boot.model == "AIFS") & (boot.lead_day <= 6)]
    sig = (a.ci_high < 0).sum()
    axs[0].text(0.02, 0.97, f"AIFS below IFS on all 6 days ({a.rmse_diff.max():+.2f} to {a.rmse_diff.min():+.2f} m s$^{{-1}}$);\n"
                f"significant on {sig} of 6 days (95% block bootstrap)",
                transform=axs[0].transAxes, va="top", fontsize=10, color=COLORS["AIFS"])
    model_legend(axs[2], loc="upper left", fontsize=10.5, ncol=2)
    save(fig, "fig04_wind_scores")


def fig05(tab):
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    lead_lines(ax, tab, "dir_mae", "Direction MAE (°), winds ≥ 3 m s$^{-1}$")
    ax.set_title("Wind-direction error", loc="left", fontsize=13)
    model_legend(ax, loc="upper left", fontsize=10.5, ncol=2)
    save(fig, "fig05_direction")


def fig06(pairs):
    ver = module("04_verify.py")
    panels = [("B", "OERR", 2), ("B", "OETR", 2), ("B", "OESK", 2), ("A", "OERR", 2)]
    fig, axs = plt.subplots(2, 2, figsize=(15, 8.2), sharey=True)
    for ax, (ev, st, dback), lab in zip(axs.flat, panels, "abcd"):
        day = pd.Timestamp(C.EVENTS[ev]["date"])
        obs = ver.load_event_obs(ev, st)
        obs = obs[(obs.valid >= day - pd.Timedelta(days=1)) & (obs.valid <= day + pd.Timedelta(days=2))]
        ax.plot(obs.valid, obs.spd, color="#9E9E9E", lw=1.6, label="Observed (hourly)")
        storm = obs[obs.dust_storm]
        ax.scatter(storm.valid, storm.spd, s=28, color="#FFD700", ec="k", lw=0.6, zorder=6, label="Dust-storm hour")
        init = (day - pd.Timedelta(days=dback)).strftime("%Y-%m-%d")
        for m in MODELS:
            x = pairs[(pairs.model == m) & (pairs.event == ev) & (pairs.station == st) & (pairs.init == init)]
            ax.plot(pd.to_datetime(x.valid), x.f_spd, color=COLORS[m], ls=LS[m], lw=2.2, marker="o", ms=4)
        p90 = float(pairs[pairs.station == st].p90.iloc[0])
        ax.axhline(p90, color="#767676", lw=1, ls=":")
        ax.text(obs.valid.iloc[1], p90 + 0.3, "station P90", fontsize=9, color="#4D4D4D")
        ax.axvspan(day, day + pd.Timedelta(days=1), color="#F6CFCB", alpha=0.35, lw=0)
        name = STATIONS[st][0]
        when = pd.Timestamp(init).strftime("%-d %b")
        ax.set_title(f"({lab}) {name}, {'May' if ev == 'B' else 'January'} storm, forecasts initialised {when} 00 UTC",
                     loc="left", fontsize=12)
        ax.xaxis.set_major_locator(matplotlib.dates.DayLocator())
        ax.xaxis.set_minor_locator(matplotlib.dates.HourLocator(byhour=[12]))
        ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%-d %b"))
        ax.set_ylabel("10 m wind (m s$^{-1}$)")
    h = [Line2D([], [], color="#9E9E9E", lw=1.6, label="Observed (hourly)"),
         Line2D([], [], marker="o", ls="", ms=6, mfc="#FFD700", mec="k", label="Dust-storm hour"),
         matplotlib.patches.Patch(fc="#F6CFCB", alpha=0.6, label="Storm day")] + \
        [Line2D([], [], color=COLORS[m], ls=LS[m], lw=2.2, marker="o", ms=4, label=m) for m in MODELS]
    axs[0, 1].legend(handles=h, loc="upper right", fontsize=9.5, ncol=2)
    save(fig, "fig06_storm_timeseries")


def fig07(tab):
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.8))
    lead_lines(axs[0], tab, "w_csi", "CSI")
    lead_lines(axs[1], tab, "d_csi", "CSI")
    axs[0].set_title("(a) Test 1: forecast vs observed wind > P90", loc="left", fontsize=13, pad=14)
    axs[1].set_title("(b) Test 2: forecast wind > P90 vs dust-storm hour", loc="left", fontsize=13, pad=14)
    axs[0].set_ylim(0, 0.6); axs[1].set_ylim(0, 0.3)
    ifs = tab[tab.model == "IFS"].sort_values("lead_day")
    for ax, pre in zip(axs, ("w_", "d_")):
        for d, nobs in zip(ifs.lead_day, ifs[pre + "hits"] + ifs[pre + "misses"]):
            ax.text(d, ax.get_ylim()[1] * 0.97, f"n={int(nobs)}", ha="center", va="top", fontsize=9.5, color="#4D4D4D")
    model_legend(axs[0], loc="lower left", fontsize=10.5, ncol=3)
    save(fig, "fig07_threshold_skill")


def fig08(pairs):
    p = pairs[(pairs.event == "B") & (pairs.lead_h > 0) & (pairs.lead_day <= 6) &
              ~pairs.init.isin(["2026-04-30", "2026-05-02"])].dropna(subset=["o_spd"])
    from common import dup
    obs = np.nansum(dup(p[p.model == "IFS"].o_spd, p[p.model == "IFS"].ut))
    r = {m: np.nansum(dup(p[p.model == m].f_spd, p[p.model == m].ut)) / obs for m in MODELS}
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    bars = ax.bar(MODELS, [r[m] for m in MODELS], color=[COLORS[m] for m in MODELS], edgecolor="k", lw=1.5)
    ax.bar_label(bars, fmt="%.1f", fontsize=12, padding=3)
    ax.axhline(1, color="#767676", lw=1.2, ls="--")
    ax.text(-0.42, 1.06, "observed = 1", fontsize=10, color="#4D4D4D", ha="left")
    ax.set_ylabel("DUP ratio (forecast / observed)")
    ax.set_title("Dust uplift potential, May storm, days 1–6", loc="left", fontsize=13)
    ax.set_ylim(0, max(r.values()) * 1.2)
    save(fig, "fig08_dup_eventB")
    print("DUP ratios event B:", {k: round(v, 2) for k, v in r.items()})


def fig09():
    t = pd.read_csv(C.OUT / "smoothing_by_lead.csv")
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.8))
    lead_lines(axs[0], t, "activity_ratio", "Activity ratio (forecast / ERA5)", ref=1)
    lead_lines(axs[1], t, "small_scale_power_ratio", "Energy < 400 km (forecast / ERA5)", ref=1)
    axs[1].set_yscale("log"); axs[1].set_yticks([0.4, 0.6, 1, 1.5, 2], ["0.4", "0.6", "1", "1.5", "2"])
    axs[0].set_title("(a) Domain-scale wind variability", loc="left", fontsize=13)
    axs[1].set_title("(b) Small-scale wind energy", loc="left", fontsize=13)
    model_legend(axs[0], loc="lower left", fontsize=10.5, ncol=3)
    save(fig, "fig09_smoothing")


def main():
    tab = pd.read_csv(C.OUT / "scores_by_lead.csv")
    tab = tab[tab.lead_day.between(1, 6)]
    boot = pd.read_csv(C.OUT / "bootstrap_vs_ifs.csv")
    pairs = pd.read_csv(C.OUT / "pairs.csv")
    pairs["init"] = pd.to_datetime(pairs["init"]).dt.strftime("%Y-%m-%d")
    fig01(); fig04(tab, boot); fig05(tab); fig06(pairs); fig07(tab); fig08(pairs); fig09()


if __name__ == "__main__":
    main()
