"""Verify all models against station observations and write tables and figures.

Outputs in results/:
  thresholds.csv        station dust-emission thresholds (Kurosaki and Mikami, 2007) and
                        climatological strong-wind thresholds (90th/95th percentile, 2016-2025)
  pairs.csv             every forecast-observation pair
  scores_by_lead.csv    continuous and categorical scores by model and forecast day
  peak_wind.csv         peak-wind ratio and timing error per model/event/init/station
  bootstrap_vs_ifs.csv  RMSE difference from IFS with indicative 95% intervals
  fig_*.png             score-versus-lead-time plots
"""
import glob
import warnings
warnings.filterwarnings("ignore", message="This pattern is interpreted as a regular expression")
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

import config as C
from common import dir_error, dup, wind_dir, wind_speed

# Any dust in the air (includes DU/SA = SYNOP 06, suspended or transported dust): hazard occurrence
DUST_RE = re.compile(r"(^|\s|[+-]|VC)(BL|DR)?(DU|SA|DS|SS)\b")
# Dust raised locally (SYNOP 07-09, 30-35; excludes 06): used only for emission thresholds,
# following Kurosaki and Mikami (2007)
EMIT_RE = re.compile(r"BLDU|BLSA|DRDU|DRSA|\bPO\b|(^|\s|[+-]|VC)(DS|SS)\b")
rng = np.random.default_rng(42)


# ---------------------------------------------------------------- observations
def read_obs(path):
    df = pd.read_csv(path)
    df["valid"] = pd.to_datetime(df["valid"])
    df["spd"] = pd.to_numeric(df.get("sknt"), errors="coerce") * C.KT2MS
    df["dir"] = pd.to_numeric(df.get("drct"), errors="coerce")
    df["gust_ms"] = pd.to_numeric(df.get("gust"), errors="coerce") * C.KT2MS
    if "vsby_km" in df:
        df["vis_km"] = pd.to_numeric(df["vsby_km"], errors="coerce")
    else:
        df["vis_km"] = pd.to_numeric(df.get("vsby"), errors="coerce") * 1.609
    wx = df.get("wxcodes", pd.Series("", index=df.index)).fillna("").astype(str)
    df["dust_code"] = wx.str.contains(DUST_RE)
    df["emit_code"] = wx.str.contains(EMIT_RE)
    df["dust_storm"] = df["dust_code"] & (df["vis_km"] < 1.0)
    df["msl_obs"] = pd.to_numeric(df.get("mslp"), errors="coerce")
    # Saudi METARs carry QNH, not SLP: fall back to QNH converted from inHg to hPa
    if df["msl_obs"].isna().all() and "alti" in df:
        df["msl_obs"] = pd.to_numeric(df["alti"], errors="coerce") * 33.8639
    df.loc[(df["msl_obs"] < 950) | (df["msl_obs"] > 1060), "msl_obs"] = np.nan
    return df


def load_event_obs(ev, st):
    ncm = C.OBS / f"ncm_{st}.csv"
    path = ncm if ncm.exists() else C.OBS / f"event{ev}_{st}.csv"
    if not path.exists():
        return None
    return read_obs(path)


def thresholds():
    rows = []
    for path in sorted(C.OBS.glob("clim_*.csv")):
        st = path.stem.split("_")[1]
        df = read_obs(path).dropna(subset=["spd"])
        df["bin"] = np.floor(df["spd"]).astype(int)
        g = df.groupby("bin")["emit_code"].agg(["mean", "size"])
        g = g[g["size"] >= 20]  # minimum reports per 1 m/s bin
        row = {"station": st, "n_obs": len(df), "emit_frac": round(df["emit_code"].mean(), 4)}
        for p in (25, 50, 75):
            ok = g[g["mean"] >= p / 100]
            row[f"ut{p}"] = float(ok.index.min()) + 0.5 if len(ok) else np.nan
        row["ut"] = row["ut50"]
        row["p90"] = round(df["spd"].quantile(0.90), 2)  # strong-wind threshold for the categorical tests
        row["p95"] = round(df["spd"].quantile(0.95), 2)
        rows.append(row)
    t = pd.DataFrame(rows)
    t.to_csv(C.OUT / "thresholds.csv", index=False)
    return t.set_index("station") if len(t) else pd.DataFrame(columns=["ut", "p90"])


# ---------------------------------------------------------------- pairing
def build_pairs(thr):
    rows = []
    for f in sorted(glob.glob(str(C.FC / "*.nc"))):
        model, ev, init = re.match(r".*/(.+)_([AB])_(\d{10})\.nc", f).groups()
        init = pd.to_datetime(init, format="%Y%m%d%H")
        ds = xr.open_dataset(f)
        for st in C.EVENTS[ev]["stations"]:
            obs = load_event_obs(ev, st)
            if obs is None or obs[["lat", "lon"]].isna().all().any():
                continue
            lat, lon = obs["lat"].dropna().iloc[0], obs["lon"].dropna().iloc[0]
            p = ds.interp(latitude=lat, longitude=lon)
            ut = thr["ut"].get(st, np.nan)
            ut = C.DEFAULT_THRESHOLD if np.isnan(ut) else ut
            p90 = thr["p90"].get(st, np.nan)
            for t in p.valid_time.values:
                t = pd.Timestamp(t)
                u, v = float(p.u10.sel(valid_time=t)), float(p.v10.sel(valid_time=t))
                near = obs.iloc[(obs["valid"] - t).abs().argsort()[:1]]
                dt = abs((near["valid"].iloc[0] - t).total_seconds()) / 60
                win = obs[(obs["valid"] - t).abs() <= pd.Timedelta(hours=C.DUST_WINDOW_H)]
                rows.append({
                    "model": model, "event": ev, "init": init, "valid": t, "station": st,
                    "lead_h": int((t - init).total_seconds() // 3600), "ut": ut, "p90": p90,
                    "period": "event" if t.date() == pd.Timestamp(C.EVENTS[ev]["date"]).date()
                              else ("pre" if t < pd.Timestamp(C.EVENTS[ev]["date"]) else "post"),
                    "f_spd": wind_speed(u, v), "f_dir": wind_dir(u, v), "f_u": u, "f_v": v,
                    "f_msl": float(p.msl.sel(valid_time=t)) if "msl" in p else np.nan,
                    "f_gust": float(p.gust.sel(valid_time=t)) if "gust" in p else np.nan,
                    "o_spd": near["spd"].iloc[0] if dt <= C.MATCH_TOL_MIN else np.nan,
                    "o_dir": near["dir"].iloc[0] if dt <= C.MATCH_TOL_MIN else np.nan,
                    "o_msl": near["msl_obs"].iloc[0] if dt <= C.MATCH_TOL_MIN else np.nan,
                    "o_dust_storm": bool(win["dust_storm"].any()),
                    "o_spd_max_day": obs.loc[obs["valid"].dt.date == t.date(), "spd"].max(),
                })
    pairs = pd.DataFrame(rows)
    if pairs.empty:
        raise SystemExit("No pairs: run the download scripts first.")
    rad = np.radians(pairs["o_dir"])
    pairs["o_u"] = -pairs["o_spd"] * np.sin(rad)
    pairs["o_v"] = -pairs["o_spd"] * np.cos(rad)
    pairs["lead_day"] = np.ceil(pairs["lead_h"] / 24).astype(int)
    pairs.to_csv(C.OUT / "pairs.csv", index=False)
    return pairs


# ---------------------------------------------------------------- scores
def contingency(fc_yes, ob_yes, pre):
    fc_yes, ob_yes = fc_yes.astype(bool), ob_yes.astype(bool)
    a, b = int((fc_yes & ob_yes).sum()), int((fc_yes & ~ob_yes).sum())
    c, dd = int((~fc_yes & ob_yes).sum()), int((~fc_yes & ~ob_yes).sum())
    n = a + b + c + dd
    ar = (a + b) * (a + c) / n if n else np.nan
    return {pre + "hits": a, pre + "false_alarms": b, pre + "misses": c, pre + "correct_neg": dd,
            pre + "pod": a / (a + c) if a + c else np.nan,
            pre + "far": b / (a + b) if a + b else np.nan,
            pre + "freq_bias": (a + b) / (a + c) if a + c else np.nan,
            pre + "csi": a / (a + b + c) if a + b + c else np.nan,
            pre + "ets": (a - ar) / (a + b + c - ar) if (a + b + c - ar) else np.nan}


def scores(g):
    ok = g.dropna(subset=["o_spd"])
    e = ok["f_spd"] - ok["o_spd"]
    vec = np.sqrt(((ok["f_u"] - ok["o_u"])**2 + (ok["f_v"] - ok["o_v"])**2).mean())
    dd = ok[ok["o_spd"] >= C.DIR_MIN_SPEED]
    derr = np.abs(dir_error(dd["f_dir"], dd["o_dir"]))
    m = g.dropna(subset=["o_msl", "f_msl"])
    # Test 1: wind skill only (forecast vs observed exceedance of the station 90th percentile)
    w = contingency(ok["f_spd"] > ok["p90"], ok["o_spd"] > ok["p90"], "w_")
    # Test 2: wind-to-dust link (forecast strong wind vs observed dust storm within +/-3 h)
    d = contingency(g["f_spd"] > g["p90"], g["o_dust_storm"].astype(bool), "d_")
    f_dup, o_dup = dup(ok["f_spd"], ok["ut"]).sum(), dup(ok["o_spd"], ok["ut"]).sum()
    return pd.Series({**w, **d,
        "n": len(ok), "bias": e.mean(), "mae": e.abs().mean(), "rmse": np.sqrt((e**2).mean()),
        "vec_rmse": vec, "dir_mae": derr.mean(), "n_dir": len(dd), "n_msl": len(m),
        "msl_bias": (m["f_msl"] - m["o_msl"]).mean(),
        "msl_rmse": np.sqrt(((m["f_msl"] - m["o_msl"])**2).mean()),
        "dup_ratio": f_dup / o_dup if o_dup else np.nan,
    })


def peak_wind(pairs):
    g = pairs[pairs["period"] == "event"].groupby(["model", "event", "init", "station"])
    rows = []
    for (m, ev, init, st), x in g:
        x = x.dropna(subset=["o_spd"])
        if x.empty:
            continue
        fi, oi = x["f_spd"].idxmax(), x["o_spd"].idxmax()
        rows.append({"model": m, "event": ev, "init": init, "station": st,
                     "lead_day": (pd.Timestamp(C.EVENTS[ev]["date"]) - init).days,  # days before event
                     "pwr_matched": x["f_spd"].max() / x["o_spd"].max(),
                     "pwr_daily_max": x["f_spd"].max() / x["o_spd_max_day"].max(),
                     "peak_timing_err_h": (x.loc[fi, "valid"] - x.loc[oi, "valid"]).total_seconds() / 3600})
    pw = pd.DataFrame(rows)
    pw.to_csv(C.OUT / "peak_wind.csv", index=False)
    return pw


def bootstrap_vs_ifs(pairs, n_boot=1000):
    rows = []
    ok = pairs.dropna(subset=["o_spd"]).copy()
    ok["se"] = (ok["f_spd"] - ok["o_spd"])**2
    for (m, day), x in ok.groupby(["model", "lead_day"]):
        if m == "IFS":
            continue
        ref = ok[(ok["model"] == "IFS") & (ok["lead_day"] == day)]
        key = ["event", "init", "station", "valid"]
        j = x.merge(ref[key + ["se"]], on=key, suffixes=("", "_ifs"))
        if j.empty:
            continue
        j["vday"] = pd.to_datetime(j["valid"]).dt.date
        groups = [grp for _, grp in j.groupby(["event", "vday"])]  # blocks: event x verification day
        diffs = []
        for _ in range(n_boot):
            s = pd.concat([groups[i] for i in rng.integers(0, len(groups), len(groups))])
            diffs.append(np.sqrt(s["se"].mean()) - np.sqrt(s["se_ifs"].mean()))
        rows.append({"model": m, "lead_day": day, "n": len(j),
                     "rmse_diff": np.sqrt(j["se"].mean()) - np.sqrt(j["se_ifs"].mean()),
                     "ci_low": np.percentile(diffs, 2.5), "ci_high": np.percentile(diffs, 97.5)})
    b = pd.DataFrame(rows)
    b.to_csv(C.OUT / "bootstrap_vs_ifs.csv", index=False)
    return b


def plot(tab, col, ylabel, fname, ref=None, xlabel="Forecast day"):
    fig, ax = plt.subplots(figsize=(6, 4))
    for m, x in tab.groupby("model"):
        x = x.sort_values("lead_day")
        ax.plot(x["lead_day"], x[col], marker="o", lw=2 if m != "IFS" else 2.5,
                ls="--" if m == "IFS" else "-", color="k" if m == "IFS" else None, label=m)
    if ref is not None:
        ax.axhline(ref, color="grey", lw=0.8)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(C.OUT / fname, dpi=200)
    plt.close(fig)


def main():
    thr = thresholds()
    print(thr[["ut25", "ut", "p90", "p95"]].to_string() if len(thr) else "thresholds: none (using default)")
    pairs_all = build_pairs(thr)
    pairs_all = pairs_all[pairs_all["lead_h"] > 0]  # lead 0 is the analysis, not a forecast
    # Fair comparison: keep only (event, init) cases available for every model
    n_models = pairs_all["model"].nunique()
    have = pairs_all.groupby(["event", "init"])["model"].nunique()
    common = have[have == n_models].index
    missing = have[have < n_models]
    if len(missing):
        print("Cases dropped (not available for all models):")
        print(missing.to_string())
    pairs = pairs_all.set_index(["event", "init"]).loc[common].reset_index()
    # Supplementary: AIFS vs IFS on every available initialisation
    ecm = pairs_all[pairs_all["model"].isin(["AIFS", "IFS"])]
    if not ecm.empty:
        ecm.groupby(["model", "lead_day"]).apply(scores, include_groups=False).reset_index() \
           .to_csv(C.OUT / "scores_by_lead_AIFS_IFS_all_inits.csv", index=False)
    tab = pairs.groupby(["model", "lead_day"]).apply(scores, include_groups=False).reset_index()
    tab_ev = pairs.groupby(["event", "model", "lead_day"]).apply(scores, include_groups=False).reset_index()
    tab.to_csv(C.OUT / "scores_by_lead.csv", index=False)
    tab_p = pairs.groupby(["period", "model", "lead_day"]).apply(scores, include_groups=False).reset_index()
    tab_p.to_csv(C.OUT / "scores_by_period_lead.csv", index=False)
    tab_ev.to_csv(C.OUT / "scores_by_event_lead.csv", index=False)
    pw = peak_wind(pairs)
    bootstrap_vs_ifs(pairs)
    plot(tab, "rmse", "10 m wind RMSE (m/s)", "fig_rmse.png")
    plot(tab, "bias", "10 m wind bias (m/s)", "fig_bias.png", ref=0)
    plot(tab, "vec_rmse", "Vector wind RMSE (m/s)", "fig_vec_rmse.png")
    plot(tab, "dir_mae", "Direction MAE (deg)", "fig_dir.png")
    plot(tab, "w_csi", "CSI: wind > station P90", "fig_csi_wind.png")
    plot(tab, "d_csi", "CSI: forecast wind > P90 vs dust storm", "fig_csi_dust.png")
    plot(tab, "dup_ratio", "DUP ratio (forecast / observed)", "fig_dup.png", ref=1)
    if not pw.empty:
        plot(pw.groupby(["model", "lead_day"])["pwr_matched"].median().reset_index(),
             "pwr_matched", "Peak-wind ratio (median)", "fig_pwr.png", ref=1,
             xlabel="Days before event (initialisation)")
    print(tab.round(2).to_string())


if __name__ == "__main__":
    main()
