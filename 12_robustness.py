"""Robustness checks requested before submission.

1. Uncertainty of CSI (Test 1 and Test 2): block bootstrap (event x verification day), CSI per model and
   CSI difference from IFS with 95% intervals, pooled and per storm.
2. Does the RMSE ranking predict the CSI ranking?  Spearman rank correlation across the five systems,
   per storm, with its bootstrap distribution.
3. Threshold sensitivity: Test 1 and Test 2 with the station P90, P95 and fixed 8 and 10 m/s thresholds.

Input: results/pairs.csv, results/thresholds.csv.  Outputs (results/):
  robust_csi_bootstrap.csv  robust_rank_agreement.csv  robust_threshold_sensitivity.csv
Usage:  python 12_robustness.py
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import config as C

MODELS = ["IFS", "AIFS", "GraphCast", "Aurora", "Pangu"]
N_BOOT = 2000
rng = np.random.default_rng(42)


def common_pairs():
    p = pd.read_csv(C.OUT / "pairs.csv", parse_dates=["valid"])
    p = p[(p.lead_h > 0) & p.lead_day.between(1, 6)]
    have = p.groupby(["event", "init"]).model.nunique()
    keep = have[have == p.model.nunique()].index
    p = p.set_index(["event", "init"]).loc[keep].reset_index()
    thr = pd.read_csv(C.OUT / "thresholds.csv").set_index("station")
    p["p95"] = p.station.map(thr["p95"])
    p["vday"] = p.valid.dt.date
    return p


def csi(f_yes, o_yes):
    a = (f_yes & o_yes).sum(); b = (f_yes & ~o_yes).sum(); c = (~f_yes & o_yes).sum()
    return a / (a + b + c) if (a + b + c) else np.nan


def table(p, thr_col=None, fixed=None):
    """CSI and RMSE per model for a set of pairs; threshold from a column or a fixed value (m/s)."""
    out = {}
    for m, x in p.groupby("model"):
        t = x[thr_col] if thr_col else fixed
        ok = x.dropna(subset=["o_spd"])
        tt = ok[thr_col] if thr_col else fixed
        out[m] = {"csi_wind": csi(ok.f_spd > tt, ok.o_spd > tt),
                  "csi_dust": csi(x.f_spd > t, x.o_dust_storm.astype(bool)),
                  "n_obs_wind": int((ok.o_spd > tt).sum()),
                  "rmse": np.sqrt(((ok.f_spd - ok.o_spd) ** 2).mean())}
    return pd.DataFrame(out).T


def bootstrap(p, label):
    """Resample whole (event, verification-day) blocks; the same blocks for every model."""
    blocks = [g for _, g in p.groupby(["event", "vday"])]
    point = table(p, "p90")
    samples = []
    for _ in range(N_BOOT):
        s = pd.concat([blocks[i] for i in rng.integers(0, len(blocks), len(blocks))])
        samples.append(table(s, "p90"))
    rows = []
    for m in MODELS:
        for k in ("csi_wind", "csi_dust"):
            v = np.array([t.loc[m, k] for t in samples])
            d = np.array([t.loc[m, k] - t.loc["IFS", k] for t in samples])
            rows.append({"subset": label, "model": m, "test": "Test 1" if k == "csi_wind" else "Test 2",
                         "csi": point.loc[m, k], "csi_low": np.nanpercentile(v, 2.5),
                         "csi_high": np.nanpercentile(v, 97.5),
                         "diff_vs_ifs": point.loc[m, k] - point.loc["IFS", k],
                         "diff_low": np.nanpercentile(d, 2.5), "diff_high": np.nanpercentile(d, 97.5)})
    rank = []
    for t in samples:
        r = spearmanr(t.loc[MODELS, "rmse"], t.loc[MODELS, "csi_wind"]).statistic
        rank.append(r)
    r0 = spearmanr(point.loc[MODELS, "rmse"], point.loc[MODELS, "csi_wind"]).statistic
    rank_row = {"subset": label, "spearman_rmse_vs_csi": r0,
                "low": np.nanpercentile(rank, 2.5), "high": np.nanpercentile(rank, 97.5),
                "share_negative": np.nanmean(np.array(rank) < 0),
                "rmse_order": " < ".join(point.loc[MODELS].sort_values("rmse").index),
                "csi_order": " > ".join(point.loc[MODELS].sort_values("csi_wind", ascending=False).index)}
    return rows, rank_row


def main():
    p = common_pairs()
    print("pairs per model:", p.groupby("model").size().to_dict())
    rows, ranks = [], []
    for label, sub in (("both", p), ("A", p[p.event == "A"]), ("B", p[p.event == "B"])):
        r, k = bootstrap(sub, label)
        rows += r; ranks.append(k)
    b = pd.DataFrame(rows); b.to_csv(C.OUT / "robust_csi_bootstrap.csv", index=False)
    rk = pd.DataFrame(ranks); rk.to_csv(C.OUT / "robust_rank_agreement.csv", index=False)
    sens = []
    for label, sub in (("both", p), ("A", p[p.event == "A"]), ("B", p[p.event == "B"])):
        for name, col, fix in (("P90", "p90", None), ("P95", "p95", None), ("8 m/s", None, 8.0), ("10 m/s", None, 10.0)):
            t = table(sub, col, fix)
            for m in MODELS:
                sens.append({"subset": label, "threshold": name, "model": m, **t.loc[m].to_dict()})
    s = pd.DataFrame(sens); s.to_csv(C.OUT / "robust_threshold_sensitivity.csv", index=False)
    pd.set_option("display.width", 200)
    print(b.round(2).to_string(index=False)); print(rk.round(2).to_string(index=False))
    print(s.pivot_table(index=["subset", "threshold"], columns="model",
                        values=["csi_wind", "csi_dust"]).round(2).to_string())
    print(s[s.model == "IFS"][["subset", "threshold", "n_obs_wind"]].to_string(index=False))


if __name__ == "__main__":
    main()
