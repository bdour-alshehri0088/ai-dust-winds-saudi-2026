"""Fig. 2: verification workflow in nine steps (Springer artwork: 174 mm, 8 pt, 600 dpi).

Output: results/springer/Fig2.tif and Fig2.pdf.  Usage: python 11_fig_workflow.py
"""
import logging
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

import config as C

logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)   # Arial absent on Linux: Liberation Sans is metric-compatible

OUT = C.OUT / "springer"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.family": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"], "font.size": 8,
                     "pdf.fonttype": 42})
STEPS = [  # (number, title, detail, kind)  kind: main | key | supp
    (1, "Verification design", "2 storms, 10 common initialisations,\nforecast days 1–6", "main"),
    (2, "Forecasts", "IFS, AIFS (ECMWF open data);\nGraphCast, Aurora, Pangu (NOAA)", "main"),
    (3, "Observations", "METAR at 9 airports,\nunit conversion, QC, dust codes", "main"),
    (4, "Thresholds", "Emission threshold uₜ;\nstation strong-wind P90", "main"),
    (5, "Matching", "Grid to station;\nnearest report within ±30 min", "main"),
    (6, "Verification metrics", "RMSE, direction, peak wind,\nTests 1–2 (CSI), DUP", "key"),
    (7, "Surface susceptibility", "E = S × G × B\n(Online Resource 1)", "supp"),
    (8, "Smoothing", "Activity ratio, kinetic-energy\nspectra against ERA5", "main"),
    (9, "Statistics", "Block bootstrap against IFS;\nper-storm consistency", "main"),
]
STYLE = {"main": dict(fc="white", ec="#4D4D4D", ls="-", lw=0.8),
         "key": dict(fc="#DCE6F2", ec="#0F4D92", ls="-", lw=1.4),
         "supp": dict(fc="white", ec="#9E9E9E", ls="--", lw=0.8)}
# serpentine layout: row 1 left->right, row 2 right->left, row 3 left->right
POS = {1: (0, 2), 2: (1, 2), 3: (2, 2), 4: (2, 1), 5: (1, 1), 6: (0, 1), 7: (0, 0), 8: (1, 0), 9: (2, 0)}
BW, BH, GX, GY = 2.0, 0.78, 0.55, 0.42


def centre(n):
    c, r = POS[n]
    return c * (BW + GX) + BW / 2, r * (BH + GY) + BH / 2


fig, ax = plt.subplots(figsize=(174 / 25.4, 3.0))
for n, title, detail, kind in STEPS:
    x, y = centre(n)
    ax.add_patch(FancyBboxPatch((x - BW / 2, y - BH / 2), BW, BH, boxstyle="round,pad=0,rounding_size=0.08",
                                **STYLE[kind]))
    ax.text(x, y + 0.17, f"{n}. {title}", ha="center", va="center", fontsize=8, fontweight="bold")
    ax.text(x, y - 0.13, detail, ha="center", va="center", fontsize=7, color="#272727", linespacing=1.15)
for a, b in zip(range(1, 9), range(2, 10)):
    (x0, y0), (x1, y1) = centre(a), centre(b)
    if y0 == y1:
        s = 1 if x1 > x0 else -1
        p0, p1 = (x0 + s * BW / 2, y0), (x1 - s * BW / 2, y1)
    else:
        p0, p1 = (x0, y0 - BH / 2), (x1, y1 + BH / 2)
    ax.annotate("", xy=p1, xytext=p0, arrowprops=dict(arrowstyle="-|>", color="#767676", lw=0.9,
                                                      shrinkA=1, shrinkB=1, mutation_scale=8))
ax.set_xlim(-0.05, 3 * BW + 2 * GX + 0.05)
ax.set_ylim(-0.05, 3 * BH + 2 * GY + 0.05)
ax.set_aspect("equal")
ax.axis("off")
fig.tight_layout(pad=0.2)
fig.savefig(OUT / "Fig2.tif", dpi=600, facecolor="white", pil_kwargs={"compression": "tiff_lzw"})
fig.savefig(OUT / "Fig2.pdf", facecolor="white")
fig.savefig(OUT / "Fig2_preview.png", dpi=200, facecolor="white")
print("saved springer/Fig2.tif")
