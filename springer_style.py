"""Export figures to Earth Systems and Environment (Springer) artwork specifications.

* width 174 mm (double column) or 84 mm (single column); height <= 234 mm
* lettering 8 pt (9 pt for panel letters), Arial/Helvetica (Liberation Sans where Arial is absent)
* no titles or captions inside figures: only panel letters (a), (b), ... are kept
* TIFF at 600 dpi (LZW) plus vector PDF, named Fig<number>.tif / Fig<number>.pdf in results/springer/
"""
import logging
import matplotlib
import matplotlib.lines
import matplotlib.text

import config as C

logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)   # Arial absent on Linux: Liberation Sans is metric-compatible

OUT = C.OUT / "springer"
OUT.mkdir(parents=True, exist_ok=True)
DOUBLE, SINGLE = 174 / 25.4, 84 / 25.4      # inches
matplotlib.rcParams["font.family"] = ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans", "sans-serif"]


def to_springer(fig, number, width="double"):
    """Rescale a finished figure to Springer size and save Fig<number>.tif/.pdf."""
    w, h = fig.get_size_inches()
    W = SINGLE if width == "single" else DOUBLE
    k = W / w                                   # shrink factor
    lk = max(k, 0.55)                           # keep lines visible after shrinking
    # titles: keep the panel letter only
    for ax in fig.axes:
        for loc in ("left", "center", "right"):
            t = ax.get_title(loc=loc)
            ax.set_title("", loc=loc)
            if t.startswith("("):
                ax.set_title(t.split(")")[0] + ")", loc="left", fontsize=9, fontweight="bold", pad=3)
    if fig._suptitle is not None:
        fig._suptitle.set_text("")
    for t in fig.findobj(matplotlib.text.Text):
        if t.get_text() and t.get_fontweight() != "bold":
            t.set_fontsize(8 if t.get_fontsize() >= 10.5 else 7)
        t.set_fontfamily(matplotlib.rcParams["font.family"])
    for ln in fig.findobj(matplotlib.lines.Line2D):
        ln.set_linewidth(max(ln.get_linewidth() * lk, 0.5))
        ln.set_markersize(ln.get_markersize() * lk)
    for ax in fig.axes:
        for s in ax.spines.values():
            s.set_linewidth(0.8)
        ax.tick_params(width=0.8, length=3, labelsize=8)
        for coll in ax.collections:
            try:
                coll.set_linewidths([max(x * lk, 0.4) for x in coll.get_linewidths()])
                if hasattr(coll, "get_sizes") and len(coll.get_sizes()):
                    coll.set_sizes(coll.get_sizes() * lk * lk)
            except Exception:
                pass
        for p in ax.patches:
            p.set_linewidth(max(p.get_linewidth() * lk, 0.5))
    # long model legends: one legend row under the panels instead of inside an axis
    legs = [(ax, ax.get_legend()) for ax in fig.axes if ax.get_legend() is not None]
    moved = None
    for ax, lg in legs:
        if len(lg.legend_handles) >= 5 and len(fig.axes) > 1 - (width == "single"):
            moved = (lg.legend_handles, [t.get_text() for t in lg.get_texts()])
            lg.remove()
            break
    for ax, lg in legs:
        if ax.get_legend() is not None:
            for t in ax.get_legend().get_texts():
                t.set_fontsize(7)
            ax.get_legend()._legend_box.sep = 2
    H = min(h * k * 1.25, 234 / 25.4)          # a little taller: text no longer shrinks with the figure
    fig.set_size_inches(W, H + (0.28 if moved else 0) + (0.2 if moved and width == "single" else 0))
    if moved:
        n = len(moved[0])
        fig.legend(moved[0], moved[1], loc="lower center", ncol=(3 if width == "single" else n) if n <= 5 else (n + 1) // 2, fontsize=8,
                   frameon=False, handlelength=2.6, columnspacing=1.6, bbox_to_anchor=(0.5, 0.0))
        fig.tight_layout(pad=0.5, rect=(0, (0.3 if (n <= 5 and width != "single") else 0.5) / (H + 0.28), 1, 1))
    else:
        fig.tight_layout(pad=0.5)
    fig.savefig(OUT / f"Fig{number}.tif", dpi=600, facecolor="white", pil_kwargs={"compression": "tiff_lzw"})
    fig.savefig(OUT / f"Fig{number}.pdf", facecolor="white")
    fig.savefig(OUT / f"Fig{number}_preview.png", dpi=200, facecolor="white")
    print(f"saved springer/Fig{number}.tif  ({W * 25.4:.0f} x {fig.get_size_inches()[1] * 25.4:.0f} mm, 600 dpi)")
