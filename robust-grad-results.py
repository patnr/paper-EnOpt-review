"""Show results."""

## Preamble
from datetime import datetime
from pathlib import Path
import pickle

import colorama as colr
import matplotlib.pyplot as plt
# from toolz import sort_legend
import numpy as np
from pandas import pandas as pd  # type: ignore

colr.init()  # enable on Windows

plt.ion()

ff = "%7.2f"
printopts = dict(precision=4, threshold=8, formatter=dict(float=lambda x: ff % x))
pd.set_option("display.precision", 2)

def find_latest_run(root: Path, timestamp="%Y-%m-%dT%H-%M-%S"):
    """Find the latest experiment (dir containing many)"""
    lst = []
    for f in root.iterdir():
        try:
            f = datetime.strptime(f.name, timestamp)
        except ValueError:
            pass
        else:
            lst.append(f)
    f = max(lst)
    f = datetime.strftime(f, timestamp)
    return f


def stripe(rows, inds=slice(None)):
    """Apply 'shading' to alternate lines in `rows` (str)."""
    if not isinstance(rows, list):
        rows = rows.splitlines()
    inds = list(range(len(rows)))[inds]
    for i in inds:
        if i % 2:
            rows[i] = colr.Fore.BLACK + colr.Back.WHITE + rows[i] + colr.Style.RESET_ALL
    rows = "\n".join(rows)
    return rows


def piecewise(x1, a1, x2=np.inf, a2=1.0):
    """Create monotonic, piecewise (linear) & inverse fun's for axis transform.

    - For      `x < x1` there is no transformation.
    - For `x1 < x < x2` intervals get compressed by a factor `a1`.
    - For `x2 < x`      intervals get compressed by a factor `a2`.

    Example:
    >>> f, fi = piecewise(10, 10, 100, 100)
    ... arr = np.array([1, 2, 3, 10, 20, 30, 100, 200, 300])
    >>> f(arr)
    array([ 1.,  2.,  3., 10., 11., 12., 19., 20., 21.])
    >>> all(fi(f(arr)) == arr)
    True
    """
    y1 = x1
    y2 = x1 + (x2 - x1) / a1

    def fun(x):
        y = np.zeros_like(x, dtype=float)
        o0 = x < x1
        o1 = (x1 <= x) & (x < x2)
        o2 = x2 <= x
        y[o0] = x[o0]
        y[o1] = y1 + (x[o1] - x1) / a1
        y[o2] = y2 + (x[o2] - x2) / a2
        return y

    def inv(y):
        x = np.zeros_like(y, dtype=float)
        o0 = y < y1
        o1 = (y1 <= y) & (y < y2)
        o2 = y2 <= y
        x[o0] = y[o0]
        x[o1] = x1 + (y[o1] - y1) * a1
        x[o2] = x2 + (y[o2] - y2) * a2
        return x

    return fun, inv

## Linestyle
def linestyle(name):
    dct = dict(
        label=name,
        marker=".",
        zorder=2,
        ms=5**2,
        markeredgewidth=.5,
        markeredgecolor="w",
        ls="-",
    )
    dct2 = {}
    if "expect_J1" in name:
        dct2 = dict(
            c=[.4, .4, .4],
            marker="v",
            label=r"$M$: Avrg. $\nabla J$",
        )
    if "paired" in name:
        dct2 = dict(
            c=[.4, .4, .4],
            marker="^",
            label=r"$M$: Paired",
        )
    if "fragile" in name:
        dct2 = dict(
            zorder=9,
            c="goldenrod",
            label="$M$: Fragile",
        )
    if "naive" in name:
        dct2 = dict(
            marker="s",
            c="k",
            label=r"$M^2$: Plain LLS",
        )
    if "duplex" in name:
        dct2 = dict(
            color="cornflowerblue",
            marker="H",
            ms=5.2**2,
            zorder=4,
            label="$2 M$: Two-sided",
        )
    if "mirrored" in name:
        dct2 = dict(
            color="darkcyan",
            marker="s",
            ms=4**2,
            zorder=5,
            label="$2 M$: Mirrored",
        )
    if "stosag" in name:
        dct2 = dict(
            marker="+",
            color="b",
            markeredgecolor="b",
            markeredgewidth=5,
            ms=6**2,
            zorder=3,
            label="$2 M$: StoSAG",
        )
    if "hybrid" in name:
        dct2 = dict(
            color="purple",
            marker="d",
            ms=4**2,
            zorder=5,
            label="$2 M$: Hybrid",
        )
    if "meanLLS2" in name:
        dct2 = dict(
            color="crimson",
            marker="P",
            ms=4**2,
            zorder=5,
            label="$2 M$: Avrg. LLS",
        )
    if "decorrelated" in name:
        dct2 = dict(
            color="C1",
            marker="X",
            ms=3**2,
            zorder=5,
            label="$2 M$: Decorr'd",
        )
    dct.update(dct2)
    # dct["label"] = name  # undo relabelling
    return dct


## Load
data_dir = Path("~").expanduser() / "data" / "robust-grad"
load_from = (
    # find_latest_run(data_dir)
    # "2023-03-26T10-39-24"  # iHermite 0<=6, nTrial=10**6, maxN=1000
    # "2023-04-14T11-01-33"  # with x_mean += 3
    "2023-04-12T13-17-23"  # nTrial=10**5, with & w/o centering
    # "2023-04-20T17-40-24"  # iHermite=3, nTrial=10**6
)
load_from = data_dir / load_from
print("Loading from", load_from)
ds = pickle.loads(load_from.read_bytes())


## Process
# Average stats over elements (of control vector)
ds = ds.mean("element")

# Sub-select
# xcld = []
xcld = ["xd=True"]
# xcld = ["hybrid", "decorrelated", "meanLLS2"]
ds = ds.isel(function=[not any(name in f.item() for name in xcld) for f in ds.function])
ds = ds.sel(iHermite=[2, 3, 5])
# ds = ds.sel(reg=.1)
# ds = ds.sel(center=False)

# Optimize over tuning params
# stat0 = "ABS bias"
stat0 = "RMS err"
# stat0 = "RMS dev"
arg0 = ds[stat0].argmin(["tikh", "reg", "center"])
ds0 = ds.isel(arg0)  # == rms_bias.min(...)


## Show
stat_kinds = {
    # "ABS bias": 3,
    "RMS err": 3,
    # "RMS dev": 3,
    # "RMS dev": 10,
    # "reg": 1,
    # "center": 1,
    # "tikh": 2,
}
# Filter for plot-able
stat_kinds = {k: stat_kinds[k] for k in stat_kinds
              if (k in ds0.data_vars)  # computed stats
              or (k in ds0.coords and k not in ds0.dims)  # argmin coords
              }
# Enlarge panel of stat0
stat_kinds[stat0] = 10

fignum = "Elements of the experiments"
fig = plt.figure(fignum)
fig.clear()
fig, axs = plt.subplots(num=fignum,
    squeeze=False, ncols=len(ds0.coords["iHermite"]),
    # figsize=(10, 5),
    figsize=(5, 4),
    sharex=True, nrows=len(stat_kinds),
    gridspec_kw=dict(height_ratios=stat_kinds.values()))
legends = []

for j, iHermite in enumerate(ds0.coords["iHermite"].values):
    print("\n", colr.Back.YELLOW, f"{iHermite=}", colr.Style.RESET_ALL, sep="")
    df = (ds0
          .sel(iHermite=iHermite, drop=True)
          .to_dataframe()
          # .rename_axis("param", axis="columns")  # collective column name
          )

    # Long/Stack/Record format --> Wide
    df1 = (df
           .reset_index(level="ens_size")
           .pivot(columns="ens_size", values=list(stat_kinds))
           )

    # Sort (jumbled, possibly MultiIndex) index
    lv_names = list(df1.index.names)
    ordering = [ds0.coords[dim].values for dim in lv_names]
    if len(ordering) == 1:
        ordering = [['paired', 'naive', 'fragile', 'expect_J1',
                     'duplex', 'mirrored', 'stosag',
                     'meanLLS2(xd=False)', 'hybrid', 'decorrelated']]
        df2 = df1.reindex(ordering[0])
    else:
        # First, swap levels
        lv1, lv2 = 0, 1
        ordering[lv1], ordering[lv2] = ordering[lv2], ordering[lv1]
        lv_names[lv1], lv_names[lv2] = lv_names[lv2], lv_names[lv1]
        df2 = df1.swaplevel(lv1, lv2)
        df2 = df2.reindex(pd.MultiIndex.from_product(ordering))
    df2.index.names = lv_names

    # Plot
    for i, stat in enumerate(stat_kinds):
        # Setup axes
        ax = axs[i, j]
        # do_legend = (stat == stat0)
        do_legend = False
        ax.grid(True, which="both", axis="both")
        if i == 0:
            ax.set_title(f"Hermite {iHermite}")
        if j == 0:
            ax.set_ylabel(stat)

        # Cast bool to int to enable plotting
        val0 = df2[stat].values.ravel()[0]
        yscale = "log"
        if isinstance(val0, (bool, int)):
            yscale = "linear"
            df2[stat] = df2[stat].astype(int)

        df2[stat].T.plot(ax=ax, legend=False, marker=".")

        # Style
        for ln in ax.lines:
            ln.set(**linestyle(ln.get_label()))
        if do_legend:
            ax.legend(handles=np.array_split(ax.lines, len(axs[0]))[j].tolist(),
                      loc="lower left", labelspacing=1.2, edgecolor="b")

        ax.set_yscale(yscale)

        # Rename axes
        ax.set_ylabel(ax.get_ylabel().replace("err", "error"))
        ax.set_xlabel(ax.get_xlabel().replace("ens_size", "Ens. size (M)"))


    # Make stats the subcolumns
    # df3 = (df2
    #        .swaplevel(axis=1)
    #        .sort_index(axis=1)
    #        )
    df3 = df2
    # Print
    print(stripe(str(df3[stat0]), slice(2, None)))

ax.set_xscale("log")
ax.set_xticks(ds0.coords["ens_size"].values)
ax.set_xticklabels(ds0.coords["ens_size"].values)
fig.tight_layout()

# Save figure
if False:
    kws = dict(bbox_inches='tight', pad_inches=0)
    pdir = Path(__file__).resolve().parent / "images"
    fig.savefig(pdir / (Path(__file__).stem + f"-{stat0}.pdf").replace(" ", "_"), **kws)
