"""Investigate accuracy of various the ensemble (EnOpt) gradient estimates

**for robust/average objective functions**.
"""

## Preamble
from datetime import datetime
from pathlib import Path
import pickle

import matplotlib.pyplot as plt
from mpl_tools.place import freshfig
import numpy as np
import numpy.random as rnd
from scipy.special import hermite
import scipy.stats as ss
from tqdm import tqdm
import xarray as xr

from toolz import parallelize, cntr, rinv

plt.ion()
np.seterr(divide='raise', invalid='raise')
timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")


## Experiment params
# For debugging, use iHermite=1 or 0 and/or N8.
d = 5
N8 = 10**5  # 8 stands for "infinity"

x_mean = np.linspace(-2, 2, d)
x_cov  = .5**2 * np.eye(d)

u_mean = np.zeros(d)
u_cov  = .1**2 * np.eye(d)

# Distributions
rvX = ss.multivariate_normal(x_mean, x_cov)
rvU = ss.multivariate_normal(u_mean, u_cov)
# Marginals (for plotting)
rvXi = lambda i: ss.norm(x_mean[i], np.sqrt(x_cov[i, i]))
rvUi = lambda i: ss.norm(u_mean[i], np.sqrt(u_cov[i, i]))

# Always 2d
# PS: use "data mat" orientation: a row is an obs, ie sample pt ⇒ easier np broadcast.
sampleX = lambda N: rvX.rvs(N).reshape((N, d))
sampleU = lambda N: rvU.rvs(N).reshape((N, d))

# The objective function must support input of any shape of ndim >= 2,
# where the "physical" dimension is axis=-1. Also, possibly, x.shape != u.shape,
# in which case the function must broadcast appropriately.
def J(x, u):
    return hermite(iHermite)(x + u).sum(-1)

def J1(x, u):
    return hermite(iHermite).deriv()(x + u)

# Validate derivative/TLM/adjoint
# x = (-1)**np.arange(d)
# u = x
# I = np.eye(d)
# du = 1e-7
# df = J(x, u + du*I) - J(x, u)
# assert np.allclose(J1(x, u), df/du, rtol=1e-4)

## Illustrate
uu = np.linspace(-2, 2, 201)
fig, ax = freshfig("Elements of the experiments")
nHermite = 7
cm = plt.color_sequences["tab10"]
for iHermite in range(nHermite):
    lbl = fr"$\mathcal{{H}}_{iHermite}$"
    xY = 1
    if iHermite >= 4:
        xY = 10
        lbl += f"/{xY}"
    ax.plot(uu, J(0, uu[:, None]) / xY, lw=3, label=lbl, color=cm[iHermite])
hermite_legend = ax.legend(framealpha=.6, ncols=2, title="Hermite polynomial")
a, b = -10, 15
ax.set_ylim(a, b)
ax.set_xlim(uu.min(), uu.max())
ax.axhline(color="k", ls="-", lw=1, alpha=.4)
ax.axvline(color="k", ls="-", lw=1, alpha=.4)
for i in range(d):
    lbl = lambda var: f"$p({var}_i)$" if not i else None
    h1 = ax.fill_between(uu, a + (b - a)/11*rvUi(i).pdf(uu), a, label=lbl("u"), color="C9", alpha=.3)
    h2 = ax.fill_between(uu, a + (b - a)/4 *rvXi(i).pdf(uu), a, label=lbl("x"), color="C7", alpha=.3)
    if not i:
        pdf_handles = [h1, h2]

ax.add_artist(hermite_legend)
ax.legend(handles=pdf_handles, title="Density", framealpha=1, loc="lower right")
ax.set_xticks(np.arange(-2, 3))
ax.set_yticks(np.arange(a, b, 10))
ax.set_xlabel("$x, u$")

# Save figure
if False:
    fig.tight_layout()
    __file__ = "robust-grad.py"
    kws = dict(bbox_inches='tight', pad_inches=0)
    pdir = Path(__file__).resolve().parent / "images"
    fig.savefig(pdir / (Path(__file__).stem + "-illust.pdf").replace(" ", "_"), **kws)

## Gradient estimators
# Must all support calling with args (X, U, V, Y)

pinv = lambda A: rinv(A, reg=reg, tikh=tikh)

def expect_J1(X, U, *_):
    """Expected (wrt. x and u) actual derivative(vs. u)"""
    return J1(X, U).mean(0)

def fragile(X, U, *_):
    """Non-robust EnOpt."""
    return pinv(cntr(U)) @ J(X.mean(0), U)

def naive(X, U, *_):
    """Average (size N) of plain LLS estimators for each x (size N)."""
    # N8 might be too large
    X = X[:10**3]
    U = U[:10**3]

    X1 = np.expand_dims(X, 1)  # ⇒ shape = (N, 1, d)
    grads = [fragile(x, U) for x in X1]  # Manual loop. Also works w/ paired()
    # grads = (pinv(cntr(U)) @ J(X1, U).T).T  # Vectorized
    return np.mean(grads, 0)

def paired(X, U, *_):
    """Original paired robust gradient."""
    return pinv(cntr(U)) @ J(X, U)

def duplex(X, U, V, *_):
    """Duplex paired gradient. Decouple effects of X and U."""
    return pinv(U - V) @ (J(X, U) - J(X, V))

def mirrored(X, U, *_):
    """Mirror members."""
    V = 2*U.mean(0) - U
    # return pinv(cntr(U)) @ (J(X, U) - J(X, V))/2
    return duplex(X, U, V)

def stosag(X, U, *_):
    """Mirror values (use mu)."""
    u = U.mean(0)
    # return pinv(cntr(U)) @ (J(X, U) - J(X, u))
    return duplex(X, U, u)

def meanLLS2_wrapper(xd):
    """Like `partial(meanLLS2, xd=xd)` but with pretty `__name__`."""
    def meanLLS2(X, U, V, *_):
        """Average of plain LLS estimators, each using 2 members."""
        # Since N=2 for [u, v], their anomalies are symmetric, say uc = (u-v)/2 = -vc.
        # By pen-and-paper, pinv := [uc; -vc]^+ = [z, -z] with z = uc^+ = 2*uc/(uc@uc).
        # Thus, pinv @ [J(x, u), J(x, v)] = 2 * z * (J(x, u) - J(x, v)).
        Uc = (U - V)
        pinvs = Uc.T / np.sum(Uc**2, 1)
        grad = pinvs @ (J(X, U) - J(X, V)) / len(U)

        # Implementation with actual pinv (also requires loop)
        # def lls(x, u, v):
        #     uvc = cntr(np.vstack((u, v)))
        #     return pinv(uvc) @ [J(x, u), J(x, v)]
        # grads = [lls(*args) for args in zip(X, U, V)]
        # grad = np.mean(grads, 0)

        # Rank compensation (since, like SPSA, we only ever investigate 1 dir)
        # TODO: Make some heuristic to estimate in the general case?
        if xd:
            grad *= d

        return grad
    meanLLS2.__name__ += f"({xd=})"
    return meanLLS2

def hybrid(X, U, V, *_):
    """Use 2-member crosscov's (like `duplex`), but full-ensemble cov (like `paired`)."""
    N = len(U)
    crosscov = (U - V).T @ (J(X, U) - J(X, V)) / (2*N)
    UV = cntr(np.concatenate([U, V]))
    Cu = UV.T @ UV / (2*N - 1)
    return pinv(Cu) @ crosscov

def decorrelated(X, U, *_):
    """Decorrelate J(X, mu) and U **before** running J(X, U)."""
    if len(U) <= 2:
        return paired(X, U)

    def refit(V, U):
        sratio = np.std(U) / np.std(V)
        return U.mean(0) + (V - V.mean(0))*sratio

    mu = U.mean(0)
    JX = cntr(J(X, mu))
    n2 = JX@JX
    V = U - np.outer(JX, JX@U / n2) if n2 > 1e-10 else U
    V = refit(V, U)
    grad = pinv(cntr(V)) @ J(X, V)
    return grad


## Run experiments
coords = dict(
    # NB: Avoid keys "method" and "tolerance" -- they're kwargs to `.sel()`.
    iHermite = [1, 2, 3, 4, 5, 6],
    # iHermite = [3],
    tikh     = [True],
    # tikh     = [True, False],
    center   = [True, False],
    # center   = [False],
    reg      = [1e-2, 2e-2, 5e-2, .1, .2, .5, 1, 2, 5],
    # reg      = [1e-2, .1, 1],
    seed     = range(1 * 10**6),
    # seed     = range(1 * 10**4),
    ens_size = [2, 3, 5, 6, 10, 20, 100, 400, 1000],
    # ens_size = [3, 10],
    function = [
        expect_J1,
        naive,
        paired,
        fragile,
        duplex,
        mirrored,
        stosag,
        meanLLS2_wrapper(xd=True),
        meanLLS2_wrapper(xd=False),
        hybrid,
        decorrelated,
        ],
    element  = range(d),
)
coords8 = {k: coords[k] for k in ["iHermite", "function", "element"]}

estimates  = xr.DataArray(coords=coords,  dims=coords.keys())
estimates8 = xr.DataArray(coords=coords8, dims=coords8.keys())

def iterdim(dim, is_top_level=False):
    return tqdm(coords[dim], desc=dim, leave=is_top_level, smoothing=.05)

for iHermite in iterdim("iHermite", True):

    # Asymptotics
    center = True
    reg    = 1e-2
    tikh   = False
    rnd.seed(300)  # for exact reproducibility
    X8 = sampleX(N8)
    U8 = sampleU(N8)
    V8 = sampleU(N8)
    for function in coords["function"]:
        estim = function(X8, U8, V8)
        estimates8.loc[dict(iHermite=iHermite, function=function)] = estim

    # Finite-N
    for center in iterdim("center"):
        for tikh in iterdim("tikh"):
            for reg in iterdim("reg"):
                loc_coord = dict(iHermite=iHermite, tikh=tikh, reg=reg, center=center)

                def compute(seed_batch):
                    """NB: since it's parallelized, this "block" cannot *set* globals!"""
                    assert estimates.dims[-4:] == ('seed', 'ens_size', 'function', 'element'),\
                        "The dimension ordering must be as specified to the array below."
                    estims = np.zeros(seed_batch.shape + estimates.shape[-3:])
                    for iSeed, seed in enumerate(seed_batch):
                        for iN, N in enumerate(coords["ens_size"]):
                            rnd.seed(3000 + seed)  # for var reduction (CRN)
                            X = sampleX(N)
                            U = sampleU(N)
                            V = sampleU(N)
                            if center:
                                U = cntr(U) + rvU.mean
                                V = cntr(V) + rvU.mean
                            for iFun, function in enumerate(coords["function"]):
                                    estims[iSeed, iN, iFun] = function(X, U, V)
                    return estims

                buffer = parallelize(compute, coords["seed"])

                assert estimates.loc[loc_coord].shape == buffer.shape,\
                    ("There's probably a coord (dim) missing in loc_coord "
                     "(would cause broadcast `buffer` in the assignment below).")
                estimates.loc[loc_coord] = buffer


## Process results
errs = estimates - estimates8.sel(function=expect_J1)

ds = xr.Dataset({
    "Mean estim": estimates.mean(["seed"]),
    "RMS err": np.sqrt((errs**2).mean(["seed"])),
    "ABS bias": np.abs(errs.mean(["seed"])),
    "RMS dev": np.sqrt(errs.var(["seed"])),
})
# Sanity check: np.allclose(0, ...) on:
# ds["ABS bias"]**2 + ds["RMS dev"]**2 - ds["RMS errs"]**2


## Save
ds.attrs["estimates8"] = estimates8

# Make primitive
for coo in [ds.coords, estimates8.coords]:
    coo["function"] = [f.item().__name__ for f in coo["function"]]

data_dir = Path.home() / "data" / "robust-grad"
save_to = data_dir / timestamp
print("Saving to", save_to)
save_to.write_bytes(pickle.dumps(ds))

## Show results
# __import__("IPython").get_ipython().run_line_magic("run", "robust-grad-results.py")
