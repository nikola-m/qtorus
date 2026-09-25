"""Spectral statistics for quantum-chaos diagnostics.

All statistics are computed on the *unfolded* spectrum (unit mean level
spacing) of a single symmetry sector.  Reference laws:

    Poisson (integrable):  P(s) = exp(-s),           Sigma2(L) = L
    GOE (chaotic, T-sym) :  P(s) ~ (pi/2) s exp(-pi s^2/4)   [Wigner surmise]
                            Sigma2(L) ~ (2/pi^2)[ln(2 pi L)+gamma+1-pi^2/8]

The Brody interpolation P_beta(s) = (beta+1) b s^beta exp(-b s^{beta+1}),
b = Gamma((beta+2)/(beta+1))^{beta+1}, has beta=0 (Poisson) and beta=1 (GOE
surmise) as limits and provides a single-number chaos indicator.
"""
from __future__ import annotations
import numpy as np
from math import gamma
from scipy.optimize import minimize_scalar

EULER_GAMMA = 0.5772156649015329


# ---- reference distributions -----------------------------------------
def poisson_pdf(s):
    return np.exp(-np.asarray(s, float))


def goe_pdf(s):
    s = np.asarray(s, float)
    return (np.pi / 2) * s * np.exp(-np.pi * s**2 / 4)


def _brody_b(beta):
    return gamma((beta + 2) / (beta + 1)) ** (beta + 1)


def brody_pdf(s, beta):
    s = np.asarray(s, float)
    b = _brody_b(beta)
    return (beta + 1) * b * s**beta * np.exp(-b * s**(beta + 1))


# ---- unfolding --------------------------------------------------------
def unfold_spectrum(E, deg=12, trim=0, return_energy=False):
    """Polynomial unfolding.  Fit a degree-`deg` polynomial to the integrated
    density of states (staircase) and remap so the mean spacing is 1.

    Parameters
    ----------
    E : 1-D array of eigenvalues (one symmetry sector), any order.
    deg : polynomial degree for the smooth staircase.
    trim : drop this many levels from each end after unfolding (edge effects).

    Returns the unfolded levels (sorted, ~unit mean spacing).
    """
    E = np.sort(np.asarray(E, float))
    N = np.arange(1, E.size + 1) - 0.5          # midpoint staircase
    coef = np.polyfit(E, N, deg)
    x = np.polyval(coef, E)
    Etr = E
    if trim:
        x = x[trim:-trim]
        Etr = E[trim:-trim]
    if return_energy:
        return Etr, x
    return x


# ---- nearest-neighbour spacing distribution --------------------------
def nnsd(unfolded):
    """Nearest-neighbour spacings of an unfolded spectrum (mean ~ 1)."""
    s = np.diff(np.sort(unfolded))
    return s / s.mean()


# ---- number variance --------------------------------------------------
def sigma2(unfolded, Lvals, n_origins=400):
    """Number variance Sigma^2(L) = <(n(L) - L)^2> over sliding windows."""
    x = np.sort(unfolded)
    xmin, xmax = x[0], x[-1]
    out = []
    for L in np.atleast_1d(Lvals):
        starts = np.linspace(xmin, xmax - L, n_origins)
        counts = np.searchsorted(x, starts + L) - np.searchsorted(x, starts)
        out.append(np.var(counts))
    return np.array(out)


# ---- spectral rigidity (Dyson-Mehta Delta_3) -------------------------
def delta3(unfolded, Lvals, n_origins=200, n_samp=200):
    """Delta_3(L): least-squares deviation of the staircase from a straight
    line, averaged over windows.  Numerically robust (direct minimisation)."""
    x = np.sort(unfolded)
    xmin, xmax = x[0], x[-1]
    out = []
    for L in np.atleast_1d(Lvals):
        starts = np.linspace(xmin, xmax - L, n_origins)
        vals = []
        for a in starts:
            xi = np.linspace(a, a + L, n_samp)
            Nx = np.searchsorted(x, xi).astype(float)
            A = np.vstack([np.ones_like(xi), xi]).T
            coef, *_ = np.linalg.lstsq(A, Nx, rcond=None)
            resid = Nx - A @ coef
            vals.append(np.trapezoid(resid**2, xi) / L)
        out.append(np.mean(vals))
    return np.array(out)


def sigma2_goe(L):
    L = np.asarray(L, float)
    return (2 / np.pi**2) * (np.log(2 * np.pi * L) + EULER_GAMMA + 1 - np.pi**2 / 8)


# ---- Brody fit (maximum likelihood) ----------------------------------
def brody_fit(spacings):
    """MLE of the Brody parameter beta in [0, 1] from unfolded spacings."""
    s = np.asarray(spacings, float)
    s = s[s > 0]

    def negloglik(beta):
        b = _brody_b(beta)
        ll = (np.log((beta + 1) * b) + beta * np.log(s) - b * s**(beta + 1))
        return -np.sum(ll)

    res = minimize_scalar(negloglik, bounds=(1e-4, 1.5), method="bounded")
    return float(res.x)


# ---- multi-sector pooling & energy-resolved statistics ----------------
def pooled_nnsd(energy_list, deg=12, trim=10):
    """Unfold each symmetry sector *separately* and pool the spacings.

    Statistics must not mix symmetry classes; the correct way to gain levels is
    to compute several sectors independently and combine their (individually
    unfolded) nearest-neighbour spacings.  ``energy_list`` is a list of raw
    eigenvalue arrays, one per sector.  Returns the pooled spacings (mean ~ 1).
    """
    chunks = []
    for E in energy_list:
        x = unfold_spectrum(E, deg=deg, trim=trim)
        chunks.append(np.diff(np.sort(x)))
    s = np.concatenate(chunks)
    return s / s.mean()


def banded_statistics(energy_list, bands, deg=12, trim=10):
    """Energy-resolved chaos indicator.

    Because the chaotic fraction of phase space depends on E/V0, the Brody
    parameter is reported in energy windows.  Each sector is unfolded over its
    full converged sequence (good staircase fit); within each band the pooled
    unfolded spacings are fitted.  ``bands`` is a list of (E_lo, E_hi).
    Returns a list of dicts: {"band", "n", "beta"}.
    """
    unf = [unfold_spectrum(E, deg=deg, trim=trim, return_energy=True)
           for E in energy_list]
    out = []
    for lo, hi in bands:
        pooled = []
        for Es, xs in unf:
            xb = xs[(Es >= lo) & (Es <= hi)]
            if xb.size > 2:
                pooled.append(np.diff(np.sort(xb)))
        if not pooled:
            out.append({"band": (lo, hi), "n": 0, "beta": float("nan")})
            continue
        s = np.concatenate(pooled); s = s / s.mean()
        out.append({"band": (lo, hi), "n": int(s.size), "beta": brody_fit(s)})
    return out
