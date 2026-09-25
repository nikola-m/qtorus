"""Publication-quality plotting helpers (matplotlib)."""
from __future__ import annotations
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


def set_style():
    mpl.rcParams.update({
        "figure.dpi": 140,
        "savefig.dpi": 300,
        "font.size": 10,
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "axes.linewidth": 0.8,
        "axes.grid": False,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "legend.frameon": False,
        "image.cmap": "magma",
    })


def plot_density(ax, geom, field, title=None, cmap="magma", overlay_R=None,
                 center=(0.0, 0.0)):
    """Plot a real-space field |psi|^2 (or V) with correct extent."""
    im = ax.imshow(field.T, origin="lower",
                   extent=[0, geom.Lx, 0, geom.Ly], cmap=cmap, aspect="equal")
    if overlay_R:
        for sx in (0, geom.Lx):
            for sy in (0, geom.Ly):
                c = plt.Circle((center[0] + sx if center[0] < 1e-9 else center[0],
                                center[1] + sy if center[1] < 1e-9 else center[1]),
                               overlay_R, fill=False, ec="cyan", lw=0.8, ls="--")
                ax.add_patch(c)
    ax.set_xlabel("$x$"); ax.set_ylabel("$y$")
    if title:
        ax.set_title(title)
    return im


def plot_nnsd(ax, spacings, beta=None, label=None):
    s = np.linspace(0, 4, 300)
    ax.hist(spacings, bins=np.linspace(0, 4, 31), density=True,
            color="0.7", edgecolor="0.3", label=label)
    ax.plot(s, np.exp(-s), "b--", lw=1.4, label="Poisson")
    ax.plot(s, (np.pi / 2) * s * np.exp(-np.pi * s**2 / 4), "r-", lw=1.4,
            label="GOE (Wigner)")
    if beta is not None:
        from .statistics import brody_pdf
        ax.plot(s, brody_pdf(s, beta), "k:", lw=1.6,
                label=fr"Brody $\beta={beta:.2f}$")
    ax.set_xlabel("$s$"); ax.set_ylabel("$P(s)$")
    ax.set_xlim(0, 4)
