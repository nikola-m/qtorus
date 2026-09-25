"""Figure: plane-wave basis convergence for a sharp vs a smooth obstacle.

Tracks a fixed interior eigenvalue (the 10th of the A1 sector) as the basis
cutoff mmax grows.  A discontinuous (sharp) obstacle converges only
algebraically (Gibbs; Norton & Scheichl, SINUM 2010); a smooth wall converges
near-exponentially.  This motivates the smoothed default for spectral work.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh
from qtorus import TorusGeometry, CircularPotential, sector_hamiltonian
from qtorus.viz import set_style

set_style()
OUT = os.path.join(os.path.dirname(__file__), "..", "paper", "figures")
os.makedirs(OUT, exist_ok=True)

geom = TorusGeometry(1.0, 1.6180339887, 320)
LEVEL = 10
mmaxes = np.array([6, 8, 10, 12, 14, 16, 18, 20, 24, 28, 32])


def track(profile, w=0.02):
    vals = []
    for mm in mmaxes:
        pot = CircularPotential(geom, R=0.25, V0=300.0, center=(0.0, 0.0),
                                profile=profile, w=w)
        Hs, _ = sector_hamiltonian(geom, pot, int(mm), "A1")
        vals.append(np.sort(eigh(Hs, eigvals_only=True))[LEVEL])
    return np.array(vals)

sharp = track("sharp")
smooth = track("smooth", w=0.02)
err_sharp = np.abs(sharp[:-1] - sharp[-1]) / abs(sharp[-1])
err_smooth = np.abs(smooth[:-1] - smooth[-1]) / abs(smooth[-1])

fig, ax = plt.subplots(figsize=(4.4, 3.4))
ax.semilogy(mmaxes[:-1], err_sharp, "s-", color="#c0392b", ms=4,
            label="sharp disk (top-hat)")
ax.semilogy(mmaxes[:-1], err_smooth, "o-", color="#2471a3", ms=4,
            label=r"smooth disk ($w=0.02$)")
# reference algebraic slope ~ mmax^{-3} for the top-hat
ax.semilogy(mmaxes[:-1], 0.5 * (mmaxes[:-1] / 6.0)**(-3.0), "k:", lw=1,
            label=r"$\propto m_{\max}^{-3}$")
ax.set_xlabel(r"basis cutoff $m_{\max}$")
ax.set_ylabel(r"relative error in $E_{10}$")
ax.set_title("Spectral convergence: sharp vs smooth obstacle")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig_convergence.pdf"))
fig.savefig(os.path.join(OUT, "fig_convergence.png"), dpi=200)
print("saved fig_convergence; final E10 sharp=%.6f smooth=%.6f" % (sharp[-1], smooth[-1]))
