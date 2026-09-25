"""Figure: obstacle potential, a montage of eigenstate densities, and a Husimi
phase-space slice.  Low states (E < V0) are expelled from the disk (billiard-
like); high states (E > V0) penetrate it -- the quantum signature of a finite,
penetrable obstacle.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib.pyplot as plt
from qtorus import (TorusGeometry, CircularPotential, PlaneWaveHamiltonian,
                    solve_spectrum, husimi)
from qtorus.viz import set_style, plot_density

set_style()
OUT = os.path.join(os.path.dirname(__file__), "..", "paper", "figures")
os.makedirs(OUT, exist_ok=True)

geom = TorusGeometry(1.0, 1.0, 256)
V0 = 800.0
pot = CircularPotential(geom, R=0.25, V0=V0, center=(0.5, 0.5),
                        profile="smooth", w=0.02)
H = PlaneWaveHamiltonian(geom, pot, mmax=20)
E, V = solve_spectrum(H, method="dense")
print("lowest energies:", np.round(E[:8], 1))

# pick indices spanning below and above the barrier
idxs = [0, 3, 12, 40, 90, 160]
fig, axes = plt.subplots(2, 4, figsize=(11.5, 5.8))

im = plot_density(axes[0, 0], geom, pot.sample(), title=r"potential $V(x,y)$",
                  cmap="viridis")
axes[0, 0].add_patch(plt.Circle((0.5, 0.5), 0.25, fill=False, ec="w", lw=0.8, ls="--"))

panel_axes = [axes[0, 1], axes[0, 2], axes[0, 3],
              axes[1, 0], axes[1, 1], axes[1, 2]]
for ax, k in zip(panel_axes, idxs):
    psi = H.eigvec_to_real(V[:, k])
    dens = np.abs(psi)**2
    regime = "E<V_0" if E[k] < V0 else "E>V_0"
    plot_density(ax, geom, dens, title=fr"$n={k}$, $E={E[k]:.0f}$ (${regime}$)")
    ax.add_patch(plt.Circle((0.5, 0.5), 0.25, fill=False, ec="cyan", lw=0.7, ls="--"))

# Husimi slice of a mid state at its dominant momentum magnitude
k = 40
psi = H.eigvec_to_real(V[:, k])
kmag = np.sqrt(E[k])
Hus = husimi(geom, psi, (kmag, 0.0), sigma=0.05)
imh = plot_density(axes[1, 3], geom, Hus,
                   title=fr"Husimi $|k_0|=\sqrt{{E_{{{k}}}}}$, $\hat x$", cmap="cividis")
axes[1, 3].add_patch(plt.Circle((0.5, 0.5), 0.25, fill=False, ec="w", lw=0.7, ls="--"))

fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig_eigenstates.pdf"))
fig.savefig(os.path.join(OUT, "fig_eigenstates.png"), dpi=200)
print("saved fig_eigenstates")
