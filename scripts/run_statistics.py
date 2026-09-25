"""Spectral-statistics sweep over barrier height V0 (Poisson -> GOE).

For each V0 we diagonalise the A1 symmetry sector of the smoothed penetrable
Sinai billiard, unfold the lowest converged levels, and compute the NNSD,
Brody parameter, and number variance.  As V0 grows from 0 (free, integrable)
towards the hard-wall limit, statistics cross from Poisson to GOE.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh
from qtorus import (TorusGeometry, CircularPotential, sector_hamiltonian,
                    unfold_spectrum, nnsd, sigma2, brody_fit)
from qtorus.statistics import sigma2_goe
from qtorus.viz import set_style, plot_nnsd
from qtorus.config import SimulationConfig
from qtorus.provenance import RunProvenance

set_style()
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "paper", "figures")
DATA = os.path.join(HERE, "..", "data")
os.makedirs(OUT, exist_ok=True); os.makedirs(DATA, exist_ok=True)

MMAX = 30
N_USE = 320          # converged low-energy window (see fig_convergence / probes)
V0_LIST = [0.0, 30.0, 100.0, 300.0, 1000.0, 3000.0]
geom = TorusGeometry(1.0, 1.6180339887498949, 260)

cfg = SimulationConfig(Lx=geom.Lx, Ly=geom.Ly, ngrid=geom.ngrid, R=0.25,
                       V0=-1, profile="smooth", w=0.02, mmax=MMAX, sector="A1",
                       label="statistics_sweep")
RunProvenance.capture(cfg.hash(), cfg.seed).to_json(
    os.path.join(DATA, "provenance_statistics.json"))

results = {}
for V0 in V0_LIST:
    pot = CircularPotential(geom, R=0.25, V0=V0, center=(0.0, 0.0),
                            profile="smooth", w=0.02)
    Hs, _ = sector_hamiltonian(geom, pot, MMAX, "A1")
    E = np.sort(eigh(Hs, eigvals_only=True))[:N_USE]
    x = unfold_spectrum(E, deg=12, trim=10)
    s = nnsd(x)
    beta = brody_fit(s)
    results[V0] = dict(E=E, unfolded=x, spacings=s, beta=beta)
    print(f"V0={V0:7.1f}  Brody beta = {beta:.3f}  (Emax={E[-1]:.0f})")

# ---- Figure 1: NNSD transition panels --------------------------------
fig, axes = plt.subplots(2, 3, figsize=(9.2, 5.4))
for ax, V0 in zip(axes.ravel(), V0_LIST):
    r = results[V0]
    plot_nnsd(ax, r["spacings"], beta=r["beta"])
    ax.set_title(fr"$V_0={V0:.0f}$,  $\beta={r['beta']:.2f}$", fontsize=9)
    if V0 != V0_LIST[0]:
        ax.set_ylabel("")
handles, labels = axes[0, 0].get_legend_handles_labels()
for ax in axes.ravel():
    lg = ax.get_legend()
    if lg: lg.remove()
fig.legend(handles, labels, loc="upper center", ncol=4, fontsize=8,
           bbox_to_anchor=(0.5, 1.02))
fig.tight_layout(rect=[0, 0, 1, 0.97])
fig.savefig(os.path.join(OUT, "fig_pofs_transition.pdf"))
fig.savefig(os.path.join(OUT, "fig_pofs_transition.png"), dpi=200)

# ---- Figure 2: Brody parameter vs V0 ---------------------------------
fig2, ax2 = plt.subplots(figsize=(4.4, 3.4))
betas = [results[v]["beta"] for v in V0_LIST]
ax2.semilogx([max(v, 1) for v in V0_LIST], betas, "o-", color="#8e44ad")
ax2.axhline(0, ls="--", color="b", lw=1); ax2.axhline(1, ls="-", color="r", lw=1)
ax2.text(1.2, 0.04, "Poisson", color="b", fontsize=8)
ax2.text(1.2, 0.92, "GOE", color="r", fontsize=8)
ax2.set_xlabel(r"barrier height $V_0$")
ax2.set_ylabel(r"Brody parameter $\beta$")
ax2.set_title("Integrable-to-chaotic transition")
ax2.set_ylim(-0.05, 1.1)
fig2.tight_layout()
fig2.savefig(os.path.join(OUT, "fig_brody_vs_V0.pdf"))
fig2.savefig(os.path.join(OUT, "fig_brody_vs_V0.png"), dpi=200)

# ---- Figure 3: number variance for extremes --------------------------
fig3, ax3 = plt.subplots(figsize=(4.4, 3.4))
Lv = np.linspace(0.2, 6, 25)
for V0, col in [(0.0, "#2471a3"), (3000.0, "#c0392b")]:
    s2 = sigma2(results[V0]["unfolded"], Lv)
    ax3.plot(Lv, s2, "o-", color=col, ms=3, label=fr"$V_0={V0:.0f}$")
ax3.plot(Lv, Lv, "b--", lw=1, label="Poisson")
ax3.plot(Lv, sigma2_goe(Lv), "r-", lw=1, label="GOE")
ax3.set_xlabel("$L$"); ax3.set_ylabel(r"$\Sigma^2(L)$")
ax3.set_title("Number variance"); ax3.legend(fontsize=8)
fig3.tight_layout()
fig3.savefig(os.path.join(OUT, "fig_sigma2.pdf"))
fig3.savefig(os.path.join(OUT, "fig_sigma2.png"), dpi=200)

np.savez(os.path.join(DATA, "statistics.npz"),
         V0=np.array(V0_LIST), betas=np.array(betas),
         **{f"spacings_{i}": results[v]["spacings"] for i, v in enumerate(V0_LIST)})
print("saved statistics figures + data; betas:", np.round(betas, 3))
