"""Wave-packet dynamics: scattering off, and tunnelling into, the obstacle.

A Gaussian packet of mean energy E is launched at the disk.
  * E > V0 : classically allowed -- the packet enters/refracts (over-barrier).
  * E < V0 : classically forbidden -- the packet is largely reflected but its
    density penetrates the disk (evanescent tunnelling).
We record the probability inside the disk and quantify the crossover of the
peak in-disk probability as V0 sweeps through the packet energy.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from qtorus import (TorusGeometry, CircularPotential,
                    SplitOperatorPropagator, gaussian_wavepacket)
from qtorus.viz import set_style, plot_density

set_style()
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "paper", "figures")
DATA = os.path.join(HERE, "..", "data")
os.makedirs(OUT, exist_ok=True); os.makedirs(DATA, exist_ok=True)

geom = TorusGeometry(1.0, 1.0, 256)
R, wsoft = 0.12, 0.015
x0, y0, sigma, k0 = 0.22, 0.5, 0.06, 50.0
E_mean = k0**2 + 1.0 / (2 * sigma**2)
dt, nsteps, rec = 8e-5, 130, 6
X, Y = geom.real_meshgrid()
disk_mask = (np.hypot(X - 0.5, Y - 0.5) <= R)
print(f"packet mean energy E ~ {E_mean:.0f}")

def run(V0, keep_frames=False):
    pot = CircularPotential(geom, R=R, V0=V0, center=(0.5, 0.5),
                            profile="smooth", w=wsoft)
    prop = SplitOperatorPropagator(geom, pot, dt)
    psi = gaussian_wavepacket(geom, x0, y0, sigma, k0, 0.0)
    frames, tin, times = [], [], []
    p = psi.copy()
    for s in range(nsteps + 1):
        if s > 0:
            p = prop.step(p)
        if s % rec == 0:
            times.append(s * dt)
            tin.append(prop.region_probability(p, disk_mask))
            if keep_frames:
                frames.append(np.abs(p)**2)
    return np.array(times), np.array(tin), frames, prop.norm(p)

# ---- two representative cases ----------------------------------------
t_over, pin_over, _, n1 = run(1200.0)          # E > V0
t_tun, pin_tun, frames_tun, n2 = run(5000.0, keep_frames=True)  # E < V0
print(f"norm conservation: {n1:.6f}, {n2:.6f}")

# ---- crossover: peak in-disk probability vs V0 -----------------------
V0_sweep = np.array([300, 700, 1200, 1800, 2400, 3000, 3800, 4600, 5600, 7000], float)
peak_in = np.array([run(v)[1].max() for v in V0_sweep])

# ---- Figure: snapshot montage (tunnelling case) ----------------------
sel = [0, 6, 11, 16, 21]
fig, axes = plt.subplots(1, len(sel), figsize=(13, 2.9))
for ax, i in zip(axes, sel):
    plot_density(ax, geom, frames_tun[i], title=fr"$t={t_tun[i]*1000:.1f}\times10^{{-3}}$")
    ax.add_patch(plt.Circle((0.5, 0.5), R, fill=False, ec="cyan", lw=0.8, ls="--"))
    ax.set_xlabel(""); ax.set_ylabel("")
fig.suptitle(fr"Tunnelling case $E\approx{E_mean:.0f} < V_0=5000$", y=1.02)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig_dynamics_montage.pdf"))
fig.savefig(os.path.join(OUT, "fig_dynamics_montage.png"), dpi=200)

# ---- Figure: in-disk probability + crossover -------------------------
fig2, (a, b) = plt.subplots(1, 2, figsize=(8.6, 3.3))
a.plot(t_over * 1e3, pin_over, "-", color="#c0392b", label=r"$E>V_0$ (over-barrier)")
a.plot(t_tun * 1e3, pin_tun, "-", color="#2471a3", label=r"$E<V_0$ (tunnelling)")
a.set_xlabel(r"time $t\ (\times10^{-3})$"); a.set_ylabel("probability inside disk")
a.legend(fontsize=8); a.set_title("In-disk probability vs time")
b.axvline(E_mean, ls=":", color="k", lw=1)
b.text(E_mean * 1.02, peak_in.max() * 0.9, r"$V_0=E$", fontsize=8)
b.semilogy(V0_sweep, peak_in, "o-", color="#8e44ad")
b.set_xlabel(r"barrier height $V_0$")
b.set_ylabel("peak in-disk probability")
b.set_title("Classical-to-tunnelling crossover")
fig2.tight_layout()
fig2.savefig(os.path.join(OUT, "fig_dynamics_crossover.pdf"))
fig2.savefig(os.path.join(OUT, "fig_dynamics_crossover.png"), dpi=200)

np.savez(os.path.join(DATA, "dynamics.npz"), t_over=t_over, pin_over=pin_over,
         t_tun=t_tun, pin_tun=pin_tun, V0_sweep=V0_sweep, peak_in=peak_in,
         E_mean=E_mean)

# ---- Animation (gif) of the tunnelling case --------------------------
figA, axA = plt.subplots(figsize=(3.6, 3.6))
vmax = max(f.max() for f in frames_tun)
imA = axA.imshow(frames_tun[0].T, origin="lower", extent=[0, 1, 0, 1],
                 cmap="magma", vmax=vmax * 0.6)
axA.add_patch(plt.Circle((0.5, 0.5), R, fill=False, ec="cyan", lw=0.9, ls="--"))
axA.set_xticks([]); axA.set_yticks([])
def upd(i):
    imA.set_data(frames_tun[i].T); axA.set_title(fr"$t={t_tun[i]*1e3:.1f}\times10^{{-3}}$")
    return imA,
anim = FuncAnimation(figA, upd, frames=len(frames_tun), interval=120, blit=False)
anim.save(os.path.join(OUT, "anim_tunnelling.gif"), writer=PillowWriter(fps=8), dpi=90)
print("saved dynamics figures + animation; peak_in:", np.round(peak_in, 4))
