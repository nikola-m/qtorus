"""Unitary time evolution by the Strang split-operator (Fourier) method.

For H = -Laplacian + V (hbar=1), one symmetric step of size dt is

    psi <- e^{-i V dt/2}  F^{-1} e^{-i |k|^2 dt} F  e^{-i V dt/2} psi,

where F is the 2-D FFT.  Each factor is unitary, so norm is conserved to
round-off and the scheme is second-order accurate in dt (Strang splitting).
This is the natural propagator on a periodic domain and is ideal for
visualising scattering off, and tunnelling through, the circular obstacle.
"""
from __future__ import annotations
import numpy as np
from .geometry import TorusGeometry
from .potential import CircularPotential


class SplitOperatorPropagator:
    def __init__(self, geom: TorusGeometry, pot: CircularPotential, dt: float):
        self.geom = geom
        self.dt = float(dt)
        self.V = pot.sample()                       # real-space potential grid
        KX, KY = geom.k_meshgrid()
        self.kin_phase = np.exp(-1j * (KX**2 + KY**2) * dt)
        self.half_pot_phase = np.exp(-0.5j * self.V * dt)

    def step(self, psi: np.ndarray) -> np.ndarray:
        psi = self.half_pot_phase * psi
        psi = np.fft.ifft2(self.kin_phase * np.fft.fft2(psi))
        psi = self.half_pot_phase * psi
        return psi

    def evolve(self, psi, nsteps, record_every=0, observers=None):
        """Propagate nsteps.  If record_every>0, call each observer(psi,step,t)
        every record_every steps and collect the returns in a dict of lists."""
        logs = {name: [] for name in (observers or {})}
        times = []
        for s in range(1, nsteps + 1):
            psi = self.step(psi)
            if record_every and (s % record_every == 0):
                t = s * self.dt
                times.append(t)
                for name, fn in (observers or {}).items():
                    logs[name].append(fn(psi, s, t))
        return psi, times, logs

    # ---- observables --------------------------------------------------
    def norm(self, psi):
        return np.sqrt(np.sum(np.abs(psi)**2) * self.geom.dx * self.geom.dy)

    def region_probability(self, psi, mask):
        return float(np.sum(np.abs(psi[mask])**2) * self.geom.dx * self.geom.dy)


def gaussian_wavepacket(geom: TorusGeometry, x0, y0, sigma, kx0, ky0):
    """Minimum-uncertainty Gaussian packet, L2-normalised on the torus.

    <p> = (kx0, ky0);  spatial width sigma.  Mean kinetic energy
    <-Laplacian> = kx0^2 + ky0^2 + 1/(2 sigma^2).
    """
    X, Y = geom.real_meshgrid()
    dx, dy = geom.min_image_delta(X, Y, (x0, y0))
    env = np.exp(-(dx**2 + dy**2) / (4 * sigma**2))
    psi = env * np.exp(1j * (kx0 * X + ky0 * Y))
    nrm = np.sqrt(np.sum(np.abs(psi)**2) * geom.dx * geom.dy)
    return psi / nrm
