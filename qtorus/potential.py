"""Circular potential obstacle on the torus (barrier V0>0 or well V0<0).

Two profiles are supported:

* ``profile="sharp"`` -- a top-hat disk  V(r) = V0 * Theta(R - |r - r0|).
  Its Fourier coefficient is *analytic*:

        Vtilde(q) = (V0/A) e^{-i q.r0} * pi R^2 * 2 J1(|q|R)/(|q|R),   q != 0
        Vtilde(0) = V0 * pi R^2 / A .

  This is the 2-D "disk / Airy" transform; verified against a brute-force FFT
  in tests/test_core.py.  A top-hat is discontinuous, so a *plane-wave* basis
  converges only algebraically (Gibbs; Norton & Scheichl, SINUM 2010).

* ``profile="smooth"`` -- a Fermi/tanh-softened disk

        V(r) = V0 * 0.5 * (1 - tanh((|r - r0| - R)/w)),

  with wall width w.  Smoothness restores fast (near-exponential) convergence
  of the plane-wave method and models a physically realistic soft wall.
  Its Fourier coefficients are obtained by one FFT of the sampled profile.

The limit V0 -> +inf of the sharp profile is the hard-wall Sinai billiard.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.special import j1
from .geometry import TorusGeometry


@dataclass(frozen=True)
class CircularPotential:
    geom: TorusGeometry
    R: float = 0.25
    V0: float = 100.0
    center: tuple = (0.5, 0.5)          # in units of (Lx, Ly)? No: absolute coords.
    profile: str = "smooth"             # "sharp" or "smooth"
    w: float = 0.03                     # wall width (only used if profile=="smooth")

    def __post_init__(self):
        if self.profile not in ("sharp", "smooth"):
            raise ValueError("profile must be 'sharp' or 'smooth'")
        if self.profile == "smooth" and self.w <= 0:
            raise ValueError("smooth profile requires wall width w > 0")

    # ---- real space ---------------------------------------------------
    def sample(self, X=None, Y=None) -> np.ndarray:
        """Sample V on the geometry's real grid (or supplied meshgrid)."""
        if X is None or Y is None:
            X, Y = self.geom.real_meshgrid()
        dx, dy = self.geom.min_image_delta(X, Y, self.center)
        r = np.hypot(dx, dy)
        if self.profile == "sharp":
            return self.V0 * (r <= self.R).astype(float)
        return self.V0 * 0.5 * (1.0 - np.tanh((r - self.R) / self.w))

    # ---- Fourier coefficients ----------------------------------------
    def _analytic_coeff(self, qx, qy) -> np.ndarray:
        """Analytic Vtilde(q) for the sharp top-hat disk (any q array)."""
        A = self.geom.area
        q = np.hypot(qx, qy)
        out = np.empty(q.shape, dtype=complex)
        small = q < 1e-12
        out[small] = self.V0 * np.pi * self.R**2 / A
        qn = q[~small]
        form = np.pi * self.R**2 * (2.0 * j1(qn * self.R) / (qn * self.R))
        phase = np.exp(-1j * (qx[~small] * self.center[0]
                              + qy[~small] * self.center[1]))
        out[~small] = (self.V0 / A) * phase * form
        return out

    def coeff_grid(self, mmax: int) -> np.ndarray:
        """Return Vtilde on the harmonic-difference grid needed to build the
        plane-wave Hamiltonian with |m|,|n| <= mmax.

        Output shape (4*mmax+1, 4*mmax+1); element [dm+2*mmax, dn+2*mmax]
        holds Vtilde(G_{dm,dn}) for dm,dn in [-2*mmax, 2*mmax].
        """
        dmax = 2 * mmax
        dm = np.arange(-dmax, dmax + 1)
        DM, DN = np.meshgrid(dm, dm, indexing="ij")
        qx, qy = self.geom.g_of(DM, DN)
        if self.profile == "sharp":
            return self._analytic_coeff(qx, qy)
        # smooth: FFT of a well-resolved real-space sampling
        n_fft = max(self.geom.ngrid, 8 * mmax + 8)
        gx = TorusGeometry(self.geom.Lx, self.geom.Ly, n_fft)
        Vr = CircularPotential(gx, self.R, self.V0, self.center,
                               "smooth", self.w).sample()
        Vk = np.fft.fft2(Vr) / (n_fft * n_fft)          # (1/A) int V e^{-iq.r}
        # gather the required harmonics (periodic index wrap)
        out = np.empty(DM.shape, dtype=complex)
        out[...] = Vk[DM % n_fft, DN % n_fft]
        return out
