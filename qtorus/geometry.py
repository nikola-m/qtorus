"""Flat-torus geometry: real-space grid and reciprocal (plane-wave) grid.

Conventions
-----------
Domain  Omega = [0, Lx) x [0, Ly)  with periodic boundary conditions.
Reciprocal vectors  G_{m,n} = (2*pi*m/Lx, 2*pi*n/Ly),  m, n integers.
Area    A = Lx * Ly.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class TorusGeometry:
    """A flat rectangular 2-torus.

    Parameters
    ----------
    Lx, Ly : float
        Side lengths of the periodic cell.
    ngrid : int
        Number of real-space samples per side (used for dynamics and for
        FFT-based Fourier coefficients of smooth potentials). Powers of two
        are fastest for the FFT.
    """
    Lx: float = 1.0
    Ly: float = 1.0
    ngrid: int = 256

    # ---- real space ---------------------------------------------------
    @property
    def area(self) -> float:
        return self.Lx * self.Ly

    @property
    def dx(self) -> float:
        return self.Lx / self.ngrid

    @property
    def dy(self) -> float:
        return self.Ly / self.ngrid

    def real_axes(self):
        """1-D coordinate axes (cell corner at origin, size ngrid)."""
        x = np.arange(self.ngrid) * self.dx
        y = np.arange(self.ngrid) * self.dy
        return x, y

    def real_meshgrid(self):
        x, y = self.real_axes()
        X, Y = np.meshgrid(x, y, indexing="ij")
        return X, Y

    def min_image_delta(self, X, Y, r0):
        """Nearest-periodic-image displacement components from r0 (=(x0,y0))."""
        dx = np.abs(X - r0[0]); dx = np.minimum(dx, self.Lx - dx)
        dy = np.abs(Y - r0[1]); dy = np.minimum(dy, self.Ly - dy)
        return dx, dy

    # ---- reciprocal space --------------------------------------------
    def k_axes(self):
        """FFT-ordered angular-wavenumber axes matching numpy.fft.fft2."""
        kx = 2 * np.pi * np.fft.fftfreq(self.ngrid, d=self.dx)
        ky = 2 * np.pi * np.fft.fftfreq(self.ngrid, d=self.dy)
        return kx, ky

    def k_meshgrid(self):
        kx, ky = self.k_axes()
        KX, KY = np.meshgrid(kx, ky, indexing="ij")
        return KX, KY

    def g_of(self, m, n):
        """Reciprocal vector components for integer harmonic indices."""
        return 2 * np.pi * m / self.Lx, 2 * np.pi * n / self.Ly
