"""Husimi (coherent-state) phase-space representation of an eigenstate.

For a state psi on the torus and a coherent state g_{r0,k0} that is a Gaussian
of width sigma centred at r0 with mean momentum k0, the Husimi slice at fixed
momentum k0 is

    H_{k0}(r0) = | < g_{r0,k0} | psi > |^2
               = | [ (psi * e^{-i k0 . r}) (convolved with) Gaussian_sigma ](r0) |^2 .

The convolution is evaluated efficiently with the FFT.  Plotting H_{k0}(r0)
for the dominant |k0| = sqrt(E) reveals where, and moving in which direction,
the eigenstate's probability concentrates (scarring, tunnelling into the disk).
"""
from __future__ import annotations
import numpy as np
from .geometry import TorusGeometry


def husimi(geom: TorusGeometry, psi: np.ndarray, k0, sigma: float):
    kx0, ky0 = k0
    X, Y = geom.real_meshgrid()
    # demodulate to the target momentum, then low-pass with a Gaussian window
    field = psi * np.exp(-1j * (kx0 * X + ky0 * Y))
    KX, KY = geom.k_meshgrid()
    gwin_k = np.exp(-0.5 * sigma**2 * (KX**2 + KY**2))     # FT of Gaussian window
    smoothed = np.fft.ifft2(np.fft.fft2(field) * gwin_k)
    H = np.abs(smoothed)**2
    return H / H.max()
