"""Eigensolvers and an independent finite-difference cross-validator.

Two routes to eigenpairs of the plane-wave Hamiltonian:

* ``method="dense"``  -- LAPACK ``eigh``; returns *all* levels. Best when the
  basis size Nb=(2 mmax+1)^2 is at most a few thousand.
* ``method="sparse"`` -- ``scipy.sparse.linalg.eigsh`` with shift-invert around
  ``sigma``; returns ``k`` interior levels near a target energy. Scales to large
  bases on HPC (the assembly is sparse for smooth potentials).

``finite_difference_spectrum`` solves the *same* physical problem on a
real-space grid with a periodic 5-point Laplacian -- a numerically independent
method used only to cross-check the spectral solver (see tests/test_core.py).
"""
from __future__ import annotations
import numpy as np
import scipy.sparse as sp
from scipy.linalg import eigh
from scipy.sparse.linalg import eigsh
from .geometry import TorusGeometry
from .potential import CircularPotential
from .hamiltonian import PlaneWaveHamiltonian


def solve_spectrum(H: PlaneWaveHamiltonian, method: str = "dense",
                   k: int | None = None, sigma: float | None = None,
                   eigenvectors: bool = True):
    """Return (energies, vectors) sorted by energy.

    Parameters
    ----------
    method : {"dense","sparse"}
    k      : number of levels (sparse only)
    sigma  : shift-invert target energy (sparse only)
    """
    if method == "dense":
        M = H.dense()
        if eigenvectors:
            w, v = eigh(M)
            return w.real, v
        return eigh(M, eigvals_only=True).real, None
    elif method == "sparse":
        if k is None:
            raise ValueError("sparse method needs k (number of levels)")
        M = H.sparse()
        w, v = eigsh(M, k=k, sigma=sigma, which="LM" if sigma is None else "LM")
        order = np.argsort(w.real)
        w = w.real[order]; v = v[:, order]
        return (w, v) if eigenvectors else (w, None)
    raise ValueError("method must be 'dense' or 'sparse'")


def finite_difference_spectrum(geom: TorusGeometry, pot: CircularPotential,
                               ng: int, k: int = 20):
    """Independent 2-D periodic finite-difference solver (cross-check only).

    Returns the k lowest eigenvalues.  Second-order accurate (error O(h^2));
    used to confirm the spectral solver, not for production statistics.
    """
    h = geom.Lx / ng
    if abs(geom.Lx - geom.Ly) > 1e-12:
        raise ValueError("finite-difference cross-check assumes a square torus")
    X, Y = TorusGeometry(geom.Lx, geom.Ly, ng).real_meshgrid()
    Vg = pot.sample(X, Y).ravel()
    main = -2.0 * np.ones(ng)
    off = np.ones(ng - 1)
    D1 = sp.diags([main, off, off, [1.0], [1.0]],
                  [0, 1, -1, ng - 1, -(ng - 1)]) / h**2
    I = sp.identity(ng)
    lap = sp.kron(D1, I) + sp.kron(I, D1)
    Hfd = (-lap + sp.diags(Vg)).tocsc()
    w = eigsh(Hfd, k=k, which="SA", return_eigenvectors=False)
    return np.sort(w.real)
