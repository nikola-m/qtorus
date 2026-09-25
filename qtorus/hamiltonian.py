"""Plane-wave (Fourier spectral) Hamiltonian on the torus.

Basis:  phi_{m,n}(r) = A^{-1/2} exp(i G_{m,n} . r),  |m|,|n| <= mmax.
Matrix: H_{G,G'} = |G|^2 delta_{G,G'} + Vtilde(G - G').

The kinetic term is diagonal and exact.  The potential term is a Toeplitz-like
convolution built from the analytic (sharp) or FFT-derived (smooth) Fourier
coefficients returned by CircularPotential.coeff_grid.

Cost: basis size Nb = (2*mmax+1)^2; dense Hermitian storage O(Nb^2),
dense diagonalisation O(Nb^3).  For large mmax use the sparse builder together
with a shift-invert eigensolver (solvers.solve_spectrum(..., method="sparse")).
"""
from __future__ import annotations
import numpy as np
import scipy.sparse as sp
from .geometry import TorusGeometry
from .potential import CircularPotential


class PlaneWaveHamiltonian:
    def __init__(self, geom: TorusGeometry, pot: CircularPotential, mmax: int):
        self.geom = geom
        self.pot = pot
        self.mmax = int(mmax)
        ms = np.arange(-mmax, mmax + 1)
        MX, MY = np.meshgrid(ms, ms, indexing="ij")
        self.m = MX.ravel()
        self.n = MY.ravel()
        gx, gy = geom.g_of(self.m, self.n)
        self.gx, self.gy = gx, gy
        self.kinetic = gx**2 + gy**2
        self.nbasis = self.m.size
        self._Vhat = pot.coeff_grid(mmax)   # (4mmax+1, 4mmax+1)

    # ------------------------------------------------------------------
    def dense(self) -> np.ndarray:
        """Assemble the dense Hermitian Hamiltonian (complex128)."""
        dmax = 2 * self.mmax
        # difference indices between every pair of basis states
        DM = self.m[:, None] - self.m[None, :] + dmax
        DN = self.n[:, None] - self.n[None, :] + dmax
        H = self._Vhat[DM, DN].astype(np.complex128, copy=True)
        H[np.diag_indices(self.nbasis)] += self.kinetic
        H = 0.5 * (H + H.conj().T)          # enforce Hermiticity vs round-off
        return H

    def sparse(self, drop_tol: float = 1e-12) -> sp.csr_matrix:
        """Assemble a sparse Hamiltonian, dropping |V| < drop_tol couplings.

        Useful for smooth potentials (rapidly decaying Fourier coefficients),
        where most off-diagonal couplings are negligible.
        """
        dmax = 2 * self.mmax
        DM = self.m[:, None] - self.m[None, :] + dmax
        DN = self.n[:, None] - self.n[None, :] + dmax
        V = self._Vhat[DM, DN]
        mask = np.abs(V) >= drop_tol
        np.fill_diagonal(mask, True)
        rows, cols = np.nonzero(mask)
        vals = V[rows, cols]
        vals[rows == cols] += self.kinetic[rows[rows == cols]]
        H = sp.csr_matrix((vals, (rows, cols)),
                          shape=(self.nbasis, self.nbasis), dtype=complex)
        H = 0.5 * (H + H.getH())
        return H.tocsr()

    # ------------------------------------------------------------------
    def eigvec_to_real(self, coeff: np.ndarray, ngrid: int | None = None):
        """Map a plane-wave eigenvector (coefficients over the basis) to a
        real-space wavefunction on an ngrid x ngrid mesh via inverse FFT."""
        ng = ngrid or self.geom.ngrid
        spec = np.zeros((ng, ng), dtype=complex)
        spec[self.m % ng, self.n % ng] = coeff
        psi = np.fft.ifft2(spec) * ng * ng / np.sqrt(self.geom.area)
        return psi
