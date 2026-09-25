"""Symmetry reduction for clean level statistics.

A *single* radially symmetric obstacle on a rectangular torus cannot break the
reflection symmetries through its own centre: the residual point group is C2v
(identity, x-reflection, y-reflection, 180-deg rotation).  Mixing several
symmetry classes destroys the universal Random-Matrix statistics, so spectral
statistics MUST be computed within a single irreducible representation
(the standard desymmetrisation used for the Sinai billiard).

We place the disk at the origin (centre=(0,0)); by torus translation invariance
this is physically identical to a centred disk but makes the Fourier
coefficients real and even, so the reflection operators act on plane waves as
simple sign flips of the harmonic indices:

    R_x : (m,n) -> (-m, n),     R_y : (m,n) -> (m, -n).

The four one-dimensional irreps are labelled by parities (sx, sy) = (+/-1,+/-1):

    A1 = (+,+)  cos*cos      B1 = (-,+)  sin*cos
    B2 = (+,-)  cos*sin      A2 = (-,-)  sin*sin

For each irrep we build symmetry-adapted orthonormal combinations of plane
waves and project:  H_sector = B^dagger H B  (real symmetric).
"""
from __future__ import annotations
import numpy as np
from .geometry import TorusGeometry
from .potential import CircularPotential
from .hamiltonian import PlaneWaveHamiltonian

SECTORS = {"A1": (+1, +1), "B1": (-1, +1), "B2": (+1, -1), "A2": (-1, -1)}


def _index_map(H: PlaneWaveHamiltonian):
    return {(int(m), int(n)): i for i, (m, n) in enumerate(zip(H.m, H.n))}


def _axis_coeffs(k: int, parity: int):
    """Plane-wave coefficients of cos(g_k .) (parity +1) or sin (parity -1)."""
    if parity > 0:
        return {0: 1.0} if k == 0 else {k: 1.0, -k: 1.0}          # ~cos
    else:
        return {k: 1.0, -k: -1.0}                                  # ~sin (up to 2i)


def build_sector_basis(H: PlaneWaveHamiltonian, sx: int, sy: int) -> np.ndarray:
    """Return an (Nb x d) matrix whose orthonormal columns span the (sx,sy)
    irrep, expressed in H's plane-wave basis."""
    idx = _index_map(H)
    M = H.mmax
    m0 = 0 if sx > 0 else 1
    n0 = 0 if sy > 0 else 1
    cols = []
    for m in range(m0, M + 1):
        for n in range(n0, M + 1):
            vec = np.zeros(H.nbasis, dtype=complex)
            for mp, cx in _axis_coeffs(m, sx).items():
                for npp, cy in _axis_coeffs(n, sy).items():
                    vec[idx[(mp, npp)]] = cx * cy
            vec /= np.linalg.norm(vec)
            cols.append(vec)
    return np.column_stack(cols)


def sector_hamiltonian(geom: TorusGeometry, pot: CircularPotential,
                       mmax: int, sector: str = "A1"):
    """Return the real-symmetric Hamiltonian block for one C2v irrep.

    Requires pot.center == (0, 0) so that the reflection operators are exact
    sign flips.  Returns (H_block, projector_B) where H_block has shape (d, d).
    """
    if tuple(pot.center) != (0.0, 0.0):
        raise ValueError(
            "sector_hamiltonian requires the disk at the origin, center=(0,0); "
            "use torus translation invariance (physically identical to centred).")
    if sector not in SECTORS:
        raise ValueError(f"sector must be one of {list(SECTORS)}")
    sx, sy = SECTORS[sector]
    H = PlaneWaveHamiltonian(geom, pot, mmax)
    Hd = H.dense()
    B = build_sector_basis(H, sx, sy)
    Hs = (B.conj().T @ (Hd @ B))
    Hs = Hs.real
    Hs = 0.5 * (Hs + Hs.T)
    return Hs, B


def sector_block_lean(geom: TorusGeometry, pot: CircularPotential,
                      mmax: int, sector: str = "A1") -> np.ndarray:
    """Memory-lean assembly of one C2v sector block.

    Identical result to ``sector_hamiltonian`` (agreement ~1e-11) but it never
    forms the full dense Hamiltonian.  Each symmetry-adapted basis function is a
    combination of at most four plane waves, so H is applied column-by-column as
    a diagonal kinetic term plus a few shifted slices of the potential's Fourier
    grid; only the (d x d) block is stored.  Peak memory is O(d^2) instead of
    O(Nbasis^2), which is what makes large cutoffs (mmax ~ 90-110) tractable on a
    workstation.  Returns the real-symmetric block (diagonalise with
    ``scipy.linalg.eigh(Hs, eigvals_only=True)`` or a subset driver).

    See also the computational appendix of the accompanying paper.
    """
    import scipy.sparse as sp
    if tuple(pot.center) != (0.0, 0.0):
        raise ValueError("sector_block_lean requires center=(0,0).")
    if sector not in SECTORS:
        raise ValueError(f"sector must be one of {list(SECTORS)}")
    sx, sy = SECTORS[sector]
    H = PlaneWaveHamiltonian(geom, pot, mmax)          # builds only the Vhat grid
    idx = _index_map(H)
    m0 = 0 if sx > 0 else 1
    n0 = 0 if sy > 0 else 1
    colinfo = []
    for mm in range(m0, mmax + 1):
        for nn in range(n0, mmax + 1):
            ii, vv = [], []
            for mp, cx in _axis_coeffs(mm, sx).items():
                for npp, cy in _axis_coeffs(nn, sy).items():
                    ii.append(idx[(mp, npp)]); vv.append(cx * cy)
            vv = np.array(vv, dtype=complex); vv /= np.linalg.norm(vv)
            colinfo.append((np.array(ii), vv))
    d = len(colinfo)
    rows = np.concatenate([ii for ii, _ in colinfo])
    cols = np.concatenate([np.full(len(ii), j) for j, (ii, _) in enumerate(colinfo)])
    data = np.concatenate([vv for _, vv in colinfo])
    BspH = sp.csr_matrix((data, (rows, cols)),
                         shape=(H.nbasis, d), dtype=complex).conj().T.tocsr()
    Vhat = H._Vhat; dmax = 2 * mmax
    m, n, kin = H.m, H.n, H.kinetic
    Hs = np.zeros((d, d))
    for j, (ii, vv) in enumerate(colinfo):
        HBj = np.zeros(H.nbasis, dtype=complex)
        HBj[ii] += kin[ii] * vv
        for a in range(len(ii)):
            HBj += vv[a] * Vhat[m - m[ii[a]] + dmax, n - n[ii[a]] + dmax]
        Hs[:, j] = (BspH @ HBj).real
    return 0.5 * (Hs + Hs.T)


def sector_spectrum(geom: TorusGeometry, pot: CircularPotential, mmax: int,
                    sector: str = "A1", n_use: int = 0, lean: bool = True):
    """Convenience: eigenvalues of one sector, lowest ``n_use`` (0 = all).

    Uses the memory-lean builder by default and an eigenvalues-only LAPACK
    driver, i.e. the recommended path for large-cutoff spectra and statistics.
    """
    from scipy.linalg import eigh
    Hs = sector_block_lean(geom, pot, mmax, sector) if lean \
        else sector_hamiltonian(geom, pot, mmax, sector)[0]
    if n_use and n_use < Hs.shape[0]:
        w = eigh(Hs, eigvals_only=True, subset_by_index=[0, n_use - 1])
    else:
        w = eigh(Hs, eigvals_only=True)
    return np.sort(w.real)
