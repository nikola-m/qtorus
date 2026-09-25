"""Automated correctness tests for the shared numerical core.

Run with:  pytest -q
These encode the four independent verifications:
  1. analytic disk Fourier coefficient == brute-force FFT (sharp disk);
  2. plane-wave solver reproduces the exact free-torus spectrum;
  3. plane-wave spectrum == independent finite-difference spectrum (smooth disk);
  4. split-operator propagator conserves norm (unitarity).
Plus a symmetry-partition sanity check.
"""
import numpy as np
from qtorus import (TorusGeometry, CircularPotential, PlaneWaveHamiltonian,
                    solve_spectrum, finite_difference_spectrum,
                    SplitOperatorPropagator, gaussian_wavepacket,
                    sector_hamiltonian, SECTORS)


def test_disk_fourier_matches_fft():
    geom = TorusGeometry(1.0, 1.0, 2048)
    pot = CircularPotential(geom, R=0.23, V0=1.0, center=(0.31, 0.42),
                            profile="sharp")
    X, Y = geom.real_meshgrid()
    Vr = pot.sample(X, Y)
    Vk = np.fft.fft2(Vr) / (geom.ngrid**2)
    err = 0.0
    for dm in range(-10, 11):
        for dn in range(-10, 11):
            qx, qy = geom.g_of(np.array([dm]), np.array([dn]))
            ana = pot._analytic_coeff(qx, qy)[0]
            num = Vk[dm % geom.ngrid, dn % geom.ngrid]
            err = max(err, abs(ana - num))
    assert err < 1e-3            # limited by sharp-edge sampling on 2048 grid


def test_free_torus_exact():
    geom = TorusGeometry(1.0, 1.0, 64)
    pot = CircularPotential(geom, R=0.2, V0=0.0, center=(0.0, 0.0),
                            profile="smooth", w=0.03)
    H = PlaneWaveHamiltonian(geom, pot, mmax=6)
    w, _ = solve_spectrum(H, method="dense")
    ms = np.arange(-6, 7)
    MX, MY = np.meshgrid(ms, ms, indexing="ij")
    exact = np.sort(((2 * np.pi)**2 * (MX**2 + MY**2)).ravel())
    assert np.max(np.abs(np.sort(w) - exact)) < 1e-9


def test_planewave_vs_finite_difference():
    geom = TorusGeometry(1.0, 1.0, 256)
    pot = CircularPotential(geom, R=0.25, V0=40.0, center=(0.5, 0.5),
                            profile="smooth", w=0.03)
    H = PlaneWaveHamiltonian(geom, pot, mmax=16)
    w_pw, _ = solve_spectrum(H, method="dense")
    w_pw = np.sort(w_pw)[:12]
    w_fd = finite_difference_spectrum(geom, pot, ng=300, k=12)
    rel = np.max(np.abs(w_pw - w_fd) / np.abs(w_pw))
    assert rel < 5e-3            # limited by 2nd-order FD on 300^2 grid


def test_propagator_unitary():
    geom = TorusGeometry(1.0, 1.0, 128)
    pot = CircularPotential(geom, R=0.2, V0=300.0, center=(0.5, 0.5),
                            profile="smooth", w=0.03)
    prop = SplitOperatorPropagator(geom, pot, dt=1e-4)
    psi = gaussian_wavepacket(geom, 0.25, 0.5, 0.06, 60.0, 0.0)
    n0 = prop.norm(psi)
    psi, *_ = prop.evolve(psi, 200)
    assert abs(prop.norm(psi) - n0) < 1e-9


def test_symmetry_partition():
    # the four C2v sectors together must contain (2 mmax+1)^2 states
    geom = TorusGeometry(1.0, 1.6180339887, 64)
    pot = CircularPotential(geom, R=0.25, V0=50.0, center=(0.0, 0.0),
                            profile="smooth", w=0.03)
    mmax = 6
    total = 0
    for sec in SECTORS:
        Hs, B = sector_hamiltonian(geom, pot, mmax, sec)
        total += Hs.shape[0]
    assert total == (2 * mmax + 1)**2


def test_sector_block_lean_matches_reference():
    from qtorus import sector_block_lean
    from scipy.linalg import eigh
    geom = TorusGeometry(1.0, 1.6180339887, 96)
    pot = CircularPotential(geom, R=0.25, V0=600.0, center=(0.0, 0.0),
                            profile="smooth", w=0.02)
    for sec in SECTORS:
        Href, _ = sector_hamiltonian(geom, pot, 10, sec)
        Hlean = sector_block_lean(geom, pot, 10, sec)
        assert Href.shape == Hlean.shape
        dE = np.max(np.abs(np.sort(eigh(Href, eigvals_only=True))
                           - np.sort(eigh(Hlean, eigvals_only=True))))
        assert dE < 1e-8
