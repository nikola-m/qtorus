"""
qtorus -- Quantum billiard on a flat torus with a penetrable circular obstacle.

A production-grade, reproducible framework for the single-particle Schroedinger
problem on a periodic square/rectangular domain (a flat 2-torus) containing a
circular potential barrier/well of *finite* height V0.  Unlike the classical
hard-wall Sinai billiard (recovered here as the limit V0 -> +inf), a finite V0
lets the wavefunction *tunnel* into and through the obstacle.

Units (fixed throughout): hbar = 1, mass m = 1/2, so that

        H = -Laplacian + V(r),      E = |k|^2.

Public API
----------
TorusGeometry            : the periodic domain and its real/reciprocal grids
CircularPotential        : sharp or smooth circular barrier/well, analytic FT
PlaneWaveHamiltonian     : spectral (Fourier) Hamiltonian; dense & sparse
solve_spectrum           : eigenpairs via dense or shift-invert sparse solver
SplitOperatorPropagator  : unitary time evolution (Strang split-operator FFT)
gaussian_wavepacket      : minimum-uncertainty launcher on the torus
sector_hamiltonian       : C2v symmetry-reduced block (clean level statistics)
unfold_spectrum, nnsd, sigma2, delta3, brody_fit  : spectral statistics
husimi                   : phase-space (Husimi) projection of an eigenstate
RunProvenance            : environment / config / seed capture for reproducibility
"""
from .geometry import TorusGeometry
from .potential import CircularPotential
from .hamiltonian import PlaneWaveHamiltonian
from .solvers import solve_spectrum, finite_difference_spectrum
from .dynamics import SplitOperatorPropagator, gaussian_wavepacket
from .symmetry import (sector_hamiltonian, sector_block_lean,
                       sector_spectrum, SECTORS)
from .statistics import (unfold_spectrum, nnsd, sigma2, delta3,
                         brody_fit, poisson_pdf, goe_pdf,
                         pooled_nnsd, banded_statistics)
from .husimi import husimi
from .provenance import RunProvenance
from .config import SimulationConfig

__version__ = "0.1.0"
__all__ = [
    "TorusGeometry", "CircularPotential", "PlaneWaveHamiltonian",
    "solve_spectrum", "finite_difference_spectrum",
    "SplitOperatorPropagator", "gaussian_wavepacket",
    "sector_hamiltonian", "sector_block_lean", "sector_spectrum", "SECTORS",
    "unfold_spectrum", "nnsd", "sigma2", "delta3", "brody_fit",
    "pooled_nnsd", "banded_statistics",
    "poisson_pdf", "goe_pdf", "husimi", "RunProvenance",
    "SimulationConfig", "__version__",
]
