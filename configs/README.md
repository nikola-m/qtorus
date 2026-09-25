# Workstation presets (6-core CPU, 32 GB RAM)

Three ready-to-load `SimulationConfig` presets for offline production runs.
Load with `SimulationConfig.from_json(path)`. Set threads first:

```bash
export OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6
```

Timings are extrapolated from single-thread benchmarks (see the paper's
computational appendix) assuming threaded LAPACK (~3.5x) and threaded FFT (~4x).
The smooth-potential Hamiltonian is dense-occupancy, so plan for dense scaling;
the levers are the four C2v blocks and eigenvalues-only solves via the
memory-lean builder (`sector_block_lean` / `sector_spectrum`).

## A -- Larger spectra (`A_larger_spectra.json`)
Push the cutoff to reach high energies with a long, accurate level list from one
sector. `mmax=90`, `sector=A1`, `lean=True`, `eigenvectors=False`, keep the
lowest `n_use=4300` (~0.52 x sector dim; verify against `mmax=94`).
Cost: ~0.6 GB, ~40 s (eigenvalues-only, threaded).

```python
from qtorus import SimulationConfig, TorusGeometry, CircularPotential, sector_spectrum
c = SimulationConfig.from_json("configs/A_larger_spectra.json")
g = TorusGeometry(c.Lx, c.Ly, c.ngrid)
p = CircularPotential(g, c.R, c.V0, tuple(c.center), c.profile, c.w)
E = sector_spectrum(g, p, c.mmax, c.sector, n_use=c.n_use, lean=c.lean)
```

## B -- Better statistics (`B_better_statistics.json`)
Pool all four C2v sectors and resolve the chaos indicator in energy bands.
`mmax=70`, `sectors=(A1,B1,B2,A2)`, `n_use=2600` per sector (~10,000 pooled
levels), `bands=((0,4000),(4000,9000))`.
Cost: ~1-2 GB, ~40 s diagonalisation (threaded).

```python
from qtorus import (SimulationConfig, TorusGeometry, CircularPotential,
                    sector_spectrum, pooled_nnsd, banded_statistics, brody_fit)
c = SimulationConfig.from_json("configs/B_better_statistics.json")
g = TorusGeometry(c.Lx, c.Ly, c.ngrid)
p = CircularPotential(g, c.R, c.V0, tuple(c.center), c.profile, c.w)
Es = [sector_spectrum(g, p, c.mmax, s, n_use=c.n_use, lean=True) for s in c.sectors]
beta = brody_fit(pooled_nnsd(Es, deg=c.unfold_deg, trim=c.unfold_trim))
bands = banded_statistics(Es, [tuple(b) for b in c.bands],
                          deg=c.unfold_deg, trim=c.unfold_trim)
```

## C -- Finer dynamics (`C_finer_dynamics.json`)
Long-time, high-resolution wave-packet tunnelling. `ngrid=1024` (or 2048),
`dt=2e-5` (accuracy target `dt*E ~ 0.05`), `nsteps=10000`, thinner wall
(`R=0.08`, `w=0.01`) for visible through-tunnelling. Use a threaded FFT backend
(`scipy.fft.fft2/ifft2(..., workers=6)`).
Cost: <1 GB; ~100 s at 1024^2, ~12 min at 2048^2 for 1e4 steps.

Always confirm the converged window by comparing `mmax` with `mmax+4` and
keeping only levels that agree to < 1e-3.
