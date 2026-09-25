"""Serializable simulation configuration with a stable content hash.

Every production run should be driven by a SimulationConfig; the config (and
its hash) is written next to the results so any figure is traceable to the
exact parameters that produced it.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict, field
import hashlib
import json


@dataclass
class SimulationConfig:
    # geometry
    Lx: float = 1.0
    Ly: float = 1.6180339887498949      # golden ratio -> incommensurate sides
    ngrid: int = 256
    # obstacle
    R: float = 0.25
    V0: float = 200.0
    profile: str = "smooth"             # "sharp" | "smooth"
    w: float = 0.03
    center: tuple = (0.0, 0.0)
    # spectral solver
    mmax: int = 24
    method: str = "dense"               # "dense" | "sparse"
    sector: str = "A1"                  # C2v irrep (single-sector runs)
    sectors: tuple = ("A1",)            # sectors to pool for statistics
    lean: bool = True                   # use the memory-lean sector builder
    eigenvectors: bool = True           # False -> eigenvalues-only (leaner)
    n_use: int = 0                      # converged levels to keep (0 = all)
    # statistics
    unfold_deg: int = 12
    unfold_trim: int = 10
    bands: tuple = ()                   # energy bands [(lo,hi),...] for banded stats
    # dynamics
    dt: float = 2.0e-4
    nsteps: int = 4000
    record_every: int = 0
    fft_workers: int = 1               # threads for the split-operator FFTs
    # bookkeeping
    seed: int = 12345
    label: str = "run"

    def to_json(self, path):
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2, default=list)

    @classmethod
    def from_json(cls, path):
        with open(path) as f:
            return cls(**json.load(f))

    def hash(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, default=list)
        return hashlib.sha256(payload.encode()).hexdigest()[:12]
