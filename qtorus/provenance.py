"""Capture the environment and configuration for reproducible runs."""
from __future__ import annotations
from dataclasses import dataclass, asdict
import platform
import sys
import json
import datetime


@dataclass
class RunProvenance:
    timestamp_utc: str
    python: str
    platform: str
    packages: dict
    config_hash: str
    seed: int

    @classmethod
    def capture(cls, config_hash: str, seed: int):
        import numpy, scipy, matplotlib
        return cls(
            timestamp_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            python=sys.version.split()[0],
            platform=platform.platform(),
            packages={"numpy": numpy.__version__,
                      "scipy": scipy.__version__,
                      "matplotlib": matplotlib.__version__},
            config_hash=config_hash,
            seed=seed,
        )

    def to_json(self, path):
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)
