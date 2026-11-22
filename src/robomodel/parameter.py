"""A model parameter: value, uncertainty and where it came from.

Every number robomodel reports is a Parameter, so the report can say which values are
trustworthy and which need calibration.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np


class Source(Enum):
    """Where a value came from, most trustworthy first."""

    MEASURED = "measured"  # weighed, measured on the real vehicle
    CALIBRATED = "calibrated"  # fitted to test data
    ANALYTIC = "analytic"  # exact formula for the given geometry (e.g. Lamb for ellipsoids)
    CAD = "cad"  # computed from the geometry file / primitive shape
    BEM = "bem"  # boundary element solver (Capytaine)
    EMPIRICAL = "empirical"  # handbook formula, rough
    ASSUMED = "assumed"  # an assumption, e.g. uniform mass distribution


@dataclass(frozen=True)
class Parameter:
    """A scalar or array value with a one-standard-deviation uncertainty.

    `sigma` has the same shape as `value` and is in the same units. A sigma of zero means
    exact for the stated model (not necessarily for the real vehicle).
    """

    value: float | np.ndarray
    sigma: float | np.ndarray
    source: Source
    unit: str = ""
    note: str = ""

    def __post_init__(self):
        if np.any(np.asarray(self.sigma) < 0):
            raise ValueError("sigma must not be negative")
        if np.shape(self.value) != np.shape(self.sigma):
            raise ValueError(
                f"value and sigma shapes differ: {np.shape(self.value)} vs {np.shape(self.sigma)}"
            )

    @classmethod
    def relative(
        cls,
        value: float | np.ndarray,
        rel_sigma: float,
        source: Source,
        unit: str = "",
        note: str = "",
    ) -> Parameter:
        """Build a parameter whose uncertainty is a fraction of its value."""
        sigma = np.abs(np.asarray(value, dtype=float)) * rel_sigma
        if np.ndim(value) == 0:
            value, sigma = float(value), float(sigma)
        return cls(value, sigma, source, unit, note)

    def format(self, precision: int = 4) -> str:
        """Short text such as '4.29 ± 0.0429 kg [measured]' (scalars only)."""
        if np.ndim(self.value) != 0:
            return f"array{np.shape(self.value)} [{self.source.value}]"
        unit = f" {self.unit}" if self.unit else ""
        return f"{self.value:.{precision}g} ± {self.sigma:.2g}{unit} [{self.source.value}]"
