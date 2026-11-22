"""Thruster model: command in [-1, 1] -> thrust, and the allocation matrix tau = B f."""

from __future__ import annotations

import numpy as np

from robomodel.vehicle import Thruster

PRESET_REL_SIGMA = 0.15  # datasheet value, battery voltage and wear vary
USER_REL_SIGMA = 0.05  # limits measured by the user


def allocation_matrix(thrusters: tuple[Thruster, ...]) -> np.ndarray:
    """6 x n matrix: column i is [d_i; r_i x d_i] for thruster i (force 1 N)."""
    if not thrusters:
        return np.zeros((6, 0))
    cols = [np.concatenate([t.direction, np.cross(t.position, t.direction)]) for t in thrusters]
    return np.column_stack(cols)


def thrust(thrusters: tuple[Thruster, ...], commands: np.ndarray) -> np.ndarray:
    """Thrust in newtons for commands in [-1, 1].

    Thrust grows with the square of the command (above the dead band), which is roughly how
    propeller thrust grows with speed. Detailed ESC curves belong in calibration.
    """
    c = np.clip(np.asarray(commands, dtype=float), -1.0, 1.0)
    fwd = np.array([t.max_forward for t in thrusters])
    rev = np.array([t.max_reverse for t in thrusters])
    band = np.array([t.dead_band for t in thrusters])
    # Inside the dead band the propeller does not turn; above it thrust grows from zero.
    c = np.sign(c) * np.clip((np.abs(c) - band) / (1.0 - band), 0.0, None)
    return np.where(c >= 0, fwd * c**2, -rev * c**2)


def thrust_rel_sigma(thruster: Thruster) -> float:
    return PRESET_REL_SIGMA if thruster.preset else USER_REL_SIGMA
