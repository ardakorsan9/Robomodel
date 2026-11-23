"""Multirotor drones: hover, control authority and aerodynamic drag.

The rigid-body part (mass, CG, inertia, rotor layout) follows from the CAD and the parts list
just as for a vehicle in water. In air there is no buoyancy or added mass worth modelling (air
is ~800x lighter than water). What remains is drag:

* Frame drag: bluff-body drag on the projected areas (same method as for open-frame ROVs,
  with the density of air).
* Rotor drag: a spinning rotor moving sideways turns the air it swallows, which acts like a
  linear drag. Momentum theory gives the "ram drag" m_dot v, with the mass flow through the
  disk m_dot = sqrt(rho A T / 2) at thrust T. Measured rotor drag is usually lower (part of
  the flow is not turned); half of it is taken, assumed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import lsq_linear

from robomodel.hydrostatics import GRAVITY

ROTOR_DRAG_SHARE = 0.5  # of the momentum-theory ram drag, assumed
GOOD_THRUST_TO_WEIGHT = 1.5  # below this the drone has little thrust left for control


@dataclass(frozen=True)
class Hover:
    thrust: np.ndarray  # N per rotor, level hover
    residual: float  # N or N m, 0 when level hover is possible
    thrust_to_weight: float  # total thrust along -z at full command / weight

    @property
    def possible(self) -> bool:
        return self.residual < 1e-6

    def commands(self, f_max: np.ndarray) -> np.ndarray:
        """Commands in [0, 1] giving the hover thrusts (thrust ~ command^2)."""
        return np.sqrt(np.clip(self.thrust / np.maximum(f_max, 1e-12), 0.0, 1.0))


def hover_wrench(weight: float, r_g: np.ndarray) -> np.ndarray:
    """tau that holds the drone level and still: it cancels gravity acting at the CG."""
    w = np.array([0.0, 0.0, weight])  # gravity, body frame, level
    return -np.concatenate([w, np.cross(r_g, w)])


def hover(b_thr: np.ndarray, f_max: np.ndarray, weight: float, r_g: np.ndarray) -> Hover:
    """Rotor thrusts (0 <= f <= f_max) for level hover, as evenly shared as the layout allows."""
    target = hover_wrench(weight, r_g)
    up = -b_thr[2]  # thrust per newton along -z (up)
    t_over_w = float(np.sum(np.clip(up, 0, None) * f_max) / weight) if weight > 0 else np.inf
    if b_thr.shape[1] == 0:
        return Hover(np.zeros(0), float(np.linalg.norm(target)), 0.0)
    # Balance exactly, then share: a small penalty pulls every rotor towards the same load.
    share = weight / max(b_thr.shape[1], 1)
    k = 1e-5  # tiny: the balance rows win, sharing only picks among exact solutions
    a = np.vstack([b_thr, k * np.eye(b_thr.shape[1])])
    b = np.concatenate([target, np.full(b_thr.shape[1], k * share)])
    sol = lsq_linear(a, b, bounds=(0.0, f_max), tol=1e-12)
    f = sol.x
    residual = float(np.max(np.abs(b_thr @ f - target)) / max(weight, 1e-12))
    return Hover(f, residual, t_over_w)


def rotor_drag(thrusters, hover_thrust: np.ndarray, rho: float) -> np.ndarray:
    """Linear drag (6,) from the rotors at hover: m_dot (I - a a^T) per rotor, translations."""
    d = np.zeros(6)
    for t, f in zip(thrusters, hover_thrust, strict=True):
        if t.diameter <= 0 or f <= 0:
            continue
        area = np.pi * t.diameter**2 / 4
        m_dot = np.sqrt(rho * area * f / 2)
        d[:3] += ROTOR_DRAG_SHARE * m_dot * np.diag(np.eye(3) - np.outer(t.direction, t.direction))
    return d


def hover_weight(mass: float) -> float:
    return mass * GRAVITY
