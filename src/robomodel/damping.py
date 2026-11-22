"""Rough hydrodynamic damping from the hull shape.

Damping cannot be computed exactly from geometry (it depends on turbulence and flow
separation), so these are handbook estimates with wide uncertainty. Pool tests
(calibration) are what makes them accurate.

D(nu) nu = (D_lin + diag(d_quad * |nu|)) nu, with positive coefficients. In Fossen's
notation X_u = -d_lin[0], X_u|u| = -d_quad[0], and so on.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from robomodel.parameter import Parameter, Source
from robomodel.sections import Sections

SKIN_FRICTION = 0.0045  # turbulent flat-plate value at typical AUV Reynolds numbers, assumed
CROSSFLOW_CD_ROUND = 1.1  # circular cylinder in crossflow (Hoerner, Fluid-Dynamic Drag)
CROSSFLOW_CD_SQUARE = 2.0  # square section in crossflow (Hoerner)
BLUFF_AXIAL_CD = 0.9  # stubby bodies (L/D < 3), rough
STREAMLINED_LD = 3.0
REF_SPEED = 0.1  # m/s: linear damping = quadratic at this speed, assumed
REF_RATE = 0.2  # rad/s, same for rotations
ROLL_FLOOR = 0.05  # roll linear damping at least this share of pitch linear damping
QUAD_REL_SIGMA = 0.5
LINEAR_REL_SIGMA = 1.0


@dataclass(frozen=True)
class Damping:
    linear: Parameter  # shape (6,), N/(m/s) and N m/(rad/s)
    quadratic: Parameter  # shape (6,), N/(m/s)^2 and N m/(rad/s)^2
    axial_cd: float  # drag coefficient on frontal area used for surge


def axial_drag_coefficient(ld: float, end_fraction: float = 0.0) -> float:
    """Surge drag coefficient on the frontal area.

    Streamlined bodies of revolution: Hoerner's fineness-ratio formula
    C_D = C_f [3 L/D + 4.5 (D/L)^0.5 + 21 (D/L)^2]. Stubby bodies, and hulls with flat
    ends (end_fraction = end section area / largest section area), get a bluff-body value.
    """
    if ld < STREAMLINED_LD:
        return BLUFF_AXIAL_CD
    dl = 1.0 / ld
    streamlined = SKIN_FRICTION * (3 * ld + 4.5 * dl**0.5 + 21 * dl**2)
    return max(streamlined, BLUFF_AXIAL_CD * end_fraction)


def crossflow_cd(sections: Sections) -> np.ndarray:
    """Per-section crossflow drag coefficient: round sections 1.1, square ones 2.0."""
    box = 4 * sections.half_width * sections.half_height
    fill = np.divide(sections.area, box, out=np.zeros_like(box), where=box > 0)
    squareness = np.clip((fill - np.pi / 4) / (1 - np.pi / 4), 0.0, 1.0)
    return CROSSFLOW_CD_ROUND + (CROSSFLOW_CD_SQUARE - CROSSFLOW_CD_ROUND) * squareness


def estimate_damping(sections: Sections, rho: float) -> Damping:
    x, bw, bh = sections.x, sections.half_width, sections.half_height
    frontal = float(sections.area.max())
    diameter = 2 * np.sqrt(frontal / np.pi)
    ld = sections.length / diameter
    end_fraction = float(max(sections.area[0], sections.area[-1]) / frontal)
    cd_axial = axial_drag_coefficient(ld, end_fraction)
    cdc = crossflow_cd(sections)
    half_rho = 0.5 * rho

    quad = np.array(
        [
            half_rho * cd_axial * frontal,
            half_rho * sections.integrate(cdc * 2 * bh),  # sway: side area
            half_rho * sections.integrate(cdc * 2 * bw),  # heave: plan area
            # roll: a section spinning about x pushes water only where it is not round
            half_rho * sections.integrate(cdc * np.abs(bw**4 - bh**4) / 2),
            half_rho * sections.integrate(cdc * 2 * bw * np.abs(x) ** 3),  # pitch
            half_rho * sections.integrate(cdc * 2 * bh * np.abs(x) ** 3),  # yaw
        ]
    )
    ref = np.array([REF_SPEED] * 3 + [REF_RATE] * 3)
    linear = quad * ref
    linear[3] = max(linear[3], ROLL_FLOOR * linear[4])

    note = (
        "handbook estimate (Hoerner); fins and other appendages are not included. "
        "Pool tests are needed for accurate damping"
    )
    return Damping(
        linear=Parameter(
            linear,
            linear * LINEAR_REL_SIGMA,
            Source.ASSUMED,
            "N s/m, N m s/rad",
            f"linear = quadratic x {REF_SPEED} m/s (or {REF_RATE} rad/s), assumed",
        ),
        quadratic=Parameter(
            quad, quad * QUAD_REL_SIGMA, Source.EMPIRICAL, "N s^2/m^2, N m s^2/rad^2", note
        ),
        axial_cd=cd_axial,
    )
