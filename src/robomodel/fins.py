"""Fins and control surfaces: lift, added mass and control input.

A fin with unit normal n at position r sees the local velocity v_l = v + w x r. In
forward motion at speed u its lift opposes the normal component v_n = n . v_l:

    F = -k |u| v_n n + k u|u| delta n,     k = 1/2 rho C_L_alpha S

In generalised coordinates h = [n; r x n] gives v_n = h . nu and tau = h F_n, so
    passive part:  tau = -|u| D_fin nu,  D_fin = sum k h h^T   (speed-dependent damping)
    control part:  tau = u|u| B_fin delta,  B_fin column = k h  (per radian of deflection)

This is the small-angle lift-slope model used for torpedo AUVs (e.g. Prestero 2001); it
ignores stall, fin-fin interference and the hull's own lift.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from robomodel.parameter import Parameter, Source
from robomodel.vehicle import Fin

LIFT_REL_SIGMA = 0.2
ADDED_MASS_REL_SIGMA = 0.3


@dataclass(frozen=True)
class FinModel:
    d_fin: Parameter  # 6x6, multiply by |u|
    b_fin: np.ndarray  # 6 x n_control, multiply by u|u| and deflection (rad)
    m_a: np.ndarray  # 6x6 added mass of all fins
    lift_slopes: np.ndarray  # C_L_alpha per fin, 1/rad
    control_names: tuple[str, ...]


def generalised_direction(fin: Fin) -> np.ndarray:
    return np.concatenate([fin.normal, np.cross(fin.position, fin.normal)])


def lift_slope(aspect_ratio: float) -> float:
    """Helmbold's lift-curve slope for a low aspect ratio wing, per radian."""
    return 2 * np.pi * aspect_ratio / (2 + np.sqrt(aspect_ratio**2 + 4))


def effective_aspect_ratio(fin: Fin) -> float:
    """The hull acts as a reflection plane at the fin root, doubling the aspect ratio."""
    return 2 * fin.span**2 / fin.area


def fin_added_mass(fin: Fin, rho: float) -> float:
    """Flat plate of width `span` moving normal to itself: rho pi (span/2)^2 per chord."""
    chord = fin.area / fin.span
    return rho * np.pi * (fin.span / 2) ** 2 * chord


def fin_model(fins: tuple[Fin, ...], rho: float) -> FinModel:
    d = np.zeros((6, 6))
    m_a = np.zeros((6, 6))
    control_cols, names, slopes = [], [], []
    for fin in fins:
        h = generalised_direction(fin)
        cla = lift_slope(effective_aspect_ratio(fin))
        k = 0.5 * rho * cla * fin.area
        d += k * np.outer(h, h)
        m_a += fin_added_mass(fin, rho) * np.outer(h, h)
        slopes.append(cla)
        if fin.control:
            control_cols.append(k * h)
            names.append(fin.name)
    b = np.column_stack(control_cols) if control_cols else np.zeros((6, 0))
    note = "fin lift: Helmbold lift slope, small angles, no stall or fin-fin interference"
    return FinModel(
        d_fin=Parameter(d, np.abs(d) * LIFT_REL_SIGMA, Source.EMPIRICAL, "N s/m (x |u|)", note),
        b_fin=b,
        m_a=m_a,
        lift_slopes=np.array(slopes),
        control_names=tuple(names),
    )
