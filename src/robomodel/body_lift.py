"""Hull lift of an elongated body at an angle of attack (Hoerner).

Flow separation on the aft, leeward side of the hull makes a sideways force that grows
with u*v (or u*w), applied at a centre of pressure about 0.65 L from the nose:

    Y = -k u v,   Z = -k u w,   k = 1/2 rho d^2 c_yd_beta,   c_yd_beta = c_y' (L/d)

with Hoerner's c_y' = 0.003 per degree (for 6.7 < L/d < 10) = 0.172 per radian. Prestero
(2001, Sec. 4.4) uses the same formula for REMUS and notes that the four methods he
compared disagreed by an order of magnitude, so the uncertainty is large.

In generalised form the force acts at r_cp: tau = -|u| D_bl nu with
    D_bl = k (h_y e_v^T + h_z e_w^T),   h = [n; r_cp x n]
(not symmetric: only sideslip makes the force, rotation does not).
"""

from __future__ import annotations

import numpy as np

from robomodel.parameter import Parameter, Source
from robomodel.rigid_body import velocity_transform
from robomodel.sections import Sections

HOERNER_CY_PER_DEG = 0.003
CENTRE_OF_PRESSURE = 0.65  # fraction of the length from the nose
MIN_LD = 3.0  # below this the hull is not slender and the formula does not apply
REL_SIGMA = 0.5


def body_lift(sections: Sections, nose_x: float, rho: float,
              centre=(0.0, 0.0, 0.0)) -> Parameter | None:
    """Body lift matrix (multiply by |u|), or None for a stubby hull."""
    frontal = float(sections.area.max())
    d = 2 * np.sqrt(frontal / np.pi)
    ld = sections.length / d
    if ld < MIN_LD:
        return None
    c_yd_beta = HOERNER_CY_PER_DEG * np.degrees(1.0) * ld
    k = 0.5 * rho * d**2 * c_yd_beta
    x_cp = nose_x - CENTRE_OF_PRESSURE * sections.length

    m = np.zeros((6, 6))
    centre = np.asarray(centre, dtype=float)
    r = np.array([x_cp, centre[1], centre[2]])  # on the hull's axis
    # The sideslip is measured at the hull centre: velocity there = H nu. This keeps the
    # force independent of where the CAD origin is.
    h_centre = velocity_transform(centre)
    for axis, n in ((1, np.array([0.0, 1.0, 0.0])), (2, np.array([0.0, 0.0, 1.0]))):
        h = np.concatenate([n, np.cross(r, n)])
        m += k * np.outer(h, h_centre[axis])
    note = (
        f"Hoerner body lift: c_yd_beta = {c_yd_beta:.3g}, centre of pressure x = {x_cp:.3g} m; "
        "methods disagree widely (Prestero 2001), measure if it matters"
    )
    return Parameter(m, np.abs(m) * REL_SIGMA, Source.EMPIRICAL, "N s/m (x |u|)", note)
