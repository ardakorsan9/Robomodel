"""Added mass from potential flow.

Exact for ellipsoids (Lamb, *Hydrodynamics*, 6th ed., §114 and §373). Any other elongated
hull gets strip theory with an end correction. BEM (Capytaine) is an optional later step.
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import quad

from robomodel.parameter import Parameter, Source
from robomodel.sections import Sections

# Real fluid, appendages and a vehicle that is only roughly ellipsoidal: assumed.
ELLIPSOID_REL_SIGMA = 0.10


def lamb_integrals(a: float, b: float, c: float) -> tuple[float, float, float]:
    """Lamb's shape integrals alpha0, beta0, gamma0 for semi-axes a (x), b (y), c (z).

    alpha0 = a b c * integral_0^inf  dl / ((a^2 + l) * Delta(l)),
    Delta(l) = sqrt((a^2 + l)(b^2 + l)(c^2 + l)). They sum to 2.
    """

    def integral(axis_sq: float) -> float:
        def f(lam):
            delta = np.sqrt((a * a + lam) * (b * b + lam) * (c * c + lam))
            return 1.0 / ((axis_sq + lam) * delta)

        scale = max(a, b, c) ** 2  # split the infinite range where the integrand changes
        return a * b * c * (quad(f, 0, scale, limit=200)[0] + quad(f, scale, np.inf, limit=200)[0])

    return integral(a * a), integral(b * b), integral(c * c)


def ellipsoid_added_mass(a: float, b: float, c: float, rho: float) -> np.ndarray:
    """Diagonal added mass [m11 .. m66] (positive) of an ellipsoid in an unbounded fluid.

    Fossen derivatives are the negatives: X_udot = -m11, ..., N_rdot = -m66.
    """
    alpha, beta, gamma = lamb_integrals(a, b, c)
    displaced = rho * 4.0 / 3.0 * np.pi * a * b * c

    def translational(k0: float) -> float:
        return k0 / (2.0 - k0) * displaced

    def rotational(p: float, q: float, k_p: float, k_q: float) -> float:
        """Rotation about the axis normal to semi-axes p and q (Lamb §373)."""
        p2, q2 = p * p, q * q
        if abs(p2 - q2) < 1e-12 * max(p2, q2):
            return 0.0  # rotating a circular section displaces no fluid
        diff = p2 - q2
        return displaced / 5.0 * diff**2 * (k_q - k_p) / (2.0 * diff + (p2 + q2) * (k_p - k_q))

    return np.array(
        [
            translational(alpha),
            translational(beta),
            translational(gamma),
            rotational(b, c, beta, gamma),  # roll: about x
            rotational(c, a, gamma, alpha),  # pitch: about y
            rotational(a, b, alpha, beta),  # yaw: about z
        ]
    )


def ellipsoid_added_mass_parameter(a: float, b: float, c: float, rho: float) -> Parameter:
    diag = ellipsoid_added_mass(a, b, c, rho)
    return Parameter(
        np.diag(diag),
        np.diag(diag * ELLIPSOID_REL_SIGMA),
        Source.ANALYTIC,
        "kg, kg m^2",
        "potential flow, exact for an ideal ellipsoid (Lamb)",
    )


# --- Strip theory for any elongated hull ---------------------------------------------

STRIP_REL_SIGMA = 0.15  # slender hulls (L/D >= 4), assumed
STRIP_REL_SIGMA_STUBBY = 0.35  # L/D < 4: the slender-body assumption is weak
STUBBY_LD = 4.0


def spheroid_coefficients(ld: float) -> tuple[float, float, float]:
    """Lamb's k1 (axial), k2 (lateral), k' (rotation) for a prolate spheroid of L/D = ld."""
    a, b = ld / 2, 0.5
    m = ellipsoid_added_mass(a, b, b, 1.0)
    volume = 4 / 3 * np.pi * a * b * b
    return m[0] / volume, m[1] / volume, m[5] / (volume * (a * a + b * b) / 5)


def strip_theory_added_mass(sections: Sections, volume: float, rho: float) -> np.ndarray:
    """6x6 added mass (positive) of an elongated hull from its cross-sections.

    Each section gets the 2D added mass of an ellipse with the section's half width bw
    and half height bh: sway rho*pi*bh^2, heave rho*pi*bw^2, roll rho*pi/8*(bw^2-bh^2)^2.
    Integrating along x gives sway, heave, pitch, yaw and their couplings. Strip theory
    ignores the flow around the ends, so lateral and rotational terms are scaled by the
    Lamb/strip ratio of the spheroid with the same length and volume, and the axial term
    is taken from that spheroid (k1 * rho * V).
    """
    x, bw, bh = sections.x, sections.half_width, sections.half_height
    sway = rho * np.pi * bh**2  # per unit length
    heave = rho * np.pi * bw**2
    roll = rho * np.pi / 8 * (bw**2 - bh**2) ** 2

    ld = equivalent_ld(sections.length, volume)
    k1, k2, kr = spheroid_coefficients(ld)
    a = ld / 2  # spheroid with D = 1; strip theory gives m66 = rho V a^2 / 5 for it
    rot_correction = kr * (a * a + 0.25) / (a * a)

    m = np.zeros((6, 6))
    m[0, 0] = k1 * rho * volume
    m[1, 1] = k2 * sections.integrate(sway)
    m[2, 2] = k2 * sections.integrate(heave)
    m[3, 3] = sections.integrate(roll)
    m[4, 4] = rot_correction * sections.integrate(heave * x**2)
    m[5, 5] = rot_correction * sections.integrate(sway * x**2)
    m[1, 5] = m[5, 1] = k2 * sections.integrate(sway * x)  # sway-yaw
    m[2, 4] = m[4, 2] = -k2 * sections.integrate(heave * x)  # heave-pitch
    return m


def equivalent_ld(length: float, volume: float) -> float:
    """Length/diameter of the prolate spheroid with this length and volume."""
    a = length / 2
    b = np.sqrt(3 * volume / (4 * np.pi * a))
    return max(length / (2 * b), 1.0 + 1e-6)  # never below a sphere


def strip_theory_parameter(sections: Sections, volume: float, rho: float) -> Parameter:
    m = strip_theory_added_mass(sections, volume, rho)
    ld = equivalent_ld(sections.length, volume)
    rel = STRIP_REL_SIGMA if ld >= STUBBY_LD else STRIP_REL_SIGMA_STUBBY
    note = f"strip theory with spheroid end correction (equivalent L/D = {ld:.2g})"
    if ld < STUBBY_LD:
        note += "; hull is stubby, BEM (Capytaine) would be more accurate"
    return Parameter(m, np.abs(m) * rel, Source.EMPIRICAL, "kg, kg m, kg m^2", note)
