"""Hydrostatics of a fully submerged vehicle: weight, buoyancy and restoring forces."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import trimesh

from robomodel.parameter import Parameter, Source

GRAVITY = 9.80665  # m/s^2
CAD_VOLUME_REL_SIGMA = 0.02  # built vehicle vs CAD (seals, fasteners, cables), assumed


@dataclass(frozen=True)
class Hydrostatics:
    volume: Parameter  # displaced volume, m^3
    r_b: Parameter  # centre of buoyancy in body frame, m
    weight: Parameter  # W = m g, N
    buoyancy: Parameter  # B = rho g V, N


MEASURED_VOLUME_REL_SIGMA = 0.005  # weighing in water with a luggage scale


def hydrostatics(
    hull: trimesh.Trimesh,
    mass: Parameter,
    water_density: float,
    measured_volume: float | None = None,
) -> Hydrostatics:
    volume = float(hull.volume)
    note = "built vehicle may differ from CAD; check by weighing in water"
    source, rel = Source.CAD, CAD_VOLUME_REL_SIGMA
    if measured_volume is not None:
        volume, source, rel = measured_volume, Source.MEASURED, MEASURED_VOLUME_REL_SIGMA
        note = f"measured (CAD gave {float(hull.volume):.4g} m^3)"
    cob_sigma = np.full(3, CAD_VOLUME_REL_SIGMA * float(np.max(hull.extents)) / 2)
    weight = mass.value * GRAVITY
    buoyancy = water_density * GRAVITY * volume
    return Hydrostatics(
        volume=Parameter.relative(volume, rel, source, "m^3", note),
        r_b=Parameter(np.asarray(hull.center_mass, dtype=float), cob_sigma, Source.CAD, "m"),
        weight=Parameter(weight, mass.sigma * GRAVITY, mass.source, "N"),
        buoyancy=Parameter.relative(buoyancy, rel, source, "N", note),
    )


def restoring_forces(
    phi: float, theta: float, weight: float, buoyancy: float, r_g: np.ndarray, r_b: np.ndarray
) -> np.ndarray:
    """Restoring force vector g(eta) for a submerged vehicle (Fossen, Handbook of Marine
    Craft Hydrodynamics, "restoring forces for underwater vehicles").

    Sign convention: the equations of motion are M nu_dot + C nu + D nu + g(eta) = tau,
    so g is the negative of the gravity-buoyancy force, in the body frame.
    """
    w, b = weight, buoyancy
    xg, yg, zg = r_g
    xb, yb, zb = r_b
    sth, cth = np.sin(theta), np.cos(theta)
    sph, cph = np.sin(phi), np.cos(phi)
    return np.array(
        [
            (w - b) * sth,
            -(w - b) * cth * sph,
            -(w - b) * cth * cph,
            -(yg * w - yb * b) * cth * cph + (zg * w - zb * b) * cth * sph,
            (zg * w - zb * b) * sth + (xg * w - xb * b) * cth * cph,
            -(xg * w - xb * b) * cth * sph - (yg * w - yb * b) * sth,
        ]
    )
