"""Rigid-body mass properties and the Fossen rigid-body mass matrix M_RB.

Mass model: the parts listed in the vehicle file are uniform solids (or point masses) with
their own inertia; the remaining mass
(total - parts) is spread uniformly over the hull volume. Everything is expressed about
the body-frame origin CO (see docs/eksenler.md).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import trimesh

from robomodel.parameter import Parameter, Source
from robomodel.vehicle import Part, Vehicle

MASS_REL_SIGMA = 0.01  # typical scale accuracy
UNIFORM_INERTIA_REL_SIGMA = 0.30  # real mass is not spread evenly over the hull
UNIFORM_COG_SIGMA_FRACTION = 0.05  # of the hull's size along each axis
MEASURED_COG_SIGMA = 0.002  # m, a careful balance test
MEASURED_INERTIA_REL_SIGMA = 0.05  # pendulum test


@dataclass(frozen=True)
class MassProperties:
    mass: Parameter  # kg
    r_g: Parameter  # centre of gravity in body frame, m, shape (3,)
    inertia_o: Parameter  # inertia tensor about CO, kg m^2, shape (3, 3)
    inertia_g: Parameter  # inertia tensor about CG, kg m^2, shape (3, 3)


def skew(v: np.ndarray) -> np.ndarray:
    """Cross-product matrix S(v), so that S(v) @ w == np.cross(v, w)."""
    x, y, z = v
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def point_inertia(mass: float, r: np.ndarray) -> np.ndarray:
    """Inertia of a point mass at r about the origin (parallel-axis term)."""
    r = np.asarray(r, dtype=float)
    return mass * (np.dot(r, r) * np.eye(3) - np.outer(r, r))


def part_inertia(part: Part) -> np.ndarray:
    """Inertia of a uniform solid part about its own centre, body axes."""
    m, d = part.mass, part.dims
    if part.shape == "point":
        return np.zeros((3, 3))
    if part.shape == "sphere":
        return 2 / 5 * m * d["radius"] ** 2 * np.eye(3)
    if part.shape == "box":
        lx, ly, lz = d["size"]
        return m / 12 * np.diag([ly**2 + lz**2, lx**2 + lz**2, lx**2 + ly**2])
    if part.shape == "cylinder":
        r, length = d["radius"], d["length"]
        axial = m * r**2 / 2
        transverse = m * (3 * r**2 + length**2) / 12
        diag = np.full(3, transverse)
        diag["xyz".index(d["axis"])] = axial
        return np.diag(diag)
    raise ValueError(f"unknown part shape {part.shape!r}")


def mass_properties(vehicle: Vehicle, hull: trimesh.Trimesh) -> MassProperties:
    parts_mass = sum(p.mass for p in vehicle.parts)
    hull_mass = vehicle.mass - parts_mass

    # Hull: uniform solid. trimesh gives the inertia about the hull centroid for density 1.
    hull_centroid = np.asarray(hull.center_mass)
    hull_inertia_c = np.asarray(hull.moment_inertia) * (hull_mass / hull.volume)

    first_moment = hull_mass * hull_centroid
    inertia_o = hull_inertia_c + point_inertia(hull_mass, hull_centroid)
    for part in vehicle.parts:
        first_moment = first_moment + part.mass * part.position
        inertia_o = inertia_o + part_inertia(part) + point_inertia(part.mass, part.position)

    r_g = first_moment / vehicle.mass
    inertia_g = inertia_o - point_inertia(vehicle.mass, r_g)
    measured = vehicle.measured
    if measured.r_g is not None:
        r_g = measured.r_g
    if measured.inertia_g is not None:
        inertia_g = measured.inertia_g
    if measured.r_g is not None or measured.inertia_g is not None:
        inertia_o = inertia_g + point_inertia(vehicle.mass, r_g)

    # Uncertainty: the uniform-hull assumption dominates when the hull carries the mass.
    hull_share = hull_mass / vehicle.mass
    # per axis: a hull cannot hide its CG further than its own size along that axis
    cog_sigma = UNIFORM_COG_SIGMA_FRACTION * np.asarray(hull.extents, dtype=float) * hull_share
    note = (
        f"hull mass {hull_mass:.3g} kg assumed uniform; list heavy parts to improve this"
        if hull_share > 0
        else "all mass given as parts"
    )
    source = Source.ASSUMED if hull_share > 0 else Source.MEASURED

    def inertia_param(tensor: np.ndarray, is_measured: bool) -> Parameter:
        if is_measured:
            return Parameter.relative(
                tensor, MEASURED_INERTIA_REL_SIGMA, Source.MEASURED, "kg m^2", "measured"
            )
        sigma = np.abs(tensor) * UNIFORM_INERTIA_REL_SIGMA * hull_share
        return Parameter(tensor, sigma, source, "kg m^2", note)

    if measured.r_g is not None:
        r_g_param = Parameter(r_g, np.full(3, MEASURED_COG_SIGMA), Source.MEASURED, "m", "measured")
    else:
        r_g_param = Parameter(r_g, cog_sigma, source, "m", note)
    inertia_measured = measured.inertia_g is not None
    return MassProperties(
        mass=Parameter.relative(vehicle.mass, MASS_REL_SIGMA, Source.MEASURED, "kg"),
        r_g=r_g_param,
        inertia_o=inertia_param(inertia_o, inertia_measured),
        inertia_g=inertia_param(inertia_g, inertia_measured),
    )


def rigid_body_mass_matrix(props: MassProperties) -> np.ndarray:
    """Fossen M_RB about CO (Fossen, Handbook of Marine Craft Hydrodynamics):

    [[ m I,      -m S(r_g) ],
     [ m S(r_g),  I_o      ]]
    """
    m = props.mass.value
    s = skew(props.r_g.value)
    return np.block([[m * np.eye(3), -m * s], [m * s, props.inertia_o.value]])
