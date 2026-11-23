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

    shell = vehicle.geometry.hull_mass == "shell" or (
        vehicle.geometry.hull_mass == "auto" and vehicle.category == "surface"
    )
    if shell and isinstance(hull, trimesh.Trimesh):
        hull_centroid, hull_inertia_c = shell_properties(hull, hull_mass)
    else:
        # Uniform solid. trimesh gives the inertia about the centroid for density 1.
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
    spread = "over the hull skin" if shell and isinstance(hull, trimesh.Trimesh) else "uniform"
    note = (
        f"hull mass {hull_mass:.3g} kg assumed {spread}; list heavy parts to improve this"
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


def shell_properties(mesh: trimesh.Trimesh, mass: float) -> tuple[np.ndarray, np.ndarray]:
    """Centroid and inertia about it of `mass` spread evenly over the mesh surface.

    Exact for thin triangles: int r r^T dA = A/12 (sum v_i v_i^T + s s^T), s = sum v_i.
    """
    tri = mesh.vertices[mesh.faces]
    area = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    total = area.sum()
    centroid = (area @ tri.mean(axis=1)) / total
    rel = tri - centroid
    s = rel.sum(axis=1)
    second = np.einsum("k,kij,kil->jl", area, rel, rel) + np.einsum("k,ki,kl->il", area, s, s)
    second *= mass / total / 12  # int r r^T dm
    inertia = np.trace(second) * np.eye(3) - second
    return centroid, inertia


def rigid_body_mass_matrix(props: MassProperties) -> np.ndarray:
    """Fossen M_RB about CO (Fossen, Handbook of Marine Craft Hydrodynamics):

    [[ m I,      -m S(r_g) ],
     [ m S(r_g),  I_o      ]]
    """
    m = props.mass.value
    s = skew(props.r_g.value)
    return np.block([[m * np.eye(3), -m * s], [m * s, props.inertia_o.value]])


def velocity_transform(r) -> np.ndarray:
    """H(r) with nu_P = H nu_O: velocity of the point P at r, from the velocity at origin O.

    v_P = v_O + w x r = v_O - S(r) w. Forces move back with H^T, and a mass (or added mass)
    matrix given about P becomes H^T M_P H about O (Fossen, "system transformation matrix").
    """
    h = np.eye(6)
    h[:3, 3:] = -skew(np.asarray(r, dtype=float))
    return h


def move_mass_matrix(m_p: np.ndarray, r) -> np.ndarray:
    """A 6x6 mass matrix about point P (at r from the origin) expressed about the origin."""
    h = velocity_transform(r)
    return h.T @ m_p @ h
