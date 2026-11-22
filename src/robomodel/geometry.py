"""Turn a Geometry description into a closed triangle mesh in the body frame (metres).

Primitive shapes are centred on their bounding box, with the long axis along body x
(forward). Mesh files are used as they are, only scaled to metres.
"""

from __future__ import annotations

import numpy as np
import trimesh

from robomodel.vehicle import Geometry

SPHERE_SUBDIVISIONS = 6  # ~41k vertices, volume error ~1e-4
REVOLVE_SECTIONS = 256  # around the axis, volume error ~1e-4
PROFILE_POINTS = 400  # along the axis for Myring hulls


class GeometryError(ValueError):
    pass


def build_mesh(geometry: Geometry) -> trimesh.Trimesh:
    if geometry.file is not None:
        mesh = _load_file(geometry)
    else:
        mesh = _PRIMITIVE_BUILDERS[geometry.primitive](**geometry.dims)
    if not mesh.is_watertight:
        raise GeometryError(
            "the mesh is not closed (watertight); volume and buoyancy would be wrong. "
            "Export the CAD body as a single solid, or repair it (e.g. in MeshLab)."
        )
    if mesh.volume < 0:
        mesh.invert()  # inward-facing normals give a negative volume
    if geometry.origin == "buoyancy":
        mesh.apply_translation(-np.asarray(mesh.center_mass))
    return mesh


def _load_file(geometry: Geometry) -> trimesh.Trimesh:
    try:
        mesh = trimesh.load(geometry.file, force="mesh")
    except Exception as e:  # trimesh raises many different types
        raise GeometryError(f"could not read {geometry.file}: {e}") from e
    if not isinstance(mesh, trimesh.Trimesh) or len(mesh.faces) == 0:
        raise GeometryError(f"{geometry.file} contains no triangles")
    mesh.apply_scale(geometry.unit_scale)
    return mesh


def _sphere(radius: float) -> trimesh.Trimesh:
    return trimesh.creation.icosphere(subdivisions=SPHERE_SUBDIVISIONS, radius=radius)


def _ellipsoid(a: float, b: float, c: float) -> trimesh.Trimesh:
    mesh = trimesh.creation.icosphere(subdivisions=SPHERE_SUBDIVISIONS, radius=1.0)
    mesh.apply_transform(np.diag([a, b, c, 1.0]))
    return mesh


def _cylinder(length: float, radius: float) -> trimesh.Trimesh:
    mesh = trimesh.creation.cylinder(radius=radius, height=length, sections=REVOLVE_SECTIONS)
    return _z_axis_to_x(mesh)


def myring_radius(h, a, b, c, d, n, theta):
    """Myring hull radius at distance h from the tip of the full (untruncated) nose.

    Prestero 2001, eqs. 2.1-2.2 and Table 2.1: a nose length, b parallel mid-body length,
    c tail length, d max diameter, n nose exponent, theta tail angle parameter (rad).
    """
    h = np.asarray(h, dtype=float)
    r = np.full_like(h, d / 2)
    nose = h < a
    r[nose] = d / 2 * np.clip(1 - ((h[nose] - a) / a) ** 2, 0, None) ** (1 / n)
    tail = h > a + b
    s = h[tail] - (a + b)
    t = np.tan(theta)
    r[tail] = d / 2 - (3 * d / (2 * c**2) - t / c) * s**2 + (d / c**3 - t / c**2) * s**3
    return np.clip(r, 0, None)


def _myring(a, b, c, d, n, theta, a_offset=0.0, c_offset=0.0) -> trimesh.Trimesh:
    """Myring hull. a_offset / c_offset cut that much off the nose / tail tip (flat ends),
    as on the real REMUS (Prestero Table 2.1)."""
    start, end = a_offset, a + b + c - c_offset
    length = end - start
    # Even spacing: denser spacing near the tips makes trimesh merge vertices there and
    # leaves holes in the mesh.
    h = np.linspace(start, end, PROFILE_POINTS)
    r = myring_radius(h, a, b, c, d, n, theta)
    height = length / 2 - (h - start)  # nose at +length/2
    # Close both ends on the axis; a truncated end becomes a flat face.
    profile = np.column_stack([r, height])
    if r[0] > 0:
        profile = np.vstack([[0.0, height[0]], profile])
    else:
        profile[0, 0] = 0.0
    if r[-1] > 0:
        profile = np.vstack([profile, [0.0, height[-1]]])
    else:
        profile[-1, 0] = 0.0
    mesh = trimesh.creation.revolve(profile, sections=REVOLVE_SECTIONS)
    return _z_axis_to_x(mesh)


def _z_axis_to_x(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Rotate so the mesh's z axis points along body x (forward)."""
    mesh.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [0, 1, 0]))
    return mesh


_PRIMITIVE_BUILDERS = {
    "sphere": _sphere,
    "ellipsoid": _ellipsoid,
    "cylinder": _cylinder,
    "myring": _myring,
}
