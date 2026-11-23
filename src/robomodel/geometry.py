"""Turn a Geometry description into a closed triangle mesh in the body frame (metres).

Primitive shapes are centred on their bounding box, with the long axis along body x
(forward). Mesh files are scaled to metres and rotated by `axes`. Assemblies (open, multi-part
CAD) become a VoxelHull, which offers the same properties as a closed mesh.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import trimesh

from robomodel.vehicle import Geometry, axes_matrix
from robomodel.voxel import VoxelHull, voxelize

SPHERE_SUBDIVISIONS = 6  # ~41k vertices, volume error ~1e-4
REVOLVE_SECTIONS = 256  # around the axis, volume error ~1e-4
PROFILE_POINTS = 400  # along the axis for Myring hulls


class GeometryError(ValueError):
    pass


def build_mesh(geometry: Geometry) -> trimesh.Trimesh | VoxelHull:
    if geometry.file is not None:
        mesh = _load_file(geometry)
    else:
        mesh = _PRIMITIVE_BUILDERS[geometry.primitive](**geometry.dims)
    notes: list[str] = []
    parts = mesh.metadata.get("cad_parts", []) if geometry.file is not None else []
    if geometry.file is not None:
        mesh, notes = _clean(mesh)
    if len(mesh.faces) == 0 or np.min(mesh.extents) <= 1e-9 * max(np.max(mesh.extents), 1e-12):
        raise GeometryError(
            "the geometry has no volume (it is flat or a line); export the solid body, not a "
            "surface or sketch"
        )
    if not geometry.assembly and not mesh.is_watertight:
        notes.append(
            "the CAD is not a closed solid (holes or separate open surfaces) and could not be "
            "repaired, so it was turned into voxels like an assembly; volume may be high: "
            "check it, or export a closed solid"
        )
    nested = not geometry.assembly and mesh.is_watertight and _has_nested_bodies(mesh)
    if nested:
        notes.append(
            "the CAD has solids inside other solids (e.g. a battery inside the hull); they were "
            "merged with voxels so their volume is not counted twice"
        )
    if geometry.assembly or not mesh.is_watertight or nested:
        pitch = geometry.voxel if geometry.assembly else float(np.max(mesh.extents)) / 200
        broken_single_body = not geometry.assembly and not mesh.is_watertight
        hull = voxelize(mesh, pitch, close_gaps=broken_single_body)
        hull.metadata["warnings"] = notes
        if geometry.origin == "buoyancy":
            shift = -hull.center_mass
            hull.apply_translation(shift)
            parts = _shifted(parts, shift)
        hull.metadata["cad_parts"] = parts
        return hull
    mesh.metadata["warnings"] = notes
    if mesh.volume < 0:
        mesh.invert()  # inward-facing normals give a negative volume
    if geometry.origin == "buoyancy":
        shift = -np.asarray(mesh.center_mass)
        mesh.apply_translation(shift)
        parts = _shifted(parts, shift)
    mesh.metadata["cad_parts"] = parts
    return mesh


def _shifted(parts, offset):
    if not parts:
        return parts
    from robomodel.mass_parts import translate

    return translate(parts, offset)


def _clean(mesh: trimesh.Trimesh) -> tuple[trimesh.Trimesh, list[str]]:
    """Fix the usual export defects: duplicate vertices and faces, degenerate triangles,
    small holes, flipped normals. Returns notes about what was repaired."""
    notes = []
    faces_before = len(mesh.faces)
    mesh.merge_vertices()
    mesh.update_faces(mesh.unique_faces())
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.remove_unreferenced_vertices()
    removed = faces_before - len(mesh.faces)
    if removed:
        notes.append(f"removed {removed} duplicate or degenerate triangles from the CAD")
    if not mesh.is_watertight:
        try:
            trimesh.repair.fill_holes(mesh)
            if not mesh.is_watertight:
                mesh = _cap_holes(mesh)
        except Exception:  # repair is best effort
            pass
        if mesh.is_watertight:
            notes.append("closed holes in the CAD surface (check the volume)")
    if mesh.is_watertight and not mesh.is_winding_consistent:
        trimesh.repair.fix_winding(mesh)
    return mesh, notes


def _load_file(geometry: Geometry) -> trimesh.Trimesh:
    suffix = Path(geometry.file).suffix.lower()
    if suffix in (".step", ".stp") and importlib.util.find_spec("cascadio") is None:
        raise GeometryError("STEP files need the 'cascadio' package: uv add cascadio")
    if suffix in (".igs", ".iges", ".sldprt", ".f3d"):
        raise GeometryError(f"{suffix} files are not supported: export STEP or STL from the CAD")
    try:
        # STEP: OpenCASCADE (cascadio) tessellates the solids and returns metres, so only an
        # explicit `unit:` in the vehicle file changes the scale.
        loaded = trimesh.load(geometry.file)
    except Exception as e:  # trimesh raises many different types
        raise GeometryError(f"could not read {geometry.file}: {e}") from e
    rotation = axes_matrix(geometry.axes)
    if any(geometry.rotate):
        rotation = _rpy_matrix(*np.radians(geometry.rotate)) @ rotation
    transform = np.eye(4)
    transform[:3, :3] = rotation * geometry.unit_scale
    parts = []
    if isinstance(loaded, trimesh.Scene):
        if len(loaded.graph.nodes_geometry) > 1:  # an assembly: keep its parts for the mass
            from robomodel.mass_parts import scene_parts

            parts = scene_parts(loaded, transform)
        mesh = loaded.to_mesh() if len(loaded.geometry) else None
    else:
        mesh = loaded
    if not isinstance(mesh, trimesh.Trimesh) or len(mesh.faces) == 0:
        raise GeometryError(f"{geometry.file} contains no triangles")
    mesh.apply_transform(transform)
    mesh.metadata["cad_parts"] = parts
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


def _box(length: float, width: float, height: float) -> trimesh.Trimesh:
    return trimesh.creation.box([length, width, height])


WIGLEY_STATIONS = 161
WIGLEY_GIRTH = 81  # points down each side, keel to deck


def _wigley(length: float, beam: float, draft: float, depth: float) -> trimesh.Trimesh:
    """Wigley hull, the classic towing-tank test form; centred on its bounding box.

    Below the design waterline: C_B = 4/9, C_WP = 2/3, KB = 5T/8, BM = 3 B^2 / (35 T).
    """
    if depth < draft:
        raise GeometryError("wigley: depth (keel to deck) must be at least the draft")
    xs = np.linspace(-length / 2, length / 2, WIGLEY_STATIONS)
    # z down, 0 at the design waterline: keel at +draft, deck at draft - depth
    freeboard = depth - draft
    zs = np.concatenate(
        [
            np.linspace(-freeboard, 0.0, max(2, int(WIGLEY_GIRTH * freeboard / depth) + 1))[:-1],
            draft * np.sin(np.linspace(0.0, np.pi / 2, WIGLEY_GIRTH)),  # finer near the keel
        ]
    )
    width = 1 - (2 * xs / length) ** 2
    below = np.where(zs > 0, 1 - (zs / draft) ** 2, 1.0)
    half = beam / 2 * np.outer(width, below)  # (stations, girth)
    # one ring per station: starboard deck -> keel -> port deck (the deck closes the ring)
    ys = np.concatenate([half, -half[:, -2::-1]], axis=1)
    zz = np.concatenate([zs, zs[-2::-1]])
    n_ring = ys.shape[1]
    verts = np.stack([np.repeat(xs, n_ring), ys.ravel(), np.tile(zz, WIGLEY_STATIONS)], axis=1)
    faces = []
    for i in range(WIGLEY_STATIONS - 1):
        a, b = i * n_ring, (i + 1) * n_ring
        for j in range(n_ring):
            k = (j + 1) % n_ring
            faces += [[a + j, b + j, b + k], [a + j, b + k, a + k]]
    mesh = trimesh.Trimesh(verts, np.array(faces), process=False)
    mesh.merge_vertices(digits_vertex=12)
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.remove_unreferenced_vertices()
    trimesh.repair.fix_normals(mesh)
    mesh.apply_translation(-mesh.bounds.mean(axis=0))
    return mesh


_PRIMITIVE_BUILDERS = {
    "box": _box,
    "wigley": _wigley,
    "sphere": _sphere,
    "ellipsoid": _ellipsoid,
    "cylinder": _cylinder,
    "myring": _myring,
}


def _rpy_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Rotation by roll (x), then pitch (y), then yaw (z): R = Rz Ry Rx."""
    cr, sr, cp, sp, cy, sy = (
        np.cos(roll),
        np.sin(roll),
        np.cos(pitch),
        np.sin(pitch),
        np.cos(yaw),
        np.sin(yaw),
    )
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return rz @ ry @ rx


def tilt_from_body_x(hull) -> float:
    """Angle (deg) between the hull's long principal axis and body x."""
    inertia = np.asarray(hull.moment_inertia)
    values, vectors = np.linalg.eigh(inertia)
    axis = vectors[:, int(np.argmin(values))]  # long axis: smallest moment of inertia
    return float(np.degrees(np.arccos(min(1.0, abs(axis[0])))))


def _has_nested_bodies(mesh: trimesh.Trimesh) -> bool:
    """True if a closed body's bounding box lies inside another body's (a solid inside a
    solid, which would count its volume twice)."""
    if len(mesh.faces) > 2_000_000:
        return False
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    nf, nv = len(mesh.faces), len(mesh.vertices)
    rows = np.repeat(np.arange(nf), 3)
    graph = coo_matrix((np.ones(3 * nf), (rows, mesh.faces.ravel() + nf)), shape=(nf + nv,) * 2)
    n, labels = connected_components(graph, directed=False)
    labels = labels[:nf]
    if n - nv <= 1 and len(np.unique(labels)) <= 1:
        return False
    boxes = []
    for body in np.unique(labels):
        verts = mesh.vertices[np.unique(mesh.faces[labels == body])]
        boxes.append((verts.min(axis=0), verts.max(axis=0)))
    for i, (lo_i, hi_i) in enumerate(boxes):
        for j, (lo_j, hi_j) in enumerate(boxes):
            if i != j and np.all(lo_i >= lo_j) and np.all(hi_i <= hi_j):
                return True
    return False


MAX_CAPPED_HOLES = 50


def _cap_holes(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Close open boundary loops with a triangle fan from each loop's centre.

    CAD exports usually lose whole flat caps (an end of a tube, a lid); a fan closes those
    exactly and approximates gently curved ones.
    """
    import networkx as nx

    edges = mesh.edges_sorted
    unique, counts = np.unique(edges, axis=0, return_counts=True)
    boundary = unique[counts == 1]
    if len(boundary) == 0:
        return mesh
    graph = nx.Graph()
    graph.add_edges_from(map(tuple, boundary))
    loops = [c for c in nx.connected_components(graph) if len(c) >= 3]
    if not loops or len(loops) > MAX_CAPPED_HOLES:
        return mesh
    vertices = [mesh.vertices]
    faces = [mesh.faces]
    next_index = len(mesh.vertices)
    for loop in loops:
        sub = graph.subgraph(loop)
        try:
            cycle = [u for u, _ in nx.find_cycle(sub)]
        except nx.NetworkXNoCycle:
            continue
        if len(cycle) != len(loop):  # not a simple loop: leave it
            continue
        centre = mesh.vertices[cycle].mean(axis=0)
        vertices.append(centre[None, :])
        ring = np.array(cycle)
        fan = np.column_stack([np.roll(ring, -1), ring, np.full(len(ring), next_index)])
        faces.append(fan)
        next_index += 1
    capped = trimesh.Trimesh(np.vstack(vertices), np.vstack(faces), process=True)
    trimesh.repair.fix_winding(capped)
    trimesh.repair.fix_normals(capped)
    return capped if capped.is_watertight else mesh
