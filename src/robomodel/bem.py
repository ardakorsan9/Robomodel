"""Added mass from a boundary element method (Capytaine), for any closed shape.

Strip theory needs a slender body and the frame method only knows the bounding ellipsoid.
A panel method solves the potential flow around the real surface, so it works for boxes,
catamarans, tori and any odd shape. The body is fully submerged in an unbounded fluid and
the added mass is the infinite-frequency limit, which is what a manoeuvring model needs.

Optional: pip install "robomodel[bem]" (Capytaine, fast-simplification).
Cost grows with the square of the panel count, so the hull is decimated to PANELS faces.
"""

from __future__ import annotations

import importlib.util
import logging

import numpy as np
import trimesh

from robomodel.parameter import Parameter, Source

PANELS = 2500
REL_SIGMA = 0.10  # 2500 panels read ~3 % high on ellipsoids, plus real-fluid effects
DOFS = ("Surge", "Sway", "Heave", "Roll", "Pitch", "Yaw")


def available() -> bool:
    return (importlib.util.find_spec("capytaine") is not None
            and importlib.util.find_spec("fast_simplification") is not None)


def surface_mesh(hull) -> trimesh.Trimesh:
    """A closed triangle surface of the hull: the mesh itself, or the outer faces of a
    voxel hull."""
    if isinstance(hull, trimesh.Trimesh):
        return hull
    from robomodel.voxel import VoxelHull

    if isinstance(hull, VoxelHull):
        return voxel_surface(hull)
    raise TypeError(f"cannot make a surface from {type(hull).__name__}")


def voxel_surface(hull) -> trimesh.Trimesh:
    """The exposed faces of a voxel hull as a closed, outward-facing triangle mesh."""
    grid = np.pad(hull.filled, 1)
    vertices, faces = [], []
    corners = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
                        [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]], float)
    # quad corner indices for each of the 6 neighbour directions, counter-clockwise from outside
    quads = {(-1, 0, 0): (0, 4, 7, 3), (1, 0, 0): (1, 2, 6, 5), (0, -1, 0): (0, 1, 5, 4),
             (0, 1, 0): (3, 7, 6, 2), (0, 0, -1): (0, 3, 2, 1), (0, 0, 1): (4, 5, 6, 7)}
    for direction, quad in quads.items():
        neighbour = np.roll(grid, shift=[-d for d in direction], axis=(0, 1, 2))
        exposed = np.argwhere(grid & ~neighbour)
        if len(exposed) == 0:
            continue
        base = len(vertices) and sum(len(v) for v in vertices)
        pts = (exposed[:, None, :] + corners[list(quad)][None, :, :]).reshape(-1, 3)
        vertices.append(pts)
        idx = base + np.arange(len(exposed))[:, None] * 4
        faces.append(np.concatenate([idx + [0, 1, 2], idx + [0, 2, 3]]))
    v = np.concatenate(vertices)
    mesh = trimesh.Trimesh(v, np.concatenate(faces), process=True)
    mesh.apply_scale(hull.pitch)
    mesh.apply_translation(hull.origin - hull.pitch)  # undo the padding
    return mesh


MAX_ASPECT = 12.0  # longest edge / sqrt(2 * area): above this a panel is a sliver
MAX_GRID_CELLS = 8_000_000  # memory: the smoothed float grid is 8 bytes per cell


def panel_aspect(mesh: trimesh.Trimesh) -> float:
    """95th percentile of a sliver measure over the faces (about 1.3 for equilateral)."""
    edges = np.linalg.norm(mesh.vertices[mesh.faces] - np.roll(mesh.vertices[mesh.faces], 1,
                                                               axis=1), axis=2)
    ratio = edges.max(axis=1) / np.sqrt(2 * np.maximum(mesh.area_faces, 1e-30))
    return float(np.percentile(ratio, 95))


def _decimate(mesh: trimesh.Trimesh, panels: int) -> trimesh.Trimesh:
    for _ in range(4):  # the decimator may stop early; repeat until small enough
        if len(mesh.faces) <= panels:
            break
        mesh = mesh.simplify_quadric_decimation(face_count=panels)
    return mesh


def bem_panels(hull, panels: int = PANELS) -> trimesh.Trimesh:
    """Even, near-equilateral panels for the BEM.

    1. If the CAD surface is closed and its triangles are not slivers, decimate it directly
       (keeps the exact shape; ellipsoids within ~3 % of Lamb).
    2. Otherwise (CAD slivers such as a cylinder side made of 1.6 m long triangles, voxel
       hulls, broken CAD): fill the hull on a voxel grid fine enough for its thinnest
       dimension, smooth lightly, take the level surface with marching cubes, rescale to
       the hull's own volume and decimate.
    """
    from robomodel.voxel import VoxelHull

    if isinstance(hull, trimesh.Trimesh) and hull.is_watertight:
        direct = _decimate(hull.copy(), panels)
        if len(direct.faces) <= 2 * panels and panel_aspect(direct) < MAX_ASPECT:
            if direct.volume < 0:
                direct.invert()
            return direct
    return _marching_panels(hull, panels) if not isinstance(hull, VoxelHull) else \
        _marching_panels(hull, panels, voxels=hull)


def _marching_panels(hull, panels: int, voxels=None) -> trimesh.Trimesh:
    from scipy import ndimage
    from skimage.measure import marching_cubes

    from robomodel.voxel import voxelize

    extents = np.asarray(hull.extents, dtype=float)
    if voxels is None:
        pitch = min(extents.max() / 160, extents.min() / 16)
        pitch = max(pitch, (np.prod(extents) / MAX_GRID_CELLS) ** (1 / 3))
        voxels = voxelize(surface_mesh(hull), pitch)
    field = ndimage.gaussian_filter(np.pad(voxels.filled, 2).astype(float), sigma=0.5)
    verts, faces, _, _ = marching_cubes(field, level=0.5)
    verts = (verts - 2 + 0.5) * voxels.pitch + voxels.origin  # voxel centres
    mesh = trimesh.Trimesh(verts, faces[:, ::-1], process=True)
    if mesh.volume < 0:
        mesh.invert()
    mesh = _decimate(mesh, panels)
    target = float(hull.volume)
    if mesh.volume > 0 and target > 0:  # keep the hull's own volume and centre
        mesh.apply_translation(-np.asarray(mesh.center_mass))
        mesh.apply_scale((target / mesh.volume) ** (1 / 3))
        mesh.apply_translation(np.asarray(hull.center_mass))
    return mesh


def bem_added_mass(hull, rho: float, centre, panels: int = PANELS) -> np.ndarray:
    """6x6 added mass about `centre` (positive), fully submerged, infinite frequency."""
    import capytaine as cpt

    logging.getLogger("capytaine").setLevel(logging.ERROR)
    surface = bem_panels(hull, panels)
    mesh = cpt.Mesh(vertices=np.asarray(surface.vertices), faces=np.asarray(surface.faces))
    body = cpt.FloatingBody(mesh=mesh, dofs=cpt.rigid_body_dofs(rotation_center=tuple(centre)))
    solver = cpt.BEMSolver()
    matrix = np.zeros((6, 6))
    for j, dof in enumerate(DOFS):
        problem = cpt.RadiationProblem(body=body, radiating_dof=dof, omega=np.inf,
                                       free_surface=np.inf, water_depth=np.inf, rho=rho)
        result = solver.solve(problem, keep_details=False)
        for i, other in enumerate(DOFS):
            matrix[i, j] = result.added_masses[other]
    matrix = 0.5 * (matrix + matrix.T)  # the panel method is symmetric up to round-off
    return matrix


def bem_parameter(hull, rho: float, centre) -> Parameter:
    m = bem_added_mass(hull, rho, centre)
    note = f"boundary element method (Capytaine), {PANELS} panels, fully submerged"
    return Parameter(m, np.abs(m) * REL_SIGMA, Source.BEM, "kg, kg m, kg m^2", note)
