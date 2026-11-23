"""Surface vessels: how the hull floats, and its hydrostatic restoring forces.

A boat floats where its weight equals the weight of the water it displaces and its centre of
gravity is straight above the centre of buoyancy. Both follow exactly from a closed CAD hull
and the mass properties, so this part of a surface-vessel model is as reliable as the CAD
and the centre of gravity (its height sets the stability, GM).

Waterline geometry: the water surface, seen from the body frame (x forward, y right, z down),
is the plane h(p) = n . p + z0 = 0, where n is the world "down" direction in body axes and z0
the depth of the body origin. Points with h > 0 are under water.

Submerged volume and centroid come from the divergence theorem with the field h n, whose
divergence is 1 and which vanishes on the water surface: only the wet part of the hull's
skin is needed, never the waterplane cap. With an edge-midpoint rule (exact for quadratics)
the results are exact for a closed triangle mesh.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import trimesh
from scipy.optimize import brentq

from robomodel.hydrostatics import GRAVITY
from robomodel.voxel import VoxelHull

GZ_STEP_DEG = 5.0
GZ_MAX_DEG = 90.0


class SurfaceError(ValueError):
    """The hull cannot float as described (too heavy, open mesh, ...)."""


def down_vector(phi: float, theta: float) -> np.ndarray:
    """World down (NED z) expressed in body axes for heel phi and trim theta."""
    return np.array([-np.sin(theta), np.cos(theta) * np.sin(phi), np.cos(theta) * np.cos(phi)])


def _wet_faces(mesh: trimesh.Trimesh, n: np.ndarray, z0: float):
    """Triangles of the hull skin below the plane h = 0 (cut exactly along it)."""
    verts, faces = trimesh.intersections.slice_faces_plane(
        mesh.vertices, mesh.faces, plane_normal=n, plane_origin=-z0 * n
    )[:2]
    return np.asarray(verts), np.asarray(faces)


def submerged(hull, n: np.ndarray, z0: float) -> tuple[float, np.ndarray]:
    """(volume m^3, centroid in body frame m) of the part of the hull under water."""
    if isinstance(hull, VoxelHull):
        c = hull._centres
        frac = np.clip((c @ n + z0) / hull.pitch + 0.5, 0.0, 1.0)  # share of each voxel
        vol = float(frac.sum()) * hull.pitch**3
        if vol <= 0:
            return 0.0, np.zeros(3)
        return vol, (frac @ c) / frac.sum()
    verts, faces = _wet_faces(hull, n, z0)
    if len(faces) == 0:
        return 0.0, np.zeros(3)
    tri = verts[faces]  # (k, 3, 3)
    area_vec = 0.5 * np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    flux = area_vec @ n  # n . N dA
    mids = 0.5 * (tri + np.roll(tri, -1, axis=1))  # edge midpoints, (k, 3, 3)
    h = mids @ n + z0  # (k, 3)
    vol = float(flux @ h.mean(axis=1))
    if vol <= 0:
        return 0.0, np.zeros(3)
    xh = (mids * h[:, :, None]).mean(axis=1)  # mean of x_k h, (k, 3)
    hh = 0.5 * (h**2).mean(axis=1)  # mean of h^2 / 2
    first = flux @ xh - n * (flux @ hh)
    return vol, first / vol


def _wrench(hull, z0, phi, theta, weight, r_g, rho):
    """Gravity + buoyancy on the vessel: force and moment about the origin, body axes."""
    n = down_vector(phi, theta)
    vol, r_b = submerged(hull, n, z0)
    b = rho * GRAVITY * vol
    force = (weight - b) * n
    moment = np.cross(r_g, weight * n) - np.cross(r_b, b * n)
    return np.concatenate([force, moment]), vol, r_b


@dataclass(frozen=True)
class Floating:
    """The vessel at rest and its hydrostatic stiffness."""

    depth: float  # z0: depth of the body origin below the water surface, m
    heel: float  # rad, positive: starboard down
    trim: float  # rad, positive: bow up
    volume: float  # displaced volume, m^3
    r_b: np.ndarray  # centre of buoyancy, body frame, m
    draft: float  # deepest point below the water, m
    freeboard: float  # highest point above the water, m
    g_matrix: np.ndarray  # 6x6: g(eta) = G (eta - eta_rest) for small motions
    gz_angle: np.ndarray  # heel angles, rad (absolute)
    roll_moment: np.ndarray  # roll restoring g_4 at those angles, N m (heave, trim free)
    weight: float  # N

    @property
    def eta_rest(self) -> np.ndarray:
        return np.array([0.0, 0.0, self.depth, self.heel, self.trim, 0.0])

    @property
    def waterplane_area(self) -> float:
        return float(self.g_matrix[2, 2] / (self.weight / self.volume))

    @property
    def gm_t(self) -> float:
        """Transverse metacentric height, m (> 0: stable in roll)."""
        return float(self.g_matrix[3, 3] / self.weight)

    @property
    def gm_l(self) -> float:
        """Longitudinal metacentric height, m."""
        return float(self.g_matrix[4, 4] / self.weight)

    @property
    def gz(self) -> np.ndarray:
        """Righting arm GZ(phi), m: the restoring moment divided by the weight."""
        return self.roll_moment / self.weight


def _rest_depth(hull, weight: float, rho: float, n: np.ndarray) -> float:
    lo, hi = _depth_range(hull, n)
    full = submerged(hull, n, hi)[0]
    if rho * GRAVITY * full < weight:
        raise SurfaceError(
            f"the hull cannot carry its weight: fully submerged it displaces {full:.4g} m^3 "
            f"({rho * full:.4g} kg), less than the mass {weight / GRAVITY:.4g} kg"
        )
    return brentq(lambda z0: rho * GRAVITY * submerged(hull, n, z0)[0] - weight, lo, hi, xtol=1e-12)


def _depth_range(hull, n: np.ndarray) -> tuple[float, float]:
    corners = np.array(
        [[x, y, z] for x in hull.bounds[:, 0] for y in hull.bounds[:, 1] for z in hull.bounds[:, 2]]
    )
    h = corners @ n
    return float(-h.max()), float(-h.min())


def _newton(residual, x0, steps, limits, tol=1e-12, iters=60) -> np.ndarray:
    """Damped Gauss-Newton with a central-difference Jacobian (few unknowns, smooth)."""
    x = np.asarray(x0, dtype=float)
    steps, limits = np.asarray(steps), np.asarray(limits)
    for _ in range(iters):
        r = np.asarray(residual(x))
        if np.max(np.abs(r)) < tol:
            return x
        jac = np.empty((len(r), len(x)))
        for j in range(len(x)):
            dx = np.zeros_like(x)
            dx[j] = steps[j]
            jac[:, j] = (np.asarray(residual(x + dx)) - np.asarray(residual(x - dx))) / (
                2 * steps[j]
            )
        delta = np.linalg.lstsq(jac, -r, rcond=None)[0]
        scale = np.max(np.abs(delta) / limits)
        x = x + (delta / scale if scale > 1 else delta)
    if np.max(np.abs(residual(x))) > 1e-6:
        raise SurfaceError("could not find a floating position (check the mesh and the CG)")
    return x


def _solve_rest(hull, weight, r_g, rho, length):
    """Depth, heel and trim at which forces and moments vanish."""
    z0 = _rest_depth(hull, weight, rho, down_vector(0.0, 0.0))

    def residual(x):
        f = _wrench(hull, x[0], x[1], x[2], weight, r_g, rho)[0]
        return np.concatenate([f[:3] / weight, f[3:] / (weight * length)])

    return _newton(residual, [z0, 0.0, 0.0], [1e-6 * length, 1e-5, 1e-5], [0.05 * length, 0.1, 0.1])


def _balanced_roll_moment(hull, phi, guess, weight, r_g, rho, length):
    """Roll restoring moment at heel phi with depth and trim free (constant displacement)."""

    def residual(x):
        f = _wrench(hull, x[0], phi, x[1], weight, r_g, rho)[0]
        return [f[2] / weight, f[4] / (weight * length)]

    x = _newton(residual, guess, [1e-6 * length, 1e-5], [0.05 * length, 0.1], tol=1e-10)
    f = _wrench(hull, x[0], phi, x[1], weight, r_g, rho)[0]
    return -f[3], x


def floating(hull, mass: float, r_g: np.ndarray, rho: float) -> Floating:
    """Rest position, restoring matrix and GZ curve of a floating hull."""
    if isinstance(hull, trimesh.Trimesh) and not hull.is_watertight:
        raise SurfaceError("a surface vessel needs a closed hull to find its waterline")
    weight = mass * GRAVITY
    r_g = np.asarray(r_g, dtype=float)
    length = float(np.max(hull.extents))
    z0, phi0, theta0 = _solve_rest(hull, weight, r_g, rho, length)
    _, vol, r_b = _wrench(hull, z0, phi0, theta0, weight, r_g, rho)

    # G = d g / d eta at rest, g = -(gravity + buoyancy); only z, phi, theta matter.
    g_matrix = np.zeros((6, 6))
    pitch = hull.pitch if isinstance(hull, VoxelHull) else 0.0
    steps = {
        2: max(1e-4 * length, pitch),
        3: max(1e-3, pitch / length),
        4: max(1e-3, pitch / length),
    }
    rest = np.array([z0, phi0, theta0])
    for col, step in steps.items():
        plus, minus = rest.copy(), rest.copy()
        plus[col - 2] += step
        minus[col - 2] -= step
        fp = _wrench(hull, *plus, weight, r_g, rho)[0]
        fm = _wrench(hull, *minus, weight, r_g, rho)[0]
        g_matrix[:, col] = -(fp - fm) / (2 * step)

    # Large heel: righting moment with heave and trim settling at each angle.
    angles = np.radians(np.arange(-GZ_MAX_DEG, GZ_MAX_DEG + 1e-9, GZ_STEP_DEG)) + phi0
    moments = np.zeros_like(angles)
    i0 = int(np.argmin(np.abs(angles - phi0)))
    for order in (range(i0, len(angles)), range(i0, -1, -1)):
        guess = [z0, theta0]
        for i in order:
            moments[i], guess = _balanced_roll_moment(
                hull, angles[i], guess, weight, r_g, rho, length
            )

    n = down_vector(phi0, theta0)
    corners_h = _hull_heights(hull, n, z0)
    return Floating(
        depth=float(z0),
        heel=float(phi0),
        trim=float(theta0),
        volume=vol,
        r_b=r_b,
        draft=float(corners_h.max()),
        freeboard=float(-corners_h.min()),
        g_matrix=g_matrix,
        gz_angle=angles,
        roll_moment=moments,
        weight=weight,
    )


def _hull_heights(hull, n, z0) -> np.ndarray:
    pts = hull._centres if isinstance(hull, VoxelHull) else hull.vertices
    return pts @ n + z0


def double_body(hull, fl: Floating):
    """The wet hull plus its mirror image in the water surface.

    Moving in infinite fluid, this body has (half of its added mass) the added mass of the
    floating hull at the usual manoeuvring limits: surge, sway and yaw at low frequency (the
    surface acts as a wall) and heave, roll and pitch at high frequency (the surface keeps
    zero pressure), because the mirror image moves along for the first and against for the
    second.
    """
    n = down_vector(fl.heel, fl.trim)
    z0 = fl.depth
    if isinstance(hull, VoxelHull):
        c = hull._centres
        wet = c[c @ n + z0 > 0]
        mirrored = wet - 2 * ((wet @ n + z0)[:, None]) * n
        pts = np.vstack([wet, mirrored])
        lo = pts.min(axis=0) - 0.5 * hull.pitch
        idx = np.floor((pts - lo) / hull.pitch).astype(int)
        grid = np.zeros(idx.max(axis=0) + 1, dtype=bool)
        grid[tuple(idx.T)] = True
        return VoxelHull(filled=grid, pitch=hull.pitch, origin=lo)
    verts, faces = _wet_faces(hull, n, z0)
    mirrored = verts - 2 * ((verts @ n + z0)[:, None]) * n
    both = trimesh.Trimesh(
        np.vstack([verts, mirrored]),
        np.vstack([faces, faces[:, ::-1] + len(verts)]),
        process=False,
    )
    both.merge_vertices(digits_vertex=9)
    both.update_faces(both.nondegenerate_faces())
    both.remove_unreferenced_vertices()
    if not both.is_watertight:
        from robomodel.voxel import voxelize

        return voxelize(both, float(max(both.extents)) / 100, close_gaps=True)
    trimesh.repair.fix_normals(both)
    return both


def natural_periods(m_total: np.ndarray, g_matrix: np.ndarray) -> dict[str, float]:
    """Undamped heave, roll and pitch periods, s (uncoupled)."""
    out = {}
    for name, i in (("heave", 2), ("roll", 3), ("pitch", 4)):
        k = g_matrix[i, i]
        out[name] = float(2 * np.pi * np.sqrt(m_total[i, i] / k)) if k > 0 else float("inf")
    return out
