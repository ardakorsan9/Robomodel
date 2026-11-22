"""Slice a hull into cross-sections along body x.

Strip theory (added mass) and the damping estimate both work section by section.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import trimesh
from shapely.geometry import MultiLineString
from shapely.ops import polygonize, unary_union

DEFAULT_STATIONS = 120
_ROUND_DECIMALS = 9  # snap segment end points (m) so shapely can join them into polygons


@dataclass(frozen=True)
class Sections:
    """Cross-sections at stations x (midpoints of equal slices of width dx)."""

    x: np.ndarray  # station positions, m
    dx: float  # slice width, m
    half_width: np.ndarray  # half of the y extent (bw), m
    half_height: np.ndarray  # half of the z extent (bh), m
    area: np.ndarray  # cross-section area, m^2
    length: float  # hull length along x, m

    def integrate(self, values: np.ndarray) -> float:
        """Midpoint rule over the hull length."""
        return float(np.sum(values) * self.dx)


def voxel_sections(hull) -> Sections:
    """Sections of a VoxelHull: one per voxel layer along x."""
    grid = hull.filled
    keep = np.flatnonzero(grid.any(axis=(1, 2)))
    grid = grid[keep[0] : keep[-1] + 1]
    n = grid.shape[0]
    half_w, half_h, area = np.zeros(n), np.zeros(n), np.zeros(n)
    for i in range(n):
        layer = grid[i]
        if not layer.any():
            continue
        ys = np.flatnonzero(layer.any(axis=1))
        zs = np.flatnonzero(layer.any(axis=0))
        half_w[i] = (ys[-1] - ys[0] + 1) * hull.pitch / 2
        half_h[i] = (zs[-1] - zs[0] + 1) * hull.pitch / 2
        area[i] = layer.sum() * hull.pitch**2
    xs = hull.origin[0] + (keep[0] + np.arange(n) + 0.5) * hull.pitch
    return Sections(xs, hull.pitch, half_w, half_h, area, n * hull.pitch)


def slice_hull(mesh: trimesh.Trimesh, stations: int = DEFAULT_STATIONS) -> Sections:
    if hasattr(mesh, "filled"):  # VoxelHull
        return voxel_sections(mesh)
    x_min, x_max = mesh.bounds[:, 0]
    length = float(x_max - x_min)
    dx = length / stations
    xs = x_min + dx * (np.arange(stations) + 0.5)

    half_w = np.zeros(stations)
    half_h = np.zeros(stations)
    area = np.zeros(stations)
    for i, x in enumerate(xs):
        segments = trimesh.intersections.mesh_plane(mesh, [1.0, 0.0, 0.0], [x, 0.0, 0.0])
        if len(segments) == 0:
            continue
        yz = np.round(segments[:, :, 1:], _ROUND_DECIMALS)
        half_w[i] = (yz[:, :, 0].max() - yz[:, :, 0].min()) / 2
        half_h[i] = (yz[:, :, 1].max() - yz[:, :, 1].min()) / 2
        polygons = list(polygonize(MultiLineString([tuple(map(tuple, s)) for s in yz])))
        if polygons:
            area[i] = unary_union(polygons).area
    return Sections(xs, dx, half_w, half_h, area, length)
