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


def slice_hull(mesh: trimesh.Trimesh, stations: int = DEFAULT_STATIONS) -> Sections:
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
