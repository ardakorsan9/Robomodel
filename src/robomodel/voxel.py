"""Voxel hulls: turn a multi-part, open CAD assembly into one solid.

ROV CAD files are assemblies of hundreds of parts (frame plates, tubes, thrusters, foam).
Many are not closed and the air inside sealed tubes is not a part at all, so the mesh has
no single volume. We sample points on every surface, mark the voxels they fall in, and fill
the enclosed cavities (sealed tubes). The result behaves like a trimesh for the rest of
robomodel: volume, centroid, inertia, extents, bounds.

Accuracy: thin plates are thickened to at least one voxel, so the volume comes out high
(BlueROV2 Heavy, 4 mm voxels: +36 %); weigh the vehicle in water (measured.volume) when the
buoyancy matters. Projected areas are good (BlueROV2: within 7 % of the paper's CAD values).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import trimesh
from scipy import ndimage

POINTS_PER_VOXEL = 30  # surface samples per voxel face area (few enough holes to fill)
MAX_POINTS = 30_000_000
CHUNK = 4_000_000


@dataclass
class VoxelHull:
    """A solid made of cubic voxels of size `pitch`; grid axes are the body axes."""

    filled: np.ndarray  # bool (nx, ny, nz)
    pitch: float  # m
    origin: np.ndarray  # body position of the corner of voxel (0, 0, 0), m
    metadata: dict = field(default_factory=dict)
    is_watertight: bool = True

    @property
    def _centres(self) -> np.ndarray:
        idx = np.argwhere(self.filled)
        return self.origin + (idx + 0.5) * self.pitch

    @property
    def volume(self) -> float:
        return float(self.filled.sum()) * self.pitch**3

    @property
    def center_mass(self) -> np.ndarray:
        return self._centres.mean(axis=0)

    @property
    def moment_inertia(self) -> np.ndarray:
        """Inertia about the centroid for density 1 (like trimesh)."""
        r = self._centres - self.center_mass
        dv = self.pitch**3
        own = self.pitch**2 / 6  # a cube's own inertia per unit mass, every axis
        sq = np.einsum("ij,ij->i", r, r)
        tensor = (sq.sum() * np.eye(3) - r.T @ r) * dv
        return tensor + own * self.volume * np.eye(3)

    @property
    def bounds(self) -> np.ndarray:
        idx = np.argwhere(self.filled)
        lo = self.origin + idx.min(axis=0) * self.pitch
        hi = self.origin + (idx.max(axis=0) + 1) * self.pitch
        return np.array([lo, hi])

    @property
    def extents(self) -> np.ndarray:
        b = self.bounds
        return b[1] - b[0]

    def apply_translation(self, offset) -> None:
        self.origin = self.origin + np.asarray(offset, dtype=float)

    def projection(self, axis: int) -> tuple[np.ndarray, np.ndarray]:
        """Voxels seen looking along body `axis`: (centres of the 2 other coordinates, area
        of each cell). Used for drag areas and their moment arms."""
        mask = self.filled.any(axis=axis)
        others = [a for a in range(3) if a != axis]
        idx = np.argwhere(mask)
        centres = self.origin[others] + (idx + 0.5) * self.pitch
        return centres, np.full(len(idx), self.pitch**2)


def voxelize(
    mesh: trimesh.Trimesh, pitch: float, seed: int = 0, close_gaps: bool = False
) -> VoxelHull:
    """Solid voxel hull of any surface mesh (open, multi-part or closed).

    close_gaps: the mesh should be one closed body but has holes (a broken export): close
    gaps of a few voxels before filling. Leave it off for real open frames (ROVs), whose
    openings must stay open.
    """
    # Empty layers around the part: binary closing erodes anything touching the array edge.
    # Room for the harder closing below. The half-voxel shift keeps flat faces that sit
    # exactly on the bounding box from landing on a voxel border (a doubled shell layer).
    margin = 8 * pitch
    lo = mesh.bounds[0] - margin + 0.5 * pitch
    shape = np.ceil((mesh.bounds[1] + margin - lo) / pitch).astype(int) + 1
    surface = np.zeros(shape, dtype=bool)
    n = int(min(MAX_POINTS, max(100_000, POINTS_PER_VOXEL * mesh.area / pitch**2)))
    rng_seed = seed
    remaining = n
    while remaining > 0:
        k = min(CHUNK, remaining)
        points, _ = trimesh.sample.sample_surface(mesh, k, seed=rng_seed)
        idx = np.floor((points - lo) / pitch).astype(int)
        surface[tuple(np.clip(idx, 0, shape - 1).T)] = True
        remaining -= k
        rng_seed += 1
    # Close pinholes between samples before filling sealed cavities. If nothing gets filled,
    # the surface has bigger gaps (an open CAD export): close harder, up to a few voxels.
    filled = surface
    for iterations in (1, 2, 4, 6) if close_gaps else (1,):
        closed = ndimage.binary_closing(surface, iterations=iterations)
        filled = ndimage.binary_fill_holes(closed)
        if filled.sum() > 1.5 * closed.sum():
            break
    return VoxelHull(filled=filled, pitch=pitch, origin=lo)
