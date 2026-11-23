"""Vehicles made of several separate hulls (catamarans, twin torpedoes, pontoons).

Each hull gets its own strip-theory added mass, damping and body lift about its own centre.
They are then added up about the vehicle's hull centre. The interaction between the hulls
(blockage, wake shielding) is ignored; with BEM installed, added mass is solved for all
hulls together instead, which includes it.

Combining drag: a hull at offset d from the centre adds, for a rotation rate w, a local
speed w * d. With quadratic drag that gives a moment d * c (w d)|w d| = c |d|^3 w|w|, so
e.g. the yaw coefficient gains d_quad_surge * |d_y|^3 + d_quad_sway * |d_x|^3 (linear drag:
squares). This keeps the damping diagonal.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import trimesh

from robomodel.damping import Damping
from robomodel.parameter import Parameter

MIN_SHARE = 0.05  # a body smaller than this share of the volume is not a separate hull

# rotation dof -> (translation dof, offset axis) pairs that add to it
_LEVER = {
    3: ((1, 2), (2, 1)),  # roll: sway drag at z offset, heave drag at y offset
    4: ((0, 2), (2, 0)),  # pitch: surge drag at z offset, heave drag at x offset
    5: ((0, 1), (1, 0)),  # yaw: surge drag at y offset, sway drag at x offset
}


def split_bodies(mesh: trimesh.Trimesh, min_share: float = MIN_SHARE) -> list[trimesh.Trimesh]:
    """The separate closed bodies holding at least min_share of the volume."""
    if len(mesh.faces) > 2_000_000:
        return [mesh]
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    nf, nv = len(mesh.faces), len(mesh.vertices)
    rows = np.repeat(np.arange(nf), 3)
    graph = coo_matrix((np.ones(3 * nf), (rows, mesh.faces.ravel() + nf)), shape=(nf + nv,) * 2)
    _, labels = connected_components(graph, directed=False)
    labels = labels[:nf]
    bodies = []
    for label in np.unique(labels):
        body = mesh.submesh([np.flatnonzero(labels == label)], append=True)
        if body.is_watertight and abs(body.volume) > 0:
            if body.volume < 0:
                body.invert()
            bodies.append(body)
    total = sum(b.volume for b in bodies)
    big = [b for b in bodies if b.volume >= min_share * total]
    return big if len(big) > 1 else [mesh]


def combine_damping(parts: list[tuple[Damping, np.ndarray]], centre) -> Damping:
    """Add up per-hull damping (each about its own centre) about `centre`."""
    centre = np.asarray(centre, dtype=float)
    lin = np.zeros(6)
    quad = np.zeros(6)
    for damping, hull_centre in parts:
        d = np.asarray(hull_centre, dtype=float) - centre
        dl = np.asarray(damping.linear.value, dtype=float)
        dq = np.asarray(damping.quadratic.value, dtype=float)
        lin += dl
        quad += dq
        for rot, terms in _LEVER.items():
            for trans, axis in terms:
                lin[rot] += dl[trans] * d[axis] ** 2
                quad[rot] += dq[trans] * abs(d[axis]) ** 3
    first = parts[0][0]
    note = f"{len(parts)} separate hulls, each estimated on its own and added up"
    return replace(
        first,
        linear=Parameter(
            lin,
            lin * (first.linear.sigma / np.maximum(first.linear.value, 1e-30)).max(),
            first.linear.source,
            first.linear.unit,
            first.linear.note,
        ),
        quadratic=Parameter(
            quad,
            np.abs(quad) * 1.0,
            first.quadratic.source,
            first.quadratic.unit,
            first.quadratic.note + "; " + note,
        ),
    )
