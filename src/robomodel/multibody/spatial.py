"""Spatial (6D) vector algebra, after Featherstone, "Rigid Body Dynamics Algorithms" (2008).

Motion vectors are [w; v] (angular first), force vectors [n; f]. A Plucker transform X
from frame A to frame B, where B's origin is at r (in A coordinates) and E rotates A
coordinates into B coordinates, is

    X = [[E, 0], [-E S(r), E]]        (motion vectors)
    X* = [[E, -E S(r)], [0, E]]       (force vectors, = X^-T)
"""

from __future__ import annotations

import numpy as np


def skew(v) -> np.ndarray:
    x, y, z = v
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def rot_axis(axis, angle: float) -> np.ndarray:
    """Rotation matrix (Rodrigues) turning vectors by `angle` about unit `axis`."""
    a = np.asarray(axis, dtype=float)
    a = a / np.linalg.norm(a)
    k = skew(a)
    return np.eye(3) + np.sin(angle) * k + (1 - np.cos(angle)) * (k @ k)


def rpy_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """URDF convention: R = Rz(yaw) Ry(pitch) Rx(roll), maps child into parent coordinates."""
    return rot_axis((0, 0, 1), yaw) @ rot_axis((0, 1, 0), pitch) @ rot_axis((1, 0, 0), roll)


def xform(rotation_child_in_parent: np.ndarray, r) -> np.ndarray:
    """Motion transform from a parent frame to a child frame at r (parent coordinates)
    whose axes are the columns of `rotation_child_in_parent`."""
    e = rotation_child_in_parent.T
    x = np.zeros((6, 6))
    x[:3, :3] = e
    x[3:, 3:] = e
    x[3:, :3] = -e @ skew(r)
    return x


def force_xform(x: np.ndarray) -> np.ndarray:
    """X* for a motion transform X."""
    return np.linalg.inv(x).T


def crm(v) -> np.ndarray:
    """Motion cross product matrix v x (.) for motion vectors."""
    w, u = v[:3], v[3:]
    out = np.zeros((6, 6))
    out[:3, :3] = skew(w)
    out[3:, :3] = skew(u)
    out[3:, 3:] = skew(w)
    return out


def crf(v) -> np.ndarray:
    """Force cross product matrix v x* (.) = -crm(v)^T."""
    return -crm(v).T


def spatial_inertia(mass: float, com, inertia_com) -> np.ndarray:
    """Spatial inertia about the frame origin of a body with mass, centre of mass `com`
    (frame coordinates) and inertia tensor `inertia_com` about the CoM."""
    c = skew(com)
    out = np.zeros((6, 6))
    out[:3, :3] = np.asarray(inertia_com) + mass * c @ c.T
    out[:3, 3:] = mass * c
    out[3:, :3] = mass * c.T
    out[3:, 3:] = mass * np.eye(3)
    return out
