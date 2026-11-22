"""Hydrodynamics of open-frame, box-like vehicles (ROVs).

Strip theory and Hoerner's formulas assume a slender body of revolution. A box-like ROV is
neither, so here:

* Added mass: the ellipsoid that encloses the vehicle (semi-axes = half its size along each
  body axis), exact for that ellipsoid (Lamb). An open frame lets water through, so the
  real values can be lower; the uncertainty is wide.
* Damping: bluff-body drag on the projected areas seen along each axis,
      X_u|u| = 1/2 rho C_D A_x,  ...
  and for rotations every projected cell adds drag at its own lever arm, e.g. for yaw
      N_r|r| = 1/2 rho C_D (sum over side-view cells |x|^3 dA + sum over front-view |y|^3 dA).
"""

from __future__ import annotations

import numpy as np

from robomodel.added_mass import ellipsoid_added_mass
from robomodel.damping import REF_RATE, REF_SPEED, Damping
from robomodel.parameter import Parameter, Source
from robomodel.voxel import VoxelHull

FRAME_CD = 1.1  # bluff body, rough (Hoerner: flat plates ~1.2, cylinders ~1.1)
QUAD_REL_SIGMA = 1.0
ADDED_MASS_REL_SIGMA = 0.5
# Which projection (looking along axis) and which coordinate gives the lever arm, for
# rotations about x (roll), y (pitch), z (yaw): (projection axis, index of the arm in the
# two remaining coordinates)
_ROTATION_TERMS = {
    3: ((2, 1), (1, 1)),  # roll: top view arm y, side view arm z
    4: ((2, 0), (0, 1)),  # pitch: top view arm x, front view arm z
    5: ((1, 0), (0, 0)),  # yaw: side view arm x, front view arm y
}


def frame_added_mass(hull, rho: float) -> Parameter:
    a, b, c = np.asarray(hull.extents) / 2
    diag = ellipsoid_added_mass(a, b, c, rho)
    note = (
        f"enclosing ellipsoid {2 * a:.2f} x {2 * b:.2f} x {2 * c:.2f} m (Lamb); an open "
        "frame can have less"
    )
    return Parameter(
        np.diag(diag), np.diag(diag) * ADDED_MASS_REL_SIGMA, Source.EMPIRICAL, "kg, kg m^2", note
    )


def frame_damping(hull: VoxelHull, rho: float, centre=(0.0, 0.0, 0.0)) -> Damping:
    """Drag about `centre` (the hull centre); the model moves it to the body origin."""
    centre = np.asarray(centre, dtype=float)
    k = 0.5 * rho * FRAME_CD
    quad = np.zeros(6)
    for axis in range(3):
        _, area = hull.projection(axis)
        quad[axis] = k * area.sum()
    for dof, terms in _ROTATION_TERMS.items():
        total = 0.0
        for proj_axis, arm_index in terms:
            centres, area = hull.projection(proj_axis)
            others = [a for a in range(3) if a != proj_axis]
            arm = centres[:, arm_index] - centre[others[arm_index]]  # about the hull centre
            total += np.sum(np.abs(arm) ** 3 * area)
        quad[dof] = k * total
    ref = np.array([REF_SPEED] * 3 + [REF_RATE] * 3)
    linear = quad * ref
    note = (
        f"open-frame drag: C_D = {FRAME_CD} on projected areas; frames with thrusters and "
        "cables often have more. Pool tests and robomodel calibrate are needed"
    )
    return Damping(
        linear=Parameter(
            linear,
            linear * 1.0,
            Source.ASSUMED,
            "N s/m, N m s/rad",
            f"linear = quadratic x {REF_SPEED} m/s (or {REF_RATE} rad/s), assumed",
        ),
        quadratic=Parameter(
            quad, quad * QUAD_REL_SIGMA, Source.EMPIRICAL, "N s^2/m^2, N m s^2/rad^2", note
        ),
        axial_cd=FRAME_CD,
    )
