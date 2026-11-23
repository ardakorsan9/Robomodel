"""Robot description file (YAML): links with their CAD, joints, and the root.

    name: arm
    category: multibody
    root: base
    floating: false              # true: the root moves freely (rover, drone, AUV)
    links:
      - name: base
        mass: 2.0
        geometry: {file: base.stl, unit: mm}        # inertia from the CAD (uniform density)
      - name: link1
        mass: 1.2
        geometry: {primitive: cylinder, length: 0.4, radius: 0.03,
                   pose: {xyz: [0.2, 0, 0], rpy: [0, 0, 0]}}   # where the shape sits
      - name: tool
        mass: 0.3
        com: [0.02, 0, 0]                            # or give the inertial data directly
        inertia: [1e-4, 1e-4, 1e-4]
    joints:
      - {name: j1, type: revolute, parent: base, child: link1, xyz: [0, 0, 0.1],
         rpy: [0, 0, 0], axis: [0, 0, 1], lower: -3.14, upper: 3.14, effort: 20}

Each link's frame is the joint frame (URDF convention). Mass comes from the user (weighed);
the CAD gives the shape, so the centre of mass and inertia assume uniform density.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from robomodel.geometry import build_mesh
from robomodel.multibody.robot import JOINT_TYPES, Joint, Link, Robot
from robomodel.multibody.spatial import rpy_matrix
from robomodel.vehicle import VehicleError, _geometry


def load_robot(path: str | Path) -> Robot:
    path = Path(path)
    with path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return parse_robot(data, base_dir=path.parent)


def _vec(value, label, problems, n=3, default=None):
    if value is None and default is not None:
        return np.asarray(default, dtype=float)
    try:
        arr = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        arr = None
    if arr is None or arr.shape != (n,) or not np.all(np.isfinite(arr)):
        problems.append(f"'{label}' must be {n} numbers, got {value!r}")
        return None
    return arr


def _link(raw, i, base_dir, problems) -> Link | None:
    if not isinstance(raw, dict) or "name" not in raw:
        problems.append(f"links[{i}] needs a name")
        return None
    name = str(raw["name"])
    mass = raw.get("mass")
    if isinstance(mass, bool) or not isinstance(mass, (int, float)) or mass <= 0:
        problems.append(f"links[{i}] ({name}): mass must be a positive number (kg)")
        return None
    mesh = None
    if "geometry" in raw:
        geo_raw = dict(raw["geometry"])
        pose = geo_raw.pop("pose", {}) or {}
        geometry = _geometry(geo_raw, base_dir, problems)
        xyz = _vec(pose.get("xyz"), f"links[{i}].geometry.pose.xyz", problems, default=(0, 0, 0))
        rpy = _vec(pose.get("rpy"), f"links[{i}].geometry.pose.rpy", problems, default=(0, 0, 0))
        if geometry is None or xyz is None or rpy is None:
            return None
        hull = build_mesh(geometry)
        rot = rpy_matrix(*rpy)
        com = rot @ np.asarray(hull.center_mass) + xyz
        inertia = rot @ (np.asarray(hull.moment_inertia) * mass / hull.volume) @ rot.T
        if hasattr(hull, "copy"):
            mesh = hull.copy()
            transform = np.eye(4)
            transform[:3, :3] = rot
            transform[:3, 3] = xyz
            mesh.apply_transform(transform)
        if "com" in raw or "inertia" in raw:
            problems.append(f"links[{i}] ({name}): give either geometry or com/inertia")
            return None
    else:
        com = _vec(raw.get("com"), f"links[{i}].com", problems, default=(0, 0, 0))
        inertia_raw = np.asarray(raw.get("inertia", [0, 0, 0]), dtype=float)
        if inertia_raw.shape == (3,):
            inertia = np.diag(inertia_raw)
        elif inertia_raw.shape == (3, 3):
            inertia = inertia_raw
        else:
            problems.append(f"links[{i}] ({name}): inertia must be [Ixx, Iyy, Izz] or 3x3")
            return None
        if com is None:
            return None
    return Link(name, float(mass), np.asarray(com, dtype=float), np.asarray(inertia), mesh)


def _joint(raw, i, problems) -> Joint | None:
    if not isinstance(raw, dict):
        problems.append(f"joints[{i}] must be a mapping")
        return None
    missing = [k for k in ("name", "type", "parent", "child") if k not in raw]
    if missing:
        problems.append(f"joints[{i}] is missing {missing}")
        return None
    if raw["type"] not in JOINT_TYPES:
        problems.append(f"joints[{i}].type must be one of {list(JOINT_TYPES)}")
        return None
    xyz = _vec(raw.get("xyz"), f"joints[{i}].xyz", problems, default=(0, 0, 0))
    rpy = _vec(raw.get("rpy"), f"joints[{i}].rpy", problems, default=(0, 0, 0))
    axis = _vec(raw.get("axis"), f"joints[{i}].axis", problems, default=(0, 0, 1))
    if axis is not None and np.linalg.norm(axis) < 1e-9:
        problems.append(f"joints[{i}].axis must not be zero")
        return None
    if any(v is None for v in (xyz, rpy, axis)):
        return None

    def num(key, default):
        value = raw.get(key, default)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            problems.append(f"joints[{i}].{key} must be a number")
            return default
        return float(value)

    return Joint(
        str(raw["name"]),
        raw["type"],
        str(raw["parent"]),
        str(raw["child"]),
        xyz,
        rpy,
        axis / np.linalg.norm(axis),
        num("lower", -np.inf),
        num("upper", np.inf),
        num("effort", np.inf),
        num("damping", 0.0),
        num("friction", 0.0),
    )


def parse_robot(data, base_dir: Path = Path(".")) -> Robot:
    problems: list[str] = []
    if not isinstance(data, dict):
        raise VehicleError(["file must contain a mapping"])
    links = [
        lk
        for i, raw in enumerate(data.get("links") or [])
        if (lk := _link(raw, i, base_dir, problems)) is not None
    ]
    joints = [
        j
        for i, raw in enumerate(data.get("joints") or [])
        if (j := _joint(raw, i, problems)) is not None
    ]
    if not links:
        problems.append("'links' is required")
    root = data.get("root", links[0].name if links else None)
    gravity = _vec(data.get("gravity"), "gravity", problems, default=(0, 0, -9.80665))
    if problems:
        raise VehicleError(problems)
    try:
        robot = Robot(
            links, joints, root=root, floating=bool(data.get("floating", False)), gravity=gravity
        )
    except ValueError as e:
        raise VehicleError([str(e)]) from e
    robot.name = str(data.get("name", "robot"))
    return robot
