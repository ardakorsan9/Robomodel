"""Robots drawn as an assembly: every link's CAD exported in the same (assembly) coordinates.

That is how CAD tools export parts of an assembly at its zero pose, and it is what the web
page asks for. The user clicks the joints on the assembly. Here this is turned into the robot
description (description.py), whose link frames sit at the joints (URDF convention):

    link frame origin o_L = the position of the joint that moves link L (root: assembly origin)
    joint xyz        = o_child - o_parent        (rpy = 0: frames stay parallel)
    link mesh pose   = -o_L                       (the CAD stays where it was drawn)

Positions and axes are in the robot frame: x forward, y left, z up (ROS / URDF).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from robomodel.multibody.description import parse_robot
from robomodel.multibody.robot import Robot
from robomodel.vehicle import VehicleError

HOLD_SAMPLES = 400  # random poses for the worst-case holding torque


def description_from_assembly(spec: dict, files: dict[str, Path]) -> dict:
    """spec: {name, root?, floating?, links: [{name, file, mass, unit?, axes?}],
    joints: [{name, type, parent, child, position, axis, lower?, upper?, effort?, damping?}]}
    with `file` a key of `files` and positions in metres in assembly coordinates."""
    problems = []
    links = spec.get("links") or []
    joints = spec.get("joints") or []
    if not links:
        raise VehicleError(["add at least one link (one CAD file per moving part)"])
    names = [str(lk.get("name")) for lk in links]
    if len(set(names)) != len(names):
        problems.append("link names must be different")
    root = str(spec.get("root") or names[0])
    moved_by = {}
    for j in joints:
        child = str(j.get("child"))
        if child in moved_by:
            problems.append(f"link {child} is moved by two joints; each link has one parent")
        moved_by[child] = j
    for lk in names:
        if lk != root and lk not in moved_by:
            problems.append(f"link {lk} is not connected: add a joint with child {lk}")
    if problems:
        raise VehicleError(problems)

    def origin(name: str) -> np.ndarray:
        if name == root or name not in moved_by:
            return np.zeros(3)
        return np.asarray(moved_by[name].get("position", [0, 0, 0]), dtype=float)

    out_links = []
    for lk in links:
        key = lk.get("file")
        if key not in files:
            raise VehicleError([f"link {lk.get('name')}: its CAD file is missing"])
        geometry = {
            "file": str(files[key]),
            "unit": lk.get("unit", "m"),
            "axes": lk.get("axes", ["x", "y", "z"]),
            "pose": {"xyz": (-origin(str(lk["name"]))).tolist()},
        }
        out_links.append({"name": str(lk["name"]), "mass": lk.get("mass"), "geometry": geometry})
    out_joints = []
    for j in joints:
        parent, child = str(j.get("parent")), str(j.get("child"))
        entry = {
            "name": str(j.get("name")),
            "type": j.get("type", "revolute"),
            "parent": parent,
            "child": child,
            "xyz": (origin(child) - origin(parent)).tolist(),
            "axis": j.get("axis", [0, 0, 1]),
        }
        for key in ("lower", "upper", "effort", "damping", "friction"):
            if j.get(key) is not None:
                entry[key] = j[key]
        out_joints.append(entry)
    return {
        "name": spec.get("name", "robot"),
        "category": "multibody",
        "root": root,
        "floating": bool(spec.get("floating", False)),
        "links": out_links,
        "joints": out_joints,
    }


def robot_from_assembly(spec: dict, files: dict[str, Path]) -> Robot:
    return parse_robot(description_from_assembly(spec, files))


def _pose_samples(robot: Robot, n: int, rng) -> np.ndarray:
    lows, highs = [], []
    for name in robot.dof_names:
        j = next((j for j in robot.joints if j.name == name), None)
        if j is None:  # floating-base coordinate
            lows.append(0.0)
            highs.append(0.0)
            continue
        if j.type == "prismatic":
            lo = j.lower if np.isfinite(j.lower) else -0.1
            hi = j.upper if np.isfinite(j.upper) else 0.1
        else:
            lo = max(j.lower, -np.pi) if np.isfinite(j.lower) else -np.pi
            hi = min(j.upper, np.pi) if np.isfinite(j.upper) else np.pi
        lows.append(lo)
        highs.append(hi)
    return rng.uniform(lows, highs, size=(n, robot.n))


def holding_torques(robot: Robot, samples: int = HOLD_SAMPLES) -> dict[str, dict[str, float]]:
    """Gravity torque (N m, or N for sliders) each joint motor must hold: at the zero pose and
    the largest over random poses within the joint limits. A first motor-sizing number."""
    if robot.floating or robot.n == 0:
        return {}
    rng = np.random.default_rng(0)
    zero = np.zeros(robot.n)
    g0 = robot.bias(zero, zero)
    poses = _pose_samples(robot, samples, rng)
    worst = np.max(np.abs([robot.bias(q, zero) for q in poses]), axis=0)
    out = {}
    for i, name in enumerate(robot.dof_names):
        j = next((j for j in robot.joints if j.name == name), None)
        out[name] = {
            "zero_pose": float(g0[i]),
            "worst": float(max(worst[i], abs(g0[i]))),
            "effort": float(j.effort) if j is not None and np.isfinite(j.effort) else None,
        }
    return out
