"""Multibody dynamics against analytic results: pendulum, 2-link arm, energy, free fall."""

import numpy as np
import pytest

from robomodel.multibody import Joint, Link, Robot

G = 9.80665


def rod(name, mass, length, along=(1, 0, 0)):
    """A thin rod from the link origin along `along`."""
    a = np.asarray(along, dtype=float)
    i = mass * length**2 / 12
    inertia = i * (np.eye(3) - np.outer(a, a)) + 1e-9 * np.eye(3)
    return Link(name, mass, a * length / 2, inertia)


def base():
    return Link("base", 1.0, np.zeros(3), 1e-3 * np.eye(3))


def pendulum(m=2.0, length=1.0):
    return Robot(
        [base(), rod("rod", m, length, (0, 0, -1))],
        [Joint("hinge", "revolute", "base", "rod", axis=np.array([0, 1.0, 0]))],
        root="base",
    )


def arm(m1=2.0, m2=1.5, l1=0.8, l2=0.6, gravity=(0, 0, 0)):
    """Planar 2-link arm in the x-y plane, joints about z."""
    return Robot(
        [base(), rod("l1", m1, l1), rod("l2", m2, l2)],
        [
            Joint("j1", "revolute", "base", "l1", axis=np.array([0, 0, 1.0])),
            Joint(
                "j2", "revolute", "l1", "l2", xyz=np.array([l1, 0, 0]), axis=np.array([0, 0, 1.0])
            ),
        ],
        root="base",
        gravity=gravity,
    )


def zero_crossings(t, x):
    i = np.flatnonzero((x[:-1] > 0) & (x[1:] <= 0))
    return t[i] + (t[i + 1] - t[i]) * x[i] / (x[i] - x[i + 1])  # linear interpolation


def test_pendulum_inertia_and_period():
    m, length = 2.0, 1.0
    r = pendulum(m, length)
    assert r.mass_matrix([0.0])[0, 0] == pytest.approx(m * length**2 / 3)
    t, q, _ = r.simulate(12.0, q0=[0.02], dt_out=0.002)
    period = np.mean(np.diff(zero_crossings(t, q[:, 0])))
    expected = 2 * np.pi * np.sqrt((m * length**2 / 3) / (m * G * length / 2))
    assert period == pytest.approx(expected, rel=1e-3)  # small-angle formula


@pytest.mark.parametrize("q2", [0.0, 0.7, 2.0])
def test_two_link_mass_matrix_is_analytic(q2):
    m1, m2, l1, l2 = 2.0, 1.5, 0.8, 0.6
    i1, i2, c1, c2 = m1 * l1**2 / 12, m2 * l2**2 / 12, l1 / 2, l2 / 2
    expected = np.array([
        [i1 + i2 + m1 * c1**2 + m2 * (l1**2 + c2**2 + 2 * l1 * c2 * np.cos(q2)),
         i2 + m2 * (c2**2 + l1 * c2 * np.cos(q2))],
        [i2 + m2 * (c2**2 + l1 * c2 * np.cos(q2)), i2 + m2 * c2**2],
    ])  # fmt: skip
    np.testing.assert_allclose(arm().mass_matrix([0.3, q2]), expected, rtol=1e-6)


def test_inverse_and_forward_dynamics_agree():
    r = arm(gravity=(0, -G, 0))
    rng = np.random.default_rng(0)
    for _ in range(5):
        q, qd, qdd = rng.normal(size=(3, 2))
        tau = r.inverse_dynamics(q, qd, qdd)
        np.testing.assert_allclose(r.mass_matrix(q) @ qdd + r.bias(q, qd), tau, atol=1e-9)
        np.testing.assert_allclose(r.forward_dynamics(q, qd, tau), qdd, atol=1e-9)


def test_double_pendulum_conserves_energy():
    r = Robot(
        [base(), rod("a", 1.0, 0.5, (0, 0, -1)), rod("b", 0.7, 0.4, (0, 0, -1))],
        [
            Joint("j1", "revolute", "base", "a", axis=np.array([0, 1.0, 0])),
            Joint(
                "j2", "revolute", "a", "b", xyz=np.array([0, 0, -0.5]), axis=np.array([0, 1.0, 0])
            ),
        ],
        root="base",
    )
    t, q, qd = r.simulate(10.0, q0=[1.2, -0.8], dt_out=0.05)
    energy = np.array([sum(r.energy(a, b)) for a, b in zip(q, qd, strict=True)])
    assert np.ptp(energy) / abs(energy[0]) < 1e-6  # chaotic motion, energy still constant


def test_forward_kinematics():
    r = arm()
    pose = r.link_transform([np.pi / 2, -np.pi / 2], "l2")
    np.testing.assert_allclose(pose[:3, 3], [0, 0.8, 0], atol=1e-12)  # elbow at (0, l1)
    np.testing.assert_allclose(pose[:3, :3], np.eye(3), atol=1e-12)  # angles cancel


def test_fixed_joint_merges_inertia():
    """A tool fixed to the arm tip changes M like a point mass there."""
    tool = Link("tool", 0.5, np.zeros(3), 1e-9 * np.eye(3))
    plain = arm()
    with_tool = Robot(
        list(plain.links.values()) + [tool],
        plain.joints + [Joint("mount", "fixed", "l2", "tool", xyz=np.array([0.6, 0, 0]))],
        root="base",
        gravity=(0, 0, 0),
    )
    assert with_tool.n == 2
    q = [0.0, 0.0]  # stretched out: tool at 1.4 m from j1, 0.6 m from j2
    diff = with_tool.mass_matrix(q) - plain.mass_matrix(q)
    np.testing.assert_allclose(
        diff, 0.5 * np.array([[1.4**2, 1.4 * 0.6], [1.4 * 0.6, 0.6**2]]), atol=1e-6
    )


def test_floating_base_falls_freely_and_keeps_momentum():
    body = Link("body", 5.0, np.array([0.1, 0, 0]), np.diag([0.2, 0.3, 0.4]))
    r = Robot([body], [], root="body", floating=True)
    assert r.n == 6
    qdd = r.forward_dynamics(np.zeros(6), np.zeros(6), np.zeros(6))
    np.testing.assert_allclose(qdd[:3], [0, 0, -G], atol=1e-9)  # free fall
    np.testing.assert_allclose(qdd[3:], 0, atol=1e-9)  # no torque about the CoM
    space = Robot([body], [], root="body", floating=True, gravity=(0, 0, 0))
    t, q, qd = space.simulate(3.0, q0=np.zeros(6), qd0=[0.2, 0, 0, 0.3, 0.1, 0.5], dt_out=0.1)
    energy = np.array([space.energy(a, b)[0] for a, b in zip(q, qd, strict=True)])
    assert np.ptp(energy) / energy[0] < 1e-6  # tumbling body keeps its kinetic energy


def test_joint_friction_removes_energy():
    r = pendulum()
    j = r.joints[0]
    damped = Robot(
        list(r.links.values()),
        [Joint(j.name, j.type, j.parent, j.child, axis=j.axis, damping=0.5)],
        root="base",
    )
    t, q, qd = damped.simulate(5.0, q0=[0.5], dt_out=0.05)
    energy = np.array([sum(damped.energy(a, b)) for a, b in zip(q, qd, strict=True)])
    assert np.all(np.diff(energy) <= 1e-9)


def test_bad_trees_are_rejected():
    with pytest.raises(ValueError, match="not connected"):
        Robot([base(), rod("x", 1, 1)], [], root="base")
    with pytest.raises(ValueError, match="type"):
        Robot([base(), rod("x", 1, 1)], [Joint("j", "ball", "base", "x")], root="base")


# --- Robot description file, URDF, CLI -----------------------------------------------

from pathlib import Path  # noqa: E402

EXAMPLES = Path(__file__).parent.parent / "examples"


def test_cad_inertia_matches_analytic_rods():
    from robomodel.multibody.description import load_robot

    r = load_robot(EXAMPLES / "arm2.yaml")
    m1, l1, r1, m2, l2, r2 = 2.0, 0.8, 0.025, 1.5, 0.6, 0.02
    i1 = m1 * (3 * r1**2 + l1**2) / 12
    i2 = m2 * (3 * r2**2 + l2**2) / 12
    m11 = i1 + i2 + m1 * (l1 / 2) ** 2 + m2 * (l1**2 + (l2 / 2) ** 2 + l1 * l2)
    m22 = i2 + m2 * (l2 / 2) ** 2
    m = r.mass_matrix([0.0, 0.0])
    assert m[0, 0] == pytest.approx(m11, rel=1e-3)
    assert m[1, 1] == pytest.approx(m22, rel=1e-3)


def test_urdf_is_valid_and_complete(tmp_path):
    import subprocess
    import xml.etree.ElementTree as ET

    from robomodel.multibody.description import load_robot
    from robomodel.multibody.urdf import write_urdf

    path = write_urdf(load_robot(EXAMPLES / "arm2.yaml"), tmp_path)
    root = ET.parse(path).getroot()
    links = {e.get("name") for e in root.iter("link")}
    assert links == {"world", "base", "upper", "fore"}
    masses = {
        e.get("name"): float(e.find("inertial/mass").get("value"))
        for e in root.iter("link")
        if e.find("inertial") is not None
    }
    assert masses == {"base": 3.0, "upper": 2.0, "fore": 1.5}
    assert (tmp_path / "meshes" / "upper.stl").exists()
    elbow = [j for j in root.iter("joint") if j.get("name") == "elbow"][0]
    assert elbow.find("limit").get("lower") == "-2.5"
    ros = Path("/opt/ros/jazzy/setup.bash")
    if ros.exists():  # Gazebo installed with ROS 2: it must accept the file
        done = subprocess.run(
            ["bash", "-c", f"source {ros} && gz sdf -p {path}"],
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert done.returncode == 0 and "<model" in done.stdout, done.stdout[-500:]


def test_robot_file_errors_are_listed():
    from robomodel.multibody.description import parse_robot
    from robomodel.vehicle import VehicleError

    with pytest.raises(VehicleError) as e:
        parse_robot(
            {
                "links": [{"name": "a", "mass": -1}],
                "joints": [{"name": "j", "type": "hinge", "parent": "a", "child": "b"}],
            }
        )
    text = str(e.value)
    assert "mass" in text and "type" in text


def test_robot_cli(capsys, tmp_path):
    from robomodel.cli import main

    assert main(["robot", str(EXAMPLES / "arm2.yaml"), "-o", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "2 degrees of freedom" in out and "[FAIL]" not in out
    assert (tmp_path / "arm2.urdf").exists()
