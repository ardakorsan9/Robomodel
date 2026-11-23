"""Jointed robots built from per-link CAD in assembly coordinates (the web page's way)."""

import io
import zipfile
from pathlib import Path

import numpy as np
import pytest
import trimesh

from robomodel.multibody.assembly import holding_torques, robot_from_assembly
from robomodel.multibody.description import load_robot
from robomodel.vehicle import VehicleError

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def _cylinder_x(length, radius, centre_x):
    mesh = trimesh.creation.cylinder(radius=radius, height=length, sections=256)
    mesh.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [0, 1, 0]))
    mesh.apply_translation([centre_x, 0, 0])
    return mesh


@pytest.fixture
def arm_files(tmp_path):
    """examples/arm2.yaml drawn as an assembly at its zero pose, in mm."""
    parts = {
        "base": trimesh.creation.cylinder(radius=0.08, height=0.1, sections=256),
        "upper": _cylinder_x(0.8, 0.025, 0.4),
        "fore": _cylinder_x(0.6, 0.02, 1.1),
    }
    parts["base"].apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [0, 1, 0]))
    files = {}
    for name, mesh in parts.items():
        mesh.apply_scale(1000)
        path = tmp_path / f"{name}.stl"
        mesh.export(path)
        files[name] = path
    return files


SPEC = {
    "name": "arm2",
    "links": [
        {"name": "base", "file": "base", "mass": 3.0, "unit": "mm"},
        {"name": "upper", "file": "upper", "mass": 2.0, "unit": "mm"},
        {"name": "fore", "file": "fore", "mass": 1.5, "unit": "mm"},
    ],
    "joints": [
        {
            "name": "shoulder",
            "type": "revolute",
            "parent": "base",
            "child": "upper",
            "position": [0, 0, 0],
            "axis": [0, 1, 0],
            "lower": -3.14,
            "upper": 3.14,
            "effort": 50,
            "damping": 0.05,
        },
        {
            "name": "elbow",
            "type": "revolute",
            "parent": "upper",
            "child": "fore",
            "position": [0.8, 0, 0],
            "axis": [0, 1, 0],
            "lower": -2.5,
            "upper": 2.5,
            "effort": 30,
            "damping": 0.05,
        },
    ],
}


def test_assembly_gives_the_same_robot_as_the_urdf_style_file(arm_files):
    ours = robot_from_assembly(SPEC, arm_files)
    ref = load_robot(EXAMPLES / "arm2.yaml")
    rng = np.random.default_rng(3)
    for _ in range(5):
        q = rng.uniform(-2, 2, 2)
        assert np.allclose(ours.mass_matrix(q), ref.mass_matrix(q), rtol=1e-3)
        assert np.allclose(ours.bias(q, np.zeros(2)), ref.bias(q, np.zeros(2)), rtol=1e-3)


def test_holding_torque_of_a_horizontal_arm(arm_files):
    hold = holding_torques(robot_from_assembly(SPEC, arm_files))
    g = 9.80665
    # stretched out along x: shoulder holds upper (2 kg at 0.4 m) and fore (1.5 kg at 1.1 m)
    assert abs(hold["shoulder"]["zero_pose"]) == pytest.approx(g * (2 * 0.4 + 1.5 * 1.1), rel=1e-3)
    assert hold["shoulder"]["worst"] >= abs(hold["shoulder"]["zero_pose"]) - 1e-9
    assert hold["elbow"]["effort"] == 30


def test_unconnected_link_is_reported(arm_files):
    spec = dict(SPEC, joints=SPEC["joints"][:1])
    with pytest.raises(VehicleError, match="fore is not connected"):
        robot_from_assembly(spec, arm_files)


def test_web_robot_build(arm_files):
    from fastapi.testclient import TestClient

    from robomodel.web import app

    client = TestClient(app)
    links = []
    for lk in SPEC["links"]:
        data = arm_files[lk["file"]].read_bytes()
        r = client.post("/api/upload", files={"file": (f"{lk['name']}.stl", data, "model/stl")})
        links.append({**{k: v for k, v in lk.items() if k != "file"}, "upload": r.json()["id"]})
    r = client.post("/api/robot/build", json={**SPEC, "links": links})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["summary"]["dof"] == ["shoulder", "elbow"]
    assert all(c["status"] != "fail" for c in body["checks"])
    assert body["holding"]["shoulder"]["worst"] > 0
    z = client.get(f"/api/download/{links[0]['upload']}/{body['result']}")
    names = zipfile.ZipFile(io.BytesIO(z.content)).namelist()
    for part in (
        "urdf/arm2.urdf",
        "urdf/meshes/fore.stl",
        "matlab/arm2_example.m",
        "matlab/robomodel_robot_mass.m",
        "matlab/arm2_check_toolbox.m",
        "report.txt",
    ):
        assert f"arm2/{part}" in names
