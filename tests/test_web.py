"""Web app: upload, preview, build and download, through the HTTP API."""

import io
import zipfile

import pytest
import trimesh
from fastapi.testclient import TestClient

from robomodel.web import app

client = TestClient(app)


def stl_bytes() -> bytes:
    hull = trimesh.creation.cylinder(radius=95, height=1600, sections=64)  # mm, along file z
    buf = io.BytesIO()
    hull.export(buf, file_type="stl")
    return buf.getvalue()


@pytest.fixture(scope="module")
def upload_id():
    r = client.post("/api/upload", files={"file": ("auv.stl", stl_bytes(), "model/stl")})
    assert r.status_code == 200, r.text
    return r.json()["id"]


def test_page_and_version():
    assert "robomodel" in client.get("/").text
    assert client.get("/api/version").json()["robomodel"]


def test_rejects_unsupported_files():
    r = client.post("/api/upload", files={"file": ("x.sldprt", b"123", "application/octet-stream")})
    assert r.status_code == 400


def test_preview(upload_id):
    r = client.get(f"/api/preview/{upload_id}")
    assert r.status_code == 200 and len(r.content) > 1000
    info = client.get(f"/api/preview/{upload_id}/info").json()
    assert info["watertight"] is True


def test_build_and_download(upload_id):
    spec = {
        "name": "web_auv",
        "category": "underwater",
        "mass": 45.0,
        "water": "fresh",
        "geometry": {"unit": "mm", "axes": ["z", "x", "y"], "assembly": False},
        "thrusters": [{"model": "T200", "position": [-0.8, 0, 0], "direction": [1, 0, 0]}],
        "fins": [
            {
                "name": "rudder",
                "position": [-0.7, 0, -0.13],
                "normal": [0, 1, 0],
                "area": 0.0066,
                "span": 0.08,
                "control": True,
            }
        ],
    }
    r = client.post(f"/api/build/{upload_id}", json=spec)
    assert r.status_code == 200, r.text
    body = r.json()
    assert abs(body["summary"]["mass"] - 45.0) < 1e-9
    assert any(c["status"] == "ok" for c in body["checks"])
    speed = body["behaviour"]["top_speed"]  # plain summary with the low / high range
    assert speed["high"] <= speed["nominal"] <= speed["low"]
    z = client.get(f"/api/download/{upload_id}/{body['result']}")
    names = zipfile.ZipFile(io.BytesIO(z.content)).namelist()
    for part in (
        "report.txt",
        "vehicle.yaml",
        "matlab/web_auv_params.m",
        "matlab/web_auv_params_high.m",
        "matlab/web_auv_build_simulink.m",
        "gazebo/web_auv/model.sdf",
    ):
        assert f"web_auv/{part}" in names


def test_build_reports_input_errors(upload_id):
    r = client.post(
        f"/api/build/{upload_id}",
        json={"name": "x", "category": "underwater", "geometry": {"unit": "mm"}},
    )
    assert r.status_code == 422 and "mass" in r.json()["detail"]


def test_bad_ids_are_refused():
    assert client.get("/api/preview/../../etc").status_code in (400, 404)
    assert client.get("/api/preview/zzzz").status_code == 400


def test_example_delete_and_health():
    r = client.post("/api/example")
    assert r.status_code == 200
    ex = r.json()
    assert ex["spec"]["mass"] == 30.49
    info = client.get(f"/api/preview/{ex['id']}/info").json()
    assert max(info["extents"]) == pytest.approx(1333, rel=0.01)  # mm
    assert client.get("/api/health").json()["ok"] is True
    assert client.delete(f"/api/upload/{ex['id']}").status_code == 200
    assert client.get(f"/api/preview/{ex['id']}/info").status_code == 404


def test_security_headers():
    r = client.get("/api/version")
    assert r.headers["X-Content-Type-Options"] == "nosniff"
    assert r.headers["X-Frame-Options"] == "DENY"


def test_build_timeout_and_busy(upload_id, monkeypatch):
    import robomodel.web as web

    monkeypatch.setattr(web, "BUILD_TIMEOUT_S", 0.01)
    spec = {
        "name": "t",
        "category": "underwater",
        "mass": 45.0,
        "water": "fresh",
        "geometry": {"unit": "mm", "axes": ["z", "x", "y"]},
    }
    assert client.post(f"/api/build/{upload_id}", json=spec).status_code == 504
    monkeypatch.setattr(web, "JOBS", __import__("threading").BoundedSemaphore(1))
    web.JOBS.acquire()
    assert client.post(f"/api/build/{upload_id}", json=spec).status_code == 503
    web.JOBS.release()


def test_surface_vessel_build():
    box = trimesh.creation.box([2000, 600, 400])  # mm: a 2 m pontoon boat
    buf = io.BytesIO()
    box.export(buf, file_type="stl")
    r = client.post("/api/upload", files={"file": ("boat.stl", buf.getvalue(), "model/stl")})
    upload = r.json()["id"]
    spec = {
        "name": "web_boat",
        "category": "surface",
        "mass": 180.0,  # fresh water: draft = 180 / (1000 * 2 * 0.6) = 0.15 m
        "water": "fresh",
        "geometry": {"unit": "mm"},
        "thrusters": [
            {"model": "T200", "position": [-0.95, 0.2, 0.1], "direction": [1, 0, 0]},
            {"model": "T200", "position": [-0.95, -0.2, 0.1], "direction": [1, 0, 0]},
        ],
    }
    r = client.post(f"/api/build/{upload}", json=spec)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["summary"]["category"] == "surface"
    assert float(body["summary"]["floating"]["draft"].split()[0]) == pytest.approx(0.15, rel=1e-3)
    assert {"top_speed", "turn_rate_thrusters", "heave_period", "roll_period"} <= set(
        body["behaviour"]
    )
    assert "net_buoyancy" not in body["behaviour"]
    names = zipfile.ZipFile(
        io.BytesIO(client.get(f"/api/download/{upload}/{body['result']}").content)
    ).namelist()
    assert "web_boat/gazebo/README.txt" in names and "web_boat/matlab/web_boat_params.m" in names


def test_drone_build():
    frame = trimesh.creation.box([300, 300, 60])  # mm
    buf = io.BytesIO()
    frame.export(buf, file_type="stl")
    r = client.post("/api/upload", files={"file": ("quad.stl", buf.getvalue(), "model/stl")})
    upload = r.json()["id"]
    rotors = [
        {"position": [x, y, -0.03], "spin": s, "max_thrust": 8.0, "diameter": 0.254}
        for x, y, s in (
            (0.15, 0.15, "ccw"),
            (-0.15, -0.15, "ccw"),
            (0.15, -0.15, "cw"),
            (-0.15, 0.15, "cw"),
        )
    ]
    spec = {
        "name": "web_quad",
        "category": "aerial",
        "mass": 1.5,
        "air": 1.225,
        "geometry": {"unit": "mm"},
        "rotors": rotors,
        "parts": [{"name": "battery", "mass": 0.35, "position": [0, 0, 0.02]}],
    }
    r = client.post(f"/api/build/{upload}", json=spec)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["model"]["hover"]["possible"] is True
    assert body["behaviour"]["hover_throttle"]["nominal"] == pytest.approx(
        (1.5 * 9.80665 / 32) ** 0.5, rel=1e-3
    )
    assert all(c["status"] != "fail" for c in body["checks"])
    names = zipfile.ZipFile(
        io.BytesIO(client.get(f"/api/download/{upload}/{body['result']}").content)
    ).namelist()
    assert "web_quad/gazebo/README.txt" in names and "web_quad/matlab/web_quad_params.m" in names
