"""Model checks, low/nominal/high range and provenance stamps."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from robomodel.checks import FAIL, OK, WARN, run_checks
from robomodel.model import build_model, to_dict
from robomodel.model_range import model_range
from robomodel.vehicle import load_vehicle

EXAMPLES = Path(__file__).parent.parent / "examples"


def statuses(checks):
    return {c.name: c.status for c in checks}


def test_good_model_passes_all_hard_checks():
    model = build_model(load_vehicle(EXAMPLES / "remus100.yaml"))
    checks = run_checks(model.dynamics())
    assert not [c for c in checks if c.status == FAIL]
    assert statuses(checks)["self-righting in roll and pitch (CG below CB)"] == OK


def test_checks_catch_a_broken_model():
    dyn = build_model(load_vehicle(EXAMPLES / "test_auv.yaml")).dynamics()
    bad = replace(dyn, d_lin=-dyn.d_lin, d_quad=-dyn.d_quad)  # damping that adds energy
    s = statuses(run_checks(bad))
    assert s["damping only removes energy (nu' D nu >= 0)"] == FAIL
    assert s["free motion loses energy (simulation)"] == FAIL
    m_rb = dyn.m_rb.copy()
    m_rb[0, 0] = -1.0  # negative mass
    assert (
        statuses(run_checks(replace(dyn, m_rb=m_rb)))[
            "rigid-body mass matrix M_RB symmetric, positive definite"
        ]
        == FAIL
    )


def test_unstable_bare_hull_is_a_warning_not_a_failure():
    model = build_model(load_vehicle(EXAMPLES / "test_auv.yaml"))  # no fins
    s = statuses(run_checks(model.dynamics()))
    assert s["directionally stable at 1 m/s (no control)"] == WARN


def test_range_brackets_the_nominal_model():
    model = build_model(load_vehicle(EXAMPLES / "test_auv.yaml"))
    r = model_range(model)
    assert list(r) == ["low", "nominal", "high"]
    low, nom, high = (np.asarray(r[k].damping.quadratic.value) for k in r)
    assert np.all(low <= nom) and np.all(nom <= high)
    assert np.all(np.diag(r["low"].m_a.value) <= np.diag(nom_ma := model.m_a.value) + 1e-12)
    assert np.all(np.diag(r["high"].m_a.value) >= np.diag(nom_ma) - 1e-12)
    assert r["low"].damping.quadratic.source.value == "assumed"
    for variant in r.values():  # every bound is still a physically valid model
        assert not [c for c in run_checks(variant.dynamics()) if c.status == FAIL]


def test_calibrated_model_has_no_range():
    from robomodel.model import apply_calibration
    from robomodel.vehicle import Calibration

    model = build_model(load_vehicle(EXAMPLES / "test_auv.yaml"))
    cal = apply_calibration(model, Calibration(damping=np.full(6, 2.0)))
    assert list(model_range(cal)) == ["nominal"]


def test_outputs_carry_provenance_and_checks():
    model = build_model(load_vehicle(EXAMPLES / "sphere.yaml"))
    data = to_dict(model)
    assert data["provenance"]["vehicle_file"] == "sphere.yaml"
    assert len(data["provenance"]["vehicle_file_sha256"]) == 64
    assert {c["status"] for c in data["checks"]} <= {OK, WARN, FAIL}


# --- Gazebo ------------------------------------------------------------------


def test_gazebo_coefficients_flip_signs_into_flu():
    from robomodel.gazebo_export import hydro_coefficients

    model = build_model(load_vehicle(EXAMPLES / "remus100.yaml"))
    c = hydro_coefficients(model)
    dyn = model.dynamics()
    assert c["xDotU"] == pytest.approx(-dyn.m_a[0, 0])
    assert c["yVabsV"] == pytest.approx(-dyn.d_quad[1])
    # y and z flip, so a y-yaw coupling keeps its sign (two flips), sway-heave too;
    # an x-yaw coupling would flip once.
    assert c["yDotR"] == pytest.approx(-dyn.m_a[1, 5])
    assert c["nVabsU"] == pytest.approx(-dyn.d_fin[5, 1])
    assert c["zQabsU"] == pytest.approx(-dyn.d_fin[2, 4])


def test_gazebo_files_are_valid_xml(tmp_path):
    import xml.etree.ElementTree as ET

    from robomodel.cli import main

    assert main(["gazebo", str(EXAMPLES / "test_auv.yaml"), "-o", str(tmp_path)]) == 0
    sdf = ET.parse(tmp_path / "test_auv" / "model.sdf").getroot()
    plugins = [p.get("name") for p in sdf.iter("plugin")]
    assert "gz::sim::systems::Hydrodynamics" in plugins
    assert "gz::sim::systems::Thruster" in plugins
    mass = float(sdf.find(".//inertial/mass").text)
    assert abs(mass - 45.0) < 1e-9
    radius = float(sdf.find(".//collision/geometry/sphere/radius").text)
    model = build_model(load_vehicle(EXAMPLES / "test_auv.yaml"))
    assert 4 / 3 * np.pi * radius**3 == pytest.approx(model.statics.volume.value, rel=1e-6)
    ET.parse(tmp_path / "world.sdf")
