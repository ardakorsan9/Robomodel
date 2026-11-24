"""Multirotor drones: rotor layout, hover, drag and checks against hand calculations."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from robomodel.aerial import ROTOR_DRAG_SHARE
from robomodel.checks import FAIL, OK, run_checks
from robomodel.dynamics import simulate
from robomodel.model import build_model
from robomodel.vehicle import VehicleError, load_vehicle, parse_vehicle

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
ARM = 0.2


def _quad(**extra):
    rotors = [
        {"position": [ARM, 0, -0.02], "spin": "ccw", "max_thrust": 6.0, "diameter": 0.2},
        {"position": [-ARM, 0, -0.02], "spin": "ccw", "max_thrust": 6.0, "diameter": 0.2},
        {"position": [0, ARM, -0.02], "spin": "cw", "max_thrust": 6.0, "diameter": 0.2},
        {"position": [0, -ARM, -0.02], "spin": "cw", "max_thrust": 6.0, "diameter": 0.2},
    ]
    data = {
        "name": "plus_quad",
        "category": "aerial",
        "mass": 1.0,
        "geometry": {"primitive": "box", "length": 0.1, "width": 0.1, "height": 0.04},
        "rotors": rotors,
    }
    data.update(extra)
    return parse_vehicle(data)


def test_rotor_columns_have_the_right_signs():
    model = build_model(_quad())
    b = model.b_thr
    kappa = model.vehicle.thrusters[0].torque
    assert np.allclose(b[2], -1)  # every rotor pushes up (-z)
    # front rotor: nose up (+M); right rotor: roll left (-K)
    assert b[4, 0] == pytest.approx(ARM) and b[3, 2] == pytest.approx(-ARM)
    # a cw rotor (seen from above) turns the body ccw: -N; ccw gives +N
    assert kappa < 0 and b[5, 0] == pytest.approx(-kappa)
    assert b[5, 2] == pytest.approx(-abs(kappa))


def test_hover_shares_the_weight_and_holds_still():
    model = build_model(_quad())
    w = 1.0 * 9.80665
    assert np.allclose(model.hover.thrust, w / 4, rtol=1e-6)
    assert model.hover.thrust_to_weight == pytest.approx(24 / w)
    from robomodel.matlab_export import example_tau

    tau, commands = example_tau(model)
    assert np.allclose(commands, np.sqrt(w / 4 / 6.0))
    traj = simulate(model.dynamics(), 5.0, tau=tau)
    assert np.max(np.abs(traj.eta[-1])) < 1e-6 and np.max(np.abs(traj.nu[-1])) < 1e-6


def test_cg_off_centre_shifts_the_load_and_too_far_fails():
    model = build_model(_quad(measured={"r_g": [0.05, 0, 0]}))
    f = model.hover.thrust
    assert f[0] > f[1]  # front rotor carries more
    assert f[0] * ARM - f[1] * ARM == pytest.approx(1.0 * 9.80665 * 0.05, rel=1e-6)
    far = build_model(_quad(measured={"r_g": [0.3, 0, 0]}))  # outside the rotors
    assert not far.hover.possible
    checks = {c.name: c.status for c in run_checks(far.dynamics())}
    assert checks["can hover level"] == FAIL


def test_free_fall_reaches_the_drag_terminal_speed():
    dyn = build_model(_quad()).dynamics()
    traj = simulate(dyn, 60.0, dt_out=1.0)
    w = traj.nu[-1, 2]
    a, b = dyn.d_quad[2], dyn.d_lin[2]
    expected = (-b + np.sqrt(b * b + 4 * a * dyn.weight)) / (2 * a)
    assert w == pytest.approx(expected, rel=1e-4)
    early = simulate(dyn, 0.1, dt_out=0.1)
    assert early.nu[-1, 2] == pytest.approx(9.80665 * 0.1, rel=0.01)  # free fall at g


def test_rotor_drag_is_half_the_ram_drag():
    model = build_model(_quad())
    area = np.pi * 0.2**2 / 4
    m_dot = np.sqrt(1.225 * area * (9.80665 / 4) / 2)
    frame = build_model(_quad(rotors=[]))
    extra = np.asarray(model.damping.linear.value) - np.asarray(frame.damping.linear.value)
    assert extra[0] == pytest.approx(4 * ROTOR_DRAG_SHARE * m_dot, rel=1e-6)
    assert extra[2] == pytest.approx(0.0, abs=1e-12)  # not along the rotor axis


def test_unbalanced_spins_are_flagged():
    rotors = [
        {"position": [ARM, 0, 0], "spin": "cw", "max_thrust": 6.0, "diameter": 0.2},
        {"position": [-ARM, 0, 0], "spin": "cw", "max_thrust": 6.0, "diameter": 0.2},
        {"position": [0, ARM, 0], "spin": "cw", "max_thrust": 6.0, "diameter": 0.2},
        {"position": [0, -ARM, 0], "spin": "ccw", "max_thrust": 6.0, "diameter": 0.2},
    ]
    model = build_model(_quad(rotors=rotors))
    assert any("yaw is unbalanced" in w for w in model.warnings)


def test_motor_mass_becomes_a_part():
    v = _quad(
        rotors=[
            {
                "position": [ARM, 0, 0],
                "spin": "cw",
                "max_thrust": 6,
                "diameter": 0.2,
                "motor_mass": 0.05,
            }
        ]
    )
    assert v.parts[0].mass == 0.05 and np.allclose(v.parts[0].position, [ARM, 0, 0])


def test_bad_rotor_input_is_reported():
    with pytest.raises(VehicleError) as e:
        _quad(rotors=[{"position": [0, 0, 0], "max_thrust": 5}])
    text = str(e.value)
    assert "spin must be cw or ccw" in text and "diameter" in text


def test_example_quad_passes_every_check():
    model = build_model(load_vehicle(EXAMPLES / "quad.yaml"))
    checks = run_checks(model.dynamics())
    assert not [c for c in checks if c.status == FAIL]
    status = {c.name: c.status for c in checks}
    assert status["rotors control thrust, roll, pitch and yaw"] == OK
    # yaw inertia: 4 motors of 60 g at (+-0.159, +-0.159) add m (x^2 + y^2) each, and the
    # uniform hull (0.32 x 0.32 m) gets 0.24 kg lighter
    without = build_model(replace(model.vehicle, parts=model.vehicle.parts[:1]))
    gained = model.mass.inertia_g.value[2, 2] - without.mass.inertia_g.value[2, 2]
    expected = 4 * 0.06 * 2 * 0.159**2 - 0.24 * (0.32**2 + 0.32**2) / 12
    assert gained == pytest.approx(expected, rel=1e-6)


def test_matlab_and_web_handle_a_drone(tmp_path):
    from robomodel.matlab_export import write_matlab

    model = build_model(load_vehicle(EXAMPLES / "quad.yaml"))
    write_matlab(model, tmp_path, with_range=False)
    example = (tmp_path / "quad450_example.m").read_text()
    assert "% hover" in example
    assert "p.B = 0;" in (tmp_path / "quad450_params.m").read_text()


BOX = {"primitive": "box", "length": 0.1, "width": 0.1, "height": 0.04}


def test_drone_without_rotors_builds_and_only_warns():
    data = {"name": "norotor", "category": "aerial", "mass": 1.0, "geometry": BOX}
    model = build_model(parse_vehicle(data))
    checks = run_checks(model.dynamics())
    assert not [c for c in checks if c.status == FAIL]
    assert any(c.status == "warn" and "rotors" in c.name for c in checks)
    assert any("no rotors given" in w for w in model.warnings)


def test_rotor_without_max_thrust_is_rated_at_the_recommended_value():
    data = {
        "name": "q",
        "category": "aerial",
        "mass": 1.0,
        "geometry": BOX,
        "rotors": [
            {"position": [ARM, 0, 0], "spin": "ccw", "diameter": 0.2},
            {"position": [-ARM, 0, 0], "spin": "ccw", "max_thrust": None, "diameter": 0.2},
            {"position": [0, ARM, 0], "spin": "cw", "diameter": 0.2},
            {"position": [0, -ARM, 0], "spin": "cw", "diameter": 0.2},
        ],
    }
    model = build_model(parse_vehicle(data))
    assert all(t.assumed_limit for t in model.vehicle.thrusters)
    assert model.vehicle.thrusters[0].max_forward == pytest.approx(2 * 9.80665 / 4)
    assert model.hover.possible and model.hover.thrust_to_weight == pytest.approx(2.0)
    assert any("maximum rotor thrust not given" in w for w in model.warnings)
