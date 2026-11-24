"""Ground robots: wheel loads, tip-over, driving and turning against hand calculations."""

from pathlib import Path

import numpy as np
import pytest

from robomodel.checks import FAIL, OK
from robomodel.ground import (
    GroundError,
    build_ground_model,
    performance,
    run_ground_checks,
    simulate,
    state_derivative,
    tip_limits,
)
from robomodel.vehicle import VehicleError, load_vehicle, parse_vehicle

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
G = 9.80665
MOTOR = {"driven": True, "stall_torque": 2.0, "no_load_rpm": 120}


def _robot(wheels, **extra):
    data = {
        "name": "bot",
        "category": "ground",
        "mass": 10.0,
        "geometry": {"primitive": "box", "length": 0.4, "width": 0.3, "height": 0.1},
        "measured": {"r_g": [0.0, 0.0, 0.0]},
        "wheels": wheels,
    }
    data.update(extra)
    return build_ground_model(parse_vehicle(data))


def _four(**motor):
    m = {**MOTOR, **motor}
    return [
        {"name": n, "position": [x, y, 0.05], "radius": 0.1, **m}
        for n, x, y in (("fl", 0.2, -0.2), ("fr", 0.2, 0.2), ("rl", -0.2, -0.2), ("rr", -0.2, 0.2))
    ]


def test_symmetric_robot_shares_its_weight_and_tips_at_the_hand_angle():
    gm = _robot(_four())
    assert np.allclose(gm.normal, 10 * G / 4)
    assert gm.cg_height == pytest.approx(0.15)  # wheel centre 0.05 below the CG + radius 0.1
    tips = tip_limits(gm)
    for side in ("front", "rear", "left", "right"):
        assert tips[side] == pytest.approx(np.degrees(np.arctan(0.2 / 0.15)))


def test_three_wheels_load_by_the_lever_rule():
    wheels = [
        {"name": "l", "position": [0.0, -0.2, 0.05], "radius": 0.1, **MOTOR},
        {"name": "r", "position": [0.0, 0.2, 0.05], "radius": 0.1, **MOTOR},
        {"name": "c", "position": [0.3, 0.0, 0.1], "radius": 0.05, "caster": True},
    ]
    gm = _robot(wheels, measured={"r_g": [0.1, 0.0, 0.0]})
    w = 10 * G
    assert gm.normal[2] == pytest.approx(w * 0.1 / 0.3)  # caster carries x_g / wheelbase
    assert gm.normal[0] == pytest.approx(gm.normal[1])


def test_cg_outside_the_wheels_fails_and_a_line_of_wheels_is_an_error():
    gm = _robot(_four(), measured={"r_g": [0.3, 0.0, 0.0]})
    status = {c.name: c.status for c in run_ground_checks(gm)}
    assert status["stands on its wheels (CG inside the wheel base)"] == FAIL
    with pytest.raises(GroundError, match="one line"):
        _robot(
            [
                {"name": "l", "position": [0.0, -0.2, 0.05], "radius": 0.1, **MOTOR},
                {"name": "r", "position": [0.0, 0.2, 0.05], "radius": 0.1, **MOTOR},
            ],
            measured={"r_g": [0.05, 0.0, 0.0]},
        )


def test_top_speed_matches_the_motor_line():
    gm = _robot(_four())
    perf = performance(gm)
    # per wheel: F_s (1 - v / v0) = C_rr N, F_s = 2 / 0.1 = 20 N, v0 = 0.1 * 120 * 2 pi / 60
    v0 = 0.1 * 120 * 2 * np.pi / 60
    expected = v0 * (1 - 0.015 * 10 * G / 4 / 20)
    assert perf["top_speed"] == pytest.approx(expected, rel=1e-9)
    traj = simulate(gm, 10.0, np.ones(4), dt_out=1.0)
    assert traj.state[-1, 3] == pytest.approx(expected, rel=1e-4)


def test_start_is_traction_limited():
    gm = _robot(_four(stall_torque=5.0))  # 50 N per wheel > grip 0.8 * 24.5 N
    accel = state_derivative(gm, np.zeros(6), np.ones(4))[3]
    assert accel == pytest.approx(0.8 * G, rel=1e-9)  # friction circle: mu g
    assert performance(gm)["max_acceleration"] == pytest.approx((0.8 - 0.015) * G)


def test_differential_drive_turns_like_its_kinematics():
    stiff = {**MOTOR, "stall_torque": 20.0}  # a strong motor holds its speed under load
    wheels = [
        {"name": "l", "position": [0.0, -0.2, 0.05], "radius": 0.1, **stiff},
        {"name": "r", "position": [0.0, 0.2, 0.05], "radius": 0.1, **stiff},
        {"name": "c", "position": [0.25, 0.0, 0.1], "radius": 0.05, "caster": True},
    ]
    # no rolling resistance: it slows the wheels below their motor speed, which is real but
    # not part of the kinematics checked here
    gm = _robot(wheels, measured={"r_g": [0.05, 0.0, 0.0]}, surface={"mu": 0.8, "rolling": 0.0})
    traj = simulate(gm, 10.0, np.array([0.2, 0.1]), dt_out=1.0)
    u, v, r = traj.state[-1, 3:]
    assert r > 0  # left wheel faster: turns right (FRD: positive yaw)
    v0 = 0.1 * 120 * 2 * np.pi / 60
    assert r == pytest.approx(0.1 * v0 / 0.4, rel=0.01)  # no-slip kinematics
    assert u == pytest.approx(0.15 * v0, rel=0.01)
    # the axle hardly slides sideways (a small slip angle makes the side force)
    assert v == pytest.approx(0.05 * r, rel=0.05)
    # The real motor (10x softer) gives way under the turning load: a wider turn at speed.
    soft = _robot(
        [dict(w, **MOTOR) if "caster" not in w else w for w in wheels],
        measured={"r_g": [0.05, 0.0, 0.0]},
    )
    fast = simulate(soft, 10.0, np.array([1.0, 0.5]), dt_out=1.0).state[-1, 5]
    assert fast < 0.9 * 0.5 * v0 / 0.4


def test_steered_robot_follows_the_bicycle_model():
    wheels = [
        {"name": "fl", "position": [0.3, -0.15, 0.05], "radius": 0.1, "steered": True},
        {"name": "fr", "position": [0.3, 0.15, 0.05], "radius": 0.1, "steered": True},
        {"name": "rl", "position": [-0.3, -0.15, 0.05], "radius": 0.1, **MOTOR},
        {"name": "rr", "position": [-0.3, 0.15, 0.05], "radius": 0.1, **MOTOR},
    ]
    gm = _robot(wheels)
    delta = 0.15
    traj = simulate(gm, 10.0, np.array([0.3, 0.3, delta, delta]), dt_out=1.0)
    u, r = traj.state[-1, 3], traj.state[-1, 5]
    assert r == pytest.approx(u * np.tan(delta) / 0.6, rel=0.05)  # r = u tan(delta) / L
    assert performance(gm)["min_turn_radius"] == pytest.approx(0.6 / np.tan(np.radians(35)))


def test_tracks_spread_their_load_and_coasting_stops():
    tracks = [
        {"name": "l", "position": [0, -0.2, 0.05], "radius": 0.1, "track_length": 0.4, **MOTOR},
        {"name": "r", "position": [0, 0.2, 0.05], "radius": 0.1, "track_length": 0.4, **MOTOR},
    ]
    gm = _robot(tracks)
    assert len(gm.normal) == 10 and gm.normal.sum() == pytest.approx(10 * G)
    assert np.allclose(gm.stall_force[:5].sum(), 20.0)  # one track's motor
    traj = simulate(gm, 10.0, np.zeros(2), state0=[0, 0, 0, 0.5, 0, 0], dt_out=1.0)
    assert abs(traj.state[-1, 3]) < 1e-3  # rolling resistance stops it
    status = {c.name: c.status for c in run_ground_checks(gm)}
    assert status["can turn (steering or driven wheels on both sides)"] == OK


def test_driven_wheel_needs_both_motor_numbers_or_neither():
    with pytest.raises(VehicleError, match="both stall_torque"):
        _robot(
            [{"name": "w", "position": [0, 0, 0], "radius": 0.1, "driven": True, "stall_torque": 1}]
        )


def test_torque_input_wheels_accelerate_by_the_hand_formula():
    """No motor data: the input is the wheel torque in N m, no limit, no motor line."""
    wheels = [
        {"name": n, "position": [x, y, 0.05], "radius": 0.1, "driven": True}
        for n, x, y in (("fl", 0.2, -0.2), ("fr", 0.2, 0.2), ("rl", -0.2, -0.2), ("rr", -0.2, 0.2))
    ]
    gm = _robot(wheels, surface={"mu": 1.5, "rolling": 0.0})
    assert gm.direct_torque and np.all(np.isinf(gm.free_speed))
    torque = 3.0  # N m per wheel: well above what any motor line would give (command clip 1)
    traj = simulate(gm, 1.0, np.full(4, torque), dt_out=0.25)
    accel = 4 * torque / 0.1 / 10.0  # sum of tau / R over m, 12 m/s^2 < mu g = 14.7
    assert np.allclose(traj.state[:, 3], accel * traj.t, rtol=1e-5, atol=1e-9)
    assert performance(gm) == {}  # no motor data: no top speed or acceleration
    assert any("no motor data" in w for w in gm.warnings)
    # the low / high models do not scale a torque input
    from robomodel.ground import model_range

    assert np.allclose(model_range(gm)["high"].stall_force, gm.stall_force)


def test_torque_input_matlab_has_the_direct_case(tmp_path):
    from robomodel.ground_matlab import write_ground_matlab

    gm = _robot(
        [
            {"name": "l", "position": [0, -0.2, 0.05], "radius": 0.1, "driven": True},
            {"name": "r", "position": [0, 0.2, 0.05], "radius": 0.1, "driven": True},
            {"name": "c", "position": [0.3, 0, 0.1], "radius": 0.05, "caster": True},
        ]
    )
    write_ground_matlab(gm, tmp_path)
    params = (tmp_path / "bot_params.m").read_text()
    assert "p.direct_in = logical([1; ...\n     1]);" in params
    assert "p.motor = [0; ...\n     0; ...\n     1];" in params
    assert (
        "cmd(p.direct_in) = ud(p.direct_in);"
        in (tmp_path / "robomodel_ground_dynamics.m").read_text()
    )


def test_example_rover_and_matlab_files(tmp_path):
    from robomodel.ground_matlab import write_ground_matlab

    gm = build_ground_model(load_vehicle(EXAMPLES / "rover.yaml"))
    assert not [c for c in run_ground_checks(gm) if c.status == FAIL]
    names = {p.name for p in write_ground_matlab(gm, tmp_path)}
    assert {"rover4_params.m", "rover4_params_low.m", "robomodel_ground_dynamics.m"} <= names
    text = (tmp_path / "rover4_params.m").read_text()
    assert "p.mu = 0.8" in text and "p.n_drive = 4;" in text
