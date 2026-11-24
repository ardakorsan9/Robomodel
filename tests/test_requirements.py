"""Requirements ("what does my robot need?") against hand formulas."""

import numpy as np
import pytest

from robomodel.ground import build_ground_model, tip_limits
from robomodel.model import build_model
from robomodel.model_range import model_range
from robomodel.requirements import format_requirements, requirements
from robomodel.vehicle import parse_vehicle

G = 9.80665
BOX = {"primitive": "box", "length": 0.5, "width": 0.3, "height": 0.2}


def _vehicle(category, mass=20.0, **extra):
    return parse_vehicle(
        {"name": "v", "category": category, "mass": mass, "geometry": BOX, **extra}
    )


@pytest.mark.parametrize("category", ["underwater", "surface"])
def test_thrust_to_hold_a_speed_is_the_damping_polynomial(category):
    model = build_model(_vehicle(category))
    req = requirements(model)["items"]
    for speed, key in ((0.5, "thrust_0_5_ms"), (1.0, "thrust_1_ms"), (2.0, "thrust_2_ms")):
        for label, variant in model_range(model).items():
            dyn = variant.dynamics()
            hand = dyn.d_lin[0] * speed + dyn.d_quad[0] * speed**2
            if label == "nominal":
                assert req[key]["nominal"] == pytest.approx(hand)
            if label == "low":
                assert req[key]["low"] == pytest.approx(hand)  # least damping: least thrust
            if label == "high":
                assert req[key]["high"] == pytest.approx(hand)
        assert req[key]["low"] < req[key]["nominal"] < req[key]["high"]
        assert req[key]["unit"] == "N"


def test_underwater_vertical_thrust_cancels_net_buoyancy():
    model = build_model(_vehicle("underwater"))
    dyn = model.dynamics()
    item = requirements(model)["items"]["vertical_thrust"]
    assert item["nominal"] == pytest.approx(abs(dyn.weight - dyn.buoyancy))
    assert item["direction"] == ("up" if dyn.weight > dyn.buoyancy else "down")
    # a boat floats: no vertical thrust entry
    assert "vertical_thrust" not in requirements(build_model(_vehicle("surface")))["items"]


def test_hover_thrust_is_the_weight():
    model = build_model(_vehicle("aerial", mass=1.5))
    out = requirements(model)
    items = out["items"]
    assert items["hover_thrust"]["nominal"] == pytest.approx(1.5 * G)
    assert items["max_thrust_total"]["nominal"] == pytest.approx(2 * 1.5 * G)
    assert items["max_thrust_per_rotor"]["nominal"] == pytest.approx(2 * 1.5 * G / 4)  # 4 assumed
    assert "rotors_assumed" in out["notes"]
    rotors = [
        {"position": [x, y, 0], "spin": s, "diameter": 0.2}
        for x, y, s in ((0.2, 0, "cw"), (-0.2, 0, "cw"), (0, 0.2, "ccw"), (0, -0.2, "ccw"))
        + ((0, 0.4, "cw"), (0, -0.4, "ccw"))
    ]
    six = requirements(build_model(_vehicle("aerial", mass=1.5, rotors=rotors)))
    assert six["items"]["max_thrust_per_rotor"]["nominal"] == pytest.approx(2 * 1.5 * G / 6)
    assert six["items"]["hover_per_rotor"]["rotors"] == 6 and not six["notes"]


def _rover(driven=True, radius=0.1, rolling=0.02):
    wheels = [
        {"position": [x, y, 0.05], "radius": radius, "driven": driven}
        for x in (0.2, -0.2)
        for y in (-0.2, 0.2)
    ]
    data = {
        "name": "bot",
        "category": "ground",
        "mass": 10.0,
        "geometry": {"primitive": "box", "length": 0.4, "width": 0.3, "height": 0.1},
        "measured": {"r_g": [0.0, 0.0, 0.0]},
        "wheels": wheels,
        "surface": {"mu": 0.8, "rolling": rolling},
    }
    return build_ground_model(parse_vehicle(data))


def test_ground_wheel_torque_matches_the_slope_and_acceleration_formulas():
    gm = _rover(rolling=0.02)
    items = requirements(gm)["items"]
    theta = np.arctan(0.2)
    # nominal: F = W (sin t + crr cos t) shared by 4 wheels, torque = F R
    force = 10 * G * (np.sin(theta) + 0.02 * np.cos(theta)) / 4
    assert items["slope_force"]["nominal"] == pytest.approx(force)
    assert items["slope_torque"]["nominal"] == pytest.approx(force * 0.1)
    acc = (10 * 1.0 + 0.02 * 10 * G) / 4
    assert items["accel_torque"]["nominal"] == pytest.approx(acc * 0.1)
    # range: rolling resistance 0.5x .. 2x spreads the need
    assert items["slope_torque"]["low"] < items["slope_torque"]["nominal"]
    assert items["slope_torque"]["nominal"] < items["slope_torque"]["high"]
    # tip-over limited acceleration = the existing static limit g tan(angle) at front / rear
    tips = tip_limits(gm)
    hand = G * np.tan(np.radians(min(tips["front"], tips["rear"])))
    assert items["tip_acceleration"]["nominal"] == pytest.approx(hand)
    assert items["slope_torque"]["wheels"] == 4


def test_ground_without_driven_wheels_shares_over_all_wheels():
    out = requirements(_rover(driven=False))
    assert "wheels_all" in out["notes"] and out["items"]["slope_force"]["wheels"] == 4
    assert "wheels_driven" in requirements(_rover())["notes"]


def test_report_text_lists_every_item():
    text = format_requirements(requirements(build_model(_vehicle("underwater"))))
    assert "hold 1 m/s" in text and "vertical thrust" in text
