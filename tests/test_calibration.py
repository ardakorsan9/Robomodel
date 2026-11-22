"""Calibration (O3): recover known terms from simulated test runs; files and CLI."""

from pathlib import Path

import numpy as np
import pytest
import yaml

from robomodel.calibration import (
    Log,
    calibrate,
    fit_percent,
    load_log,
    save_log,
    simulate_log,
)
from robomodel.cli import main
from robomodel.model import apply_calibration, build_model
from robomodel.vehicle import Calibration, load_vehicle, parse_vehicle

EXAMPLES = Path(__file__).parent.parent / "examples"
VEHICLE = {
    "name": "pod",
    "category": "underwater",
    "mass": 43.7,
    "water": "fresh",
    "geometry": {"primitive": "cylinder", "length": 1.6, "radius": 0.095},
    "parts": [{"mass": 6.0, "position": [0.0, 0.0, 0.04]}],
    "thrusters": [
        {
            "model": "T200",
            "position": [-0.85, 0, 0],
            "direction": [1, 0, 0],
            "max_forward": 50,
            "max_reverse": 40,
        },
    ],
}


@pytest.fixture(scope="module")
def cad():
    return build_model(parse_vehicle(VEHICLE))


def surge_log(model, name="surge", seed=0):
    t = np.arange(0.0, 20.0, 0.1)
    cmd = np.where(t < 12, 0.7, 0.0)  # accelerate, then coast
    clean = simulate_log(model, Log(t, {"thr_0": cmd}, {"u": np.zeros_like(t)}, name))
    rng = np.random.default_rng(seed)
    u = clean[:, 6] + rng.normal(0, 0.01, len(t))
    u[0] = 0.0
    return Log(t, {"thr_0": cmd}, {"u": u}, name)


def test_fit_percent_is_100_for_the_true_model(cad):
    t = np.arange(0.0, 10.0, 0.1)
    cmd = np.full_like(t, 0.5)
    clean = simulate_log(cad, Log(t, {"thr_0": cmd}, {"u": np.zeros_like(t)}))
    log = Log(t, {"thr_0": cmd}, {"u": clean[:, 6]})
    assert fit_percent(cad, log)["u"] == pytest.approx(100.0, abs=1e-3)


def test_calibration_recovers_surge_damping(cad):
    truth = apply_calibration(cad, Calibration(damping=np.array([4.0, 1, 1, 1, 1, 1])))
    result = calibrate(
        cad, [surge_log(truth)], groups=("damping",), validation=[surge_log(truth, "check", seed=1)]
    )
    assert result.calibration.damping[0] == pytest.approx(4.0, rel=0.1)
    assert result.calibration.damping_sigma[0] < 0.2  # well identified
    # Unexcited DOFs stay near the CAD value, with a large uncertainty.
    assert np.allclose(result.calibration.damping[1:], 1.0, atol=0.05)
    assert result.validation_after["check"]["u"] > 95
    assert result.validation_before["check"]["u"] < result.validation_after["check"]["u"]


def test_calibration_block_in_vehicle_file_is_applied(cad):
    data = dict(
        VEHICLE,
        calibration={
            "damping": [2, 1, 1, 1, 1, 1],
            "damping_sigma": [0.1] * 6,
            "note": "pool 2026-11",
        },
    )
    model = build_model(parse_vehicle(data))
    np.testing.assert_allclose(model.damping.quadratic.value[0], 2 * cad.damping.quadratic.value[0])
    assert model.damping.quadratic.source.value == "calibrated"


def test_bad_calibration_block_rejected():
    with pytest.raises(ValueError, match="calibration.damping"):
        parse_vehicle(dict(VEHICLE, calibration={"damping": [1, 2, 3]}))


def test_log_round_trip_and_template_columns(tmp_path, cad):
    log = surge_log(cad)
    path = tmp_path / "run.csv"
    save_log(log, path)
    back = load_log(path)
    np.testing.assert_allclose(back.states["u"], log.states["u"], atol=1e-6)
    # A test-plan template: empty state columns are ignored, half-filled ones rejected.
    lines = path.read_text().splitlines()
    lines[0] += ",psi"
    lines[1:] = [line + "," for line in lines[1:]]
    path.write_text("\n".join(lines) + "\n")
    assert "psi" not in load_log(path).states
    lines[1] = lines[1] + "0.1"
    path.write_text("\n".join(lines) + "\n")
    with pytest.raises(ValueError, match="empty cells"):
        load_log(path)


def test_unknown_input_rejected(cad):
    t = np.arange(0.0, 1.0, 0.1)
    log = Log(t, {"fin_missing": np.zeros_like(t)}, {"u": np.zeros_like(t)}, "bad")
    with pytest.raises(ValueError, match="not in the vehicle"):
        simulate_log(cad, log)


def test_cli_testplan_and_calibrate(tmp_path, cad):
    vehicle = tmp_path / "pod.yaml"
    vehicle.write_text(yaml.safe_dump(VEHICLE))
    assert main(["testplan", str(vehicle), "-o", str(tmp_path / "plan")]) == 0
    assert (tmp_path / "plan" / "TEST_PLANI.md").exists()
    assert (tmp_path / "plan" / "1_ileri_ve_sonumlenme.csv").exists()

    truth = apply_calibration(cad, Calibration(damping=np.array([3.0, 1, 1, 1, 1, 1])))
    save_log(surge_log(truth), tmp_path / "run.csv")
    out = tmp_path / "pod_cal.yaml"
    assert (
        main(
            [
                "calibrate",
                str(vehicle),
                str(tmp_path / "run.csv"),
                "--groups",
                "damping",
                "-o",
                str(out),
            ]
        )
        == 0
    )
    calibrated = load_vehicle(out)
    assert calibrated.calibration.damping[0] == pytest.approx(3.0, rel=0.1)


def test_remus_testplan_uses_fins():
    from robomodel.testplan import plan

    model = build_model(load_vehicle(EXAMPLES / "remus100.yaml"))
    names = [m.name for m in plan(model)]
    assert names == ["2_zikzak", "3_dalis", "4_serbest_salinim"]  # no thruster in the file


# --- Measurement wizard (O4) -------------------------------------------------


def test_wizard_ranks_buoyancy_first_for_a_near_neutral_vehicle(cad):
    from robomodel.wizard import analyse, report

    base, effects = analyse(cad)
    assert "son hız" in base and "net kaldırma" in base
    # 2 % volume uncertainty on a near-neutral vehicle decides whether it floats or sinks.
    assert effects[0].parameter == "kaldırma kuvveti (hacim)"
    assert "Önce bunları ölç" in report(cad)


def test_wizard_measured_values_drop_down_the_list(cad):
    from robomodel.wizard import analyse

    measured = build_model(parse_vehicle(dict(VEHICLE, measured={"r_g": [0, 0, 0.007]})))

    def rank(model):
        names = [e.parameter for e in analyse(model)[1]]
        return names.index("kütle merkezi yüksekliği")

    assert rank(measured) > rank(cad)


# --- Gaps closed after review ------------------------------------------------


def rov():
    """Four-thruster ROV: two forward (differential yaw) and two vertical."""
    thr = {"model": "x", "max_forward": 40, "max_reverse": 30}
    return parse_vehicle({
        "name": "rov", "category": "underwater", "mass": 21.2, "water": "fresh",
        "geometry": {"primitive": "ellipsoid", "a": 0.25, "b": 0.17, "c": 0.12},
        "parts": [{"mass": 3.0, "position": [0, 0, 0.05]}],
        "thrusters": [
            dict(thr, position=[-0.2, 0.12, 0], direction=[1, 0, 0]),
            dict(thr, position=[-0.2, -0.12, 0], direction=[1, 0, 0]),
            dict(thr, position=[0, 0.12, 0], direction=[0, 0, 1]),
            dict(thr, position=[0, -0.12, 0], direction=[0, 0, 1]),
        ],
    })  # fmt: skip


def test_wizard_covers_rov_turn_and_dive():
    from robomodel.wizard import analyse

    base, _ = analyse(build_model(rov()))
    assert base["dönüş hızı (%50 itici)"] > 0
    assert base["dikey hız (%50 itici)"] > 0  # pushing down (z positive)


def test_calibration_recovers_thrust_and_damping_together(cad):
    """Steady speed alone mixes thrust and damping; the coast-down separates them."""
    truth = apply_calibration(cad, Calibration(damping=np.array([3.0, 1, 1, 1, 1, 1]), thrust=0.8))
    result = calibrate(cad, [surge_log(truth)], groups=("damping", "thrust"))
    assert result.calibration.thrust == pytest.approx(0.8, rel=0.1)
    assert result.calibration.damping[0] == pytest.approx(3.0, rel=0.1)


def test_fin_deflection_is_limited_by_stall():
    from robomodel.dynamics import fin_control_force

    v = parse_vehicle(
        dict(
            VEHICLE,
            fins=[
                {
                    "name": "r",
                    "position": [-0.7, 0, -0.14],
                    "normal": [0, 1, 0],
                    "area": 0.0066,
                    "span": 0.08,
                    "control": True,
                    "max_angle": 15,
                }
            ],
        )
    )
    dyn = build_model(v).dynamics()
    nu = np.array([1.5, 0, 0, 0, 0, 0])
    at_limit = fin_control_force(dyn, nu, [np.radians(15)])
    np.testing.assert_allclose(fin_control_force(dyn, nu, [1.0]), at_limit)
    with pytest.raises(ValueError, match="max_angle"):
        parse_vehicle(
            dict(
                VEHICLE,
                fins=[
                    {
                        "position": [0, 0, 0],
                        "normal": [0, 1, 0],
                        "area": 0.01,
                        "span": 0.1,
                        "max_angle": 90,
                    }
                ],
            )
        )


def test_hull_along_the_wrong_axis_is_warned(tmp_path):
    import trimesh

    path = tmp_path / "tall.stl"
    trimesh.creation.cylinder(radius=0.1, height=1.5).export(path)  # long axis z
    model = build_model(
        parse_vehicle(
            {"name": "t", "category": "underwater", "mass": 47, "geometry": {"file": str(path)}}
        )
    )
    assert any("longest along body z" in w for w in model.warnings)


def test_check_shows_fins_measured_and_calibration(capsys):
    assert main(["check", str(EXAMPLES / "remus100.yaml")]) == 0
    out = capsys.readouterr().out
    assert "4 (4 control)" in out and "r_g, inertia_g" in out and "calibration   none" in out
