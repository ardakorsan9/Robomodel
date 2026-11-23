"""Surface vessels: floating position, stability and motion against hand formulas."""

from pathlib import Path

import numpy as np
import pytest
import trimesh

from robomodel.checks import FAIL, OK, run_checks
from robomodel.dynamics import simulate
from robomodel.model import build_model
from robomodel.rigid_body import shell_properties
from robomodel.surface import SurfaceError, floating, natural_periods
from robomodel.vehicle import load_vehicle, parse_vehicle

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
RHO = 1025.0


def _barge(**extra):
    data = {
        "name": "barge",
        "category": "surface",
        "mass": RHO * 10 * 4 * 1.0,
        "water": "sea",
        "geometry": {
            "primitive": "box",
            "length": 10,
            "width": 4,
            "height": 2,
            "hull_mass": "solid",
        },
    }
    data.update(extra)
    return parse_vehicle(data)


def test_box_barge_hydrostatics_are_exact():
    fl = build_model(_barge()).floating
    assert fl.volume == pytest.approx(40.0, rel=1e-9)
    assert fl.depth == pytest.approx(0.0, abs=1e-9)  # waterline through the box centre
    assert fl.draft == pytest.approx(1.0, rel=1e-9)
    assert fl.waterplane_area == pytest.approx(40.0, rel=1e-6)
    assert fl.gm_t == pytest.approx(0.5 + 16 / 12 - 1, rel=1e-5)  # KB + BM - KG
    assert fl.gm_l == pytest.approx(0.5 + 100 / 12 - 1, rel=1e-5)


def test_box_barge_gz_follows_the_wall_sided_formula():
    fl = build_model(_barge()).floating
    gm, bm = 0.5 + 16 / 12 - 1, 16 / 12
    for deg in (5, 10, 20, 25):  # the deck edge goes under at 26.6 deg
        phi = np.radians(deg)
        wall = np.sin(phi) * (gm + 0.5 * bm * np.tan(phi) ** 2)
        assert np.interp(phi, fl.gz_angle, fl.gz) == pytest.approx(wall, rel=1e-6)
    # symmetric hull: GZ is odd in the heel angle
    assert np.allclose(fl.gz, -fl.gz[::-1], atol=1e-9)


def test_cg_forward_trims_the_bow_down():
    fl = build_model(_barge(measured={"r_g": [0.3, 0, 0]})).floating
    # small angle: the CB moves forward by x_g, tan(trim) = -x_g / GM_L
    assert np.tan(fl.trim) == pytest.approx(-0.3 / (0.5 + 100 / 12 - 1), rel=0.02)
    assert fl.heel == pytest.approx(0.0, abs=1e-9)


def test_wigley_hull_matches_its_closed_form():
    model = build_model(load_vehicle(EXAMPLES / "wigley.yaml"))
    fl = model.floating
    L, B, T = 4.0, 0.4, 0.25
    keel = model.hull.bounds[1, 2]
    kg = keel - model.mass.r_g.value[2]
    assert fl.volume / (L * B * T) == pytest.approx(4 / 9, rel=5e-4)
    assert fl.waterplane_area / (L * B) == pytest.approx(2 / 3, rel=5e-4)
    assert keel - fl.r_b[2] == pytest.approx(5 * T / 8, rel=5e-4)
    assert fl.gm_t == pytest.approx(5 * T / 8 + 3 * B**2 / (35 * T) - kg, rel=5e-3)


def test_floating_hemisphere_added_mass_is_half_the_displaced_mass():
    # Half-submerged sphere: the mirror image makes a whole sphere, so every translation has
    # added mass 1/2 rho V_sphere / 2 = 0.5 rho V_displaced (heave at high frequency, surge at
    # low frequency).
    r = 0.5
    displaced = RHO * 2 / 3 * np.pi * r**3
    vehicle = parse_vehicle(
        {
            "name": "buoy",
            "category": "surface",
            "mass": displaced,
            "geometry": {"primitive": "sphere", "radius": r},
            "measured": {"r_g": [0, 0, 0.2]},  # ballast below the centre: stable
        }
    )
    model = build_model(vehicle)
    assert model.floating.depth == pytest.approx(0.0, abs=1e-3)
    ma = np.diag(model.m_a.value)
    for i in (0, 1, 2):
        assert ma[i] == pytest.approx(0.5 * displaced, rel=0.06)


def test_too_heavy_hull_sinks_with_a_clear_error():
    with pytest.raises(SurfaceError, match="cannot carry its weight"):
        floating(trimesh.creation.box([1, 1, 1]), 2000.0, np.zeros(3), RHO)


def test_vessel_rests_floating_and_heaves_at_its_natural_period():
    model = build_model(_barge())
    dyn = model.dynamics()
    still = simulate(dyn, 5.0, dt_out=0.5)
    assert np.allclose(still.eta[-1], dyn.eta_rest, atol=1e-9)
    assert np.allclose(still.nu[-1], 0, atol=1e-9)

    from dataclasses import replace

    undamped = replace(dyn, d_lin=np.zeros(6), d_quad=np.zeros(6))
    period = natural_periods(dyn.mass_matrix, dyn.g_matrix)["heave"]
    x0 = dyn.rest_state.copy()
    x0[2] += 0.05  # pushed 5 cm down
    traj = simulate(undamped, 2.5 * period, state0=x0, dt_out=period / 200)
    z = traj.eta[:, 2] - dyn.eta_rest[2]
    peaks = traj.t[1:-1][(z[1:-1] > z[:-2]) & (z[1:-1] > z[2:])]
    assert peaks[1] - peaks[0] == pytest.approx(period, rel=0.01)


def test_high_cg_capsizes_and_is_flagged():
    model = build_model(_barge(measured={"r_g": [0, 0, -1.0]}))  # KG 2 m > KM 1.83 m
    assert model.floating.gm_t < 0
    checks = {c.name: c.status for c in run_checks(model.dynamics())}
    assert checks["stable in roll (GM_T > 0)"] == FAIL
    assert checks["stable in pitch (GM_L > 0)"] == OK


def test_surface_checks_pass_for_the_examples():
    for name in ("barge.yaml", "wigley.yaml"):
        checks = run_checks(build_model(load_vehicle(EXAMPLES / name)).dynamics())
        assert not [c for c in checks if c.status == FAIL], name


def test_shell_mass_of_a_sphere():
    mesh = trimesh.creation.icosphere(subdivisions=5, radius=2.0)
    centroid, inertia = shell_properties(mesh, 3.0)
    assert np.allclose(centroid, 0, atol=1e-9)
    assert np.allclose(inertia, 2 / 3 * 3.0 * 4.0 * np.eye(3), rtol=2e-3)


def test_matlab_export_carries_the_floating_terms(tmp_path):
    from robomodel.matlab_export import write_matlab

    write_matlab(build_model(load_vehicle(EXAMPLES / "wigley.yaml")), tmp_path, with_range=False)
    params = (tmp_path / "wigley_usv_params.m").read_text()
    assert "p.surface = true;" in params and "p.G = " in params and "p.roll_table" in params
    example = (tmp_path / "wigley_usv_example.m").read_text()
    assert "x0 = [0" in example  # starts at rest, z = origin depth
    under = tmp_path / "under"
    write_matlab(build_model(load_vehicle(EXAMPLES / "sphere.yaml")), under, with_range=False)
    assert "p.surface = false;" in (under / "sphere_params.m").read_text()
