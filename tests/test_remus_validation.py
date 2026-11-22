"""REMUS 100 against Prestero (2001): the agreement we have shown, kept from regressing.

The full table is in docs/dogrulama_remus100.md (validation/remus100.py). Terms where we
differ a lot from Prestero (X_udot, M_qdot, cross terms, body lift) are listed there and
not asserted here.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "validation"))

import remus100  # noqa: E402

from robomodel.model import build_model  # noqa: E402
from robomodel.vehicle import load_vehicle  # noqa: E402


@pytest.fixture(scope="module")
def model():
    return build_model(load_vehicle(remus100.VEHICLE))


@pytest.fixture(scope="module")
def rows(model):
    return {r.name: r for r in remus100.comparison(model)}


def check(row, ref, tolerance):
    assert row.sign_ok(ref), f"{row.name}: sign differs ({row.ours} vs {ref})"
    assert abs(row.error(ref)) < tolerance, f"{row.name}: {row.error(ref):+.0%} vs {ref}"


@pytest.mark.parametrize("name", ["V (m^3)", "B (N)"])
def test_hydrostatics_match_prestero_estimate(rows, name):
    check(rows[name], rows[name].theory, 0.05)  # plan target for group A: < 5 %


@pytest.mark.parametrize("name", ["B (N)", "x_cb burundan (m)"])
def test_hydrostatics_match_measurement(rows, name):
    check(rows[name], rows[name].adjusted, 0.05)


@pytest.mark.parametrize("name", ["Y_vdot", "Z_wdot", "K_pdot"])
def test_added_mass_matches_prestero_theory(rows, name):
    check(rows[name], rows[name].theory, 0.20)  # plan target for group B: < 20 %


@pytest.mark.parametrize("name", ["Y_v|v|", "Z_w|w|", "M_q|q|", "N_r|r|"])
def test_crossflow_damping_matches_prestero_theory(rows, name):
    check(rows[name], rows[name].theory, 0.25)


def test_axial_drag_matches_sea_measurement(rows):
    check(rows["eksenel C_d"], rows["eksenel C_d"].adjusted, 0.10)


@pytest.mark.parametrize(
    "name", ["Y_uvf", "Y_urf", "N_uvf", "N_urf", "M_uqf", "Y_uudr", "N_uudr", "Z_uuds", "M_uuds"]
)
def test_fin_lift_matches_prestero_formula(rows, name):
    check(rows[name], rows[name].theory, 0.15)


def test_turn_with_sea_tuned_damping_matches_sea_trial(model):
    """With Prestero's sea-tuned damping factors, our fins and dynamics give the measured
    turn rate. Without them (CAD only) the vehicle turns far too fast: see the report."""
    scale = [1, 10, 10, 1, 12.5, 10.0]
    rate = remus100.steady_turn(model, damping_scale=np.array(scale))
    assert rate == pytest.approx(remus100.FIELD_YAW_RATE, rel=0.3)
