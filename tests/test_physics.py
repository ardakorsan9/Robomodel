"""Phase 1 acceptance tests: every computed quantity against an exact answer.

Doğrulama merdiveni basamak 1-3 (docs/plan.md): küre, elipsoit, silindir, Myring.
"""

from pathlib import Path

import numpy as np
import pytest
from scipy.integrate import quad

from robomodel.added_mass import ellipsoid_added_mass, lamb_integrals
from robomodel.geometry import GeometryError, build_mesh, myring_radius
from robomodel.hydrostatics import GRAVITY, restoring_forces
from robomodel.model import build_model, to_dict
from robomodel.parameter import Parameter, Source
from robomodel.rigid_body import mass_properties, point_inertia, rigid_body_mass_matrix, skew
from robomodel.vehicle import Geometry, load_vehicle, parse_vehicle

EXAMPLES = Path(__file__).parent.parent / "examples"
MESH_RTOL = 1e-3  # acceptance: < 0.1 % against the exact answer
REMUS = dict(a=0.191, b=0.654, c=0.541, d=0.191, n=2.0, theta=0.436)


def vehicle(geometry, mass=10.0, parts=None, water="sea"):
    return parse_vehicle(
        {
            "name": "t",
            "category": "underwater",
            "mass": mass,
            "water": water,
            "geometry": geometry,
            **({"parts": parts} if parts else {}),
        }
    )


# --- Parameter ---------------------------------------------------------------


def test_parameter_rejects_negative_sigma_and_shape_mismatch():
    with pytest.raises(ValueError):
        Parameter(1.0, -0.1, Source.CAD)
    with pytest.raises(ValueError):
        Parameter(np.zeros(3), 0.1, Source.CAD)


def test_parameter_relative_and_format():
    p = Parameter.relative(10.0, 0.05, Source.MEASURED, "kg")
    assert p.sigma == pytest.approx(0.5)
    assert p.format() == "10 ± 0.5 kg [measured]"


# --- Geometry: volume and centroid ------------------------------------------


def test_sphere_volume():
    mesh = build_mesh(Geometry(primitive="sphere", dims={"radius": 0.1}))
    assert mesh.volume == pytest.approx(4 / 3 * np.pi * 0.1**3, rel=MESH_RTOL)


def test_ellipsoid_volume_and_extent():
    mesh = build_mesh(Geometry(primitive="ellipsoid", dims={"a": 0.5, "b": 0.2, "c": 0.1}))
    assert mesh.volume == pytest.approx(4 / 3 * np.pi * 0.5 * 0.2 * 0.1, rel=MESH_RTOL)
    np.testing.assert_allclose(mesh.extents, [1.0, 0.4, 0.2], rtol=MESH_RTOL)


def test_cylinder_axis_is_body_x():
    mesh = build_mesh(Geometry(primitive="cylinder", dims={"length": 1.6, "radius": 0.095}))
    assert mesh.volume == pytest.approx(np.pi * 0.095**2 * 1.6, rel=MESH_RTOL)
    np.testing.assert_allclose(mesh.extents, [1.6, 0.19, 0.19], rtol=MESH_RTOL)


def test_myring_profile_is_continuous_and_closed():
    a, b, c, d = REMUS["a"], REMUS["b"], REMUS["c"], REMUS["d"]
    r = myring_radius(np.array([0.0, a, a + b, a + b + c]), **REMUS)
    np.testing.assert_allclose(r, [0, d / 2, d / 2, 0], atol=1e-12)


def test_myring_volume_and_centroid_match_integral():
    mesh = build_mesh(Geometry(primitive="myring", dims=REMUS))
    length = REMUS["a"] + REMUS["b"] + REMUS["c"]
    breaks = [REMUS["a"], REMUS["a"] + REMUS["b"]]

    def area(h):
        return np.pi * myring_radius(np.array([h]), **REMUS)[0] ** 2

    volume = quad(area, 0, length, points=breaks, limit=200)[0]
    x_centroid = (
        quad(lambda h: area(h) * (length / 2 - h), 0, length, points=breaks, limit=200)[0] / volume
    )
    assert mesh.volume == pytest.approx(volume, rel=MESH_RTOL)
    assert mesh.center_mass[0] == pytest.approx(x_centroid, abs=1e-4)
    assert mesh.center_mass[0] > 0  # blunt nose forward, so the centroid is ahead of mid-length


def test_open_mesh_file_is_rejected(tmp_path):
    stl = tmp_path / "open.stl"
    stl.write_text(
        "solid t\nfacet normal 0 0 1\nouter loop\nvertex 0 0 0\nvertex 1 0 0\nvertex 0 1 0\n"
        "endloop\nendfacet\nendsolid t\n"
    )
    with pytest.raises(GeometryError, match="no volume"):  # a lone triangle
        build_mesh(Geometry(file=stl))


def test_mesh_file_in_mm_is_scaled(tmp_path):
    import trimesh

    path = tmp_path / "box.stl"
    trimesh.creation.box(extents=[1000, 200, 100]).export(path)
    mesh = build_mesh(Geometry(file=path, unit_scale=1e-3))
    assert mesh.volume == pytest.approx(1.0 * 0.2 * 0.1, rel=1e-9)


# --- Rigid body --------------------------------------------------------------


def test_skew_is_cross_product():
    v, w = np.array([1.0, 2.0, 3.0]), np.array([-2.0, 0.5, 4.0])
    np.testing.assert_allclose(skew(v) @ w, np.cross(v, w))


def test_uniform_sphere_inertia():
    v = vehicle({"primitive": "sphere", "radius": 0.1}, mass=4.0)
    props = mass_properties(v, build_mesh(v.geometry))
    np.testing.assert_allclose(props.r_g.value, 0, atol=1e-9)
    expected = 2 / 5 * 4.0 * 0.1**2
    np.testing.assert_allclose(
        props.inertia_g.value, expected * np.eye(3), rtol=MESH_RTOL, atol=1e-9
    )


def test_uniform_cylinder_inertia():
    m, length, r = 30.0, 1.6, 0.1
    v = vehicle({"primitive": "cylinder", "length": length, "radius": r}, mass=m)
    props = mass_properties(v, build_mesh(v.geometry))
    ixx = m * r**2 / 2
    iyy = m * (3 * r**2 + length**2) / 12
    np.testing.assert_allclose(np.diag(props.inertia_g.value), [ixx, iyy, iyy], rtol=MESH_RTOL)


def test_point_parts_shift_cog_and_add_inertia():
    # Hull 6 kg uniform sphere at the origin plus a 4 kg part 0.2 m below.
    v = vehicle(
        {"primitive": "sphere", "radius": 0.1},
        mass=10.0,
        parts=[{"mass": 4.0, "position": [0, 0, 0.2]}],
    )
    props = mass_properties(v, build_mesh(v.geometry))
    np.testing.assert_allclose(props.r_g.value, [0, 0, 0.08], atol=1e-9)
    hull = 2 / 5 * 6.0 * 0.1**2
    expected_o = hull * np.eye(3) + point_inertia(4.0, [0, 0, 0.2])
    np.testing.assert_allclose(props.inertia_o.value, expected_o, rtol=MESH_RTOL, atol=1e-9)
    # Parallel-axis theorem, both directions.
    np.testing.assert_allclose(
        props.inertia_o.value, props.inertia_g.value + point_inertia(10.0, props.r_g.value)
    )


def test_all_mass_in_parts_has_no_assumption():
    v = vehicle(
        {"primitive": "sphere", "radius": 0.1},
        mass=2.0,
        parts=[{"mass": 2.0, "position": [0.1, 0, 0]}],
    )
    props = mass_properties(v, build_mesh(v.geometry))
    assert props.r_g.source is Source.MEASURED
    np.testing.assert_allclose(props.r_g.sigma, 0)


def test_rigid_body_mass_matrix_is_symmetric_positive_definite():
    v = vehicle(
        {"primitive": "cylinder", "length": 1.0, "radius": 0.1},
        mass=20.0,
        parts=[{"mass": 5.0, "position": [0.2, 0.01, 0.04]}],
    )
    m_rb = rigid_body_mass_matrix(mass_properties(v, build_mesh(v.geometry)))
    np.testing.assert_allclose(m_rb, m_rb.T, atol=1e-12)
    assert np.all(np.linalg.eigvalsh(m_rb) > 0)
    # Off-diagonal block is -m S(r_g): check one entry by hand.
    m, (_, yg, zg) = 20.0, np.array([0.2, 0.01, 0.04]) * 5 / 20
    assert m_rb[0, 4] == pytest.approx(m * zg)  # -m S(r_g)[0,1] = m zg
    assert m_rb[1, 3] == pytest.approx(-m * zg)
    assert m_rb[2, 3] == pytest.approx(m * yg)


# --- Hydrostatics ------------------------------------------------------------


def test_neutral_sphere_buoyancy():
    m = build_model(load_vehicle(EXAMPLES / "sphere.yaml"))
    volume = 4 / 3 * np.pi * 0.1**3
    assert m.statics.buoyancy.value == pytest.approx(1025 * GRAVITY * volume, rel=MESH_RTOL)
    assert m.statics.weight.value == pytest.approx(4.29 * GRAVITY)
    assert not any("neutrally buoyant" in w for w in m.warnings)


def test_restoring_forces_upright_neutral_is_zero():
    g = restoring_forces(0.0, 0.0, 100.0, 100.0, np.zeros(3), np.zeros(3))
    np.testing.assert_allclose(g, 0, atol=1e-12)


def test_restoring_forces_heavy_vehicle_sinks():
    g = restoring_forces(0.0, 0.0, 110.0, 100.0, np.zeros(3), np.zeros(3))
    # tau = g + M nu_dot ...: with no thrust, M nu_dot = -g, so heave acceleration +z (down).
    assert -g[2] == pytest.approx(10.0)


@pytest.mark.parametrize("phi", [0.3, -0.3, np.pi / 2])
def test_cg_below_cb_rights_the_roll(phi):
    w = b = 100.0
    zg = 0.05  # CG 5 cm below CB
    g = restoring_forces(phi, 0.0, w, b, np.array([0, 0, zg]), np.zeros(3))
    assert g[3] == pytest.approx(zg * w * np.sin(phi))  # metacentric moment
    assert np.sign(-g[3]) == -np.sign(phi)  # restoring moment opposes the roll


def test_cg_below_cb_rights_the_pitch():
    g = restoring_forces(0.0, 0.2, 100.0, 100.0, np.array([0, 0, 0.05]), np.zeros(3))
    assert -g[4] < 0  # nose-up pitch (theta > 0) gets a nose-down moment


# --- Added mass --------------------------------------------------------------


def test_lamb_integrals_sum_to_two():
    assert sum(lamb_integrals(1.0, 0.7, 0.3)) == pytest.approx(2.0, abs=1e-9)


def test_sphere_added_mass_is_half_displaced_mass():
    rho, r = 1025.0, 0.1
    m = ellipsoid_added_mass(r, r, r, rho)
    half = 0.5 * rho * 4 / 3 * np.pi * r**3
    np.testing.assert_allclose(m, [half, half, half, 0, 0, 0], rtol=1e-9, atol=1e-12)


@pytest.mark.parametrize("ld", [1.5, 2.0, 5.0, 10.0])
def test_prolate_spheroid_integrals_match_closed_form(ld):
    a, b = ld / 2, 0.5
    e = np.sqrt(1 - (b / a) ** 2)
    log = np.log((1 + e) / (1 - e))
    alpha = 2 * (1 - e**2) / e**3 * (log / 2 - e)
    beta = 1 / e**2 - (1 - e**2) / (2 * e**3) * log
    np.testing.assert_allclose(lamb_integrals(a, b, b), [alpha, beta, beta], rtol=1e-9)


# Lamb's coefficients for prolate spheroids (Lamb, Hydrodynamics; tabulated in Newman,
# Marine Hydrodynamics): length/diameter -> k1 (axial), k2 (lateral), k' (rotation).
LAMB_TABLE = [(2.0, 0.210, 0.704, 0.240), (4.0, 0.082, 0.860, 0.608), (10.0, 0.021, 0.960, 0.883)]


@pytest.mark.parametrize(("ld", "k1", "k2", "kr"), LAMB_TABLE)
def test_prolate_spheroid_matches_lamb_table(ld, k1, k2, kr):
    a, b = ld / 2, 0.5
    volume = 4 / 3 * np.pi * a * b * b
    m = ellipsoid_added_mass(a, b, b, 1.0)
    fluid_inertia = volume * (a**2 + b**2) / 5  # displaced fluid's moment of inertia
    assert m[0] / volume == pytest.approx(k1, abs=0.002)
    assert m[1] / volume == pytest.approx(k2, abs=0.002)
    assert m[2] == pytest.approx(m[1])
    assert m[3] == 0.0  # rolling a body of revolution moves no fluid
    assert m[4] / fluid_inertia == pytest.approx(kr, abs=0.002)
    assert m[5] == pytest.approx(m[4])


def test_flat_ellipsoid_has_large_heave_added_mass():
    m = ellipsoid_added_mass(0.5, 0.5, 0.05, 1000.0)  # a disc-like body, thin in z
    assert m[2] > 5 * m[0]
    assert m[3] > 0  # rolling a non-circular section moves fluid


# --- Whole model -------------------------------------------------------------


@pytest.mark.parametrize("name", ["sphere", "ellipsoid", "remus100", "test_auv"])
def test_examples_build_and_serialise(name):
    import yaml

    model = build_model(load_vehicle(EXAMPLES / f"{name}.yaml"))
    data = to_dict(model)
    yaml.safe_dump(data)  # must be plain data
    assert data["mass"]["source"] == "measured"
    assert np.array(data["M_RB"]).shape == (6, 6)


def test_ellipsoid_example_is_self_righting():
    model = build_model(load_vehicle(EXAMPLES / "ellipsoid.yaml"))
    assert not any("self-righting" in w for w in model.warnings)
    assert model.m_a is not None
