"""Acceptance tests for added mass on any hull, damping, thrusters and the simulator.

Every test checks the model against physics with a known answer (docs/plan.md, K3-K7).
"""

from pathlib import Path

import numpy as np
import pytest
import trimesh

from robomodel.added_mass import ellipsoid_added_mass, strip_theory_added_mass
from robomodel.damping import axial_drag_coefficient, estimate_damping
from robomodel.dynamics import (
    Dynamics,
    coriolis_matrix,
    euler_rate_matrix,
    linearize,
    rotation_zyx,
    simulate,
)
from robomodel.geometry import build_mesh
from robomodel.model import build_model
from robomodel.rigid_body import part_inertia, point_inertia
from robomodel.sections import slice_hull
from robomodel.thrusters import allocation_matrix, thrust
from robomodel.vehicle import Geometry, load_vehicle, parse_vehicle

EXAMPLES = Path(__file__).parent.parent / "examples"
RHO = 1025.0


def make_vehicle(**overrides):
    data = {
        "name": "t",
        "category": "underwater",
        "mass": 10.0,
        "geometry": {"primitive": "sphere", "radius": 0.1},
    }
    data.update(overrides)
    return parse_vehicle(data)


def simple_dynamics(**overrides) -> Dynamics:
    """Diagonal test body: easy to reason about by hand."""
    base = dict(
        m_rb=np.diag([10.0, 10.0, 10.0, 1.0, 2.0, 2.0]),
        m_a=np.diag([1.0, 5.0, 5.0, 0.0, 0.5, 0.5]),
        d_lin=np.zeros(6),
        d_quad=np.zeros(6),
        weight=98.0,
        buoyancy=98.0,
        r_g=np.zeros(3),
        r_b=np.zeros(3),
        b_thr=np.zeros((6, 0)),
    )
    base.update(overrides)
    return Dynamics(**base)


# --- Parts -------------------------------------------------------------------


def test_box_and_cylinder_part_inertia():
    v = make_vehicle(
        parts=[
            {"mass": 2.0, "position": [0, 0, 0], "shape": "box", "size": [0.3, 0.1, 0.2]},
            {
                "mass": 1.0,
                "position": [0, 0, 0],
                "shape": "cylinder",
                "radius": 0.05,
                "length": 0.4,
                "axis": "y",
            },
        ]
    )
    box, cyl = (part_inertia(p) for p in v.parts)
    np.testing.assert_allclose(np.diag(box), 2 / 12 * np.array([0.05, 0.13, 0.10]))
    transverse = (3 * 0.05**2 + 0.4**2) / 12
    np.testing.assert_allclose(np.diag(cyl), [transverse, 0.05**2 / 2, transverse])


def test_shaped_part_adds_own_inertia_to_vehicle():
    point = make_vehicle(parts=[{"mass": 5.0, "position": [0, 0, 0.05]}])
    boxed = make_vehicle(
        parts=[{"mass": 5.0, "position": [0, 0, 0.05], "shape": "box", "size": [0.2, 0.1, 0.1]}]
    )
    a, b = (build_model(v).mass.inertia_o.value for v in (point, boxed))
    np.testing.assert_allclose(b - a, part_inertia(boxed.parts[0]), atol=1e-12)


def test_bad_part_shape_rejected():
    with pytest.raises(ValueError, match="shape"):
        make_vehicle(parts=[{"mass": 1, "position": [0, 0, 0], "shape": "cone"}])
    with pytest.raises(ValueError, match="axis"):
        make_vehicle(
            parts=[
                {
                    "mass": 1,
                    "position": [0, 0, 0],
                    "shape": "cylinder",
                    "radius": 0.1,
                    "length": 0.2,
                    "axis": "w",
                }
            ]
        )


# --- Added mass on any hull --------------------------------------------------


@pytest.mark.parametrize("ld", [2.0, 5.0, 10.0])
def test_strip_theory_on_spheroid_mesh_matches_lamb(tmp_path, ld):
    """A spheroid given as a mesh file (no 'ellipsoid' shortcut) must still match Lamb."""
    a, b = 0.1 * ld, 0.1
    path = tmp_path / "spheroid.stl"
    build_mesh(Geometry(primitive="ellipsoid", dims={"a": a, "b": b, "c": b})).export(path)
    mesh = build_mesh(Geometry(file=path))
    m = strip_theory_added_mass(slice_hull(mesh), mesh.volume, RHO)
    exact = ellipsoid_added_mass(a, b, b, RHO)
    np.testing.assert_allclose(np.diag(m)[[0, 1, 2, 4, 5]], exact[[0, 1, 2, 4, 5]], rtol=0.01)
    assert abs(m[1, 5]) < 1e-6 * m[1, 1]  # symmetric body: no sway-yaw coupling


def test_strip_theory_flat_hull_heave_bigger_than_sway():
    mesh = build_mesh(Geometry(primitive="ellipsoid", dims={"a": 0.6, "b": 0.2, "c": 0.08}))
    m = strip_theory_added_mass(slice_hull(mesh), mesh.volume, RHO)
    exact = ellipsoid_added_mass(0.6, 0.2, 0.08, RHO)
    assert m[2, 2] > 2 * m[1, 1]
    np.testing.assert_allclose(np.diag(m)[1:3], exact[1:3], rtol=0.1)  # within its sigma


def test_strip_theory_couplings_for_asymmetric_hull():
    """Blunt nose forward: more sway added mass ahead of the origin, so m26 > 0."""
    mesh = build_mesh(
        Geometry(
            primitive="myring", dims=dict(a=0.191, b=0.654, c=0.541, d=0.191, n=2.0, theta=0.436)
        )
    )
    m = strip_theory_added_mass(slice_hull(mesh), mesh.volume, RHO)
    assert m[1, 5] > 0 and m[2, 4] < 0
    assert m[1, 5] == pytest.approx(-m[2, 4])
    np.testing.assert_allclose(m, m.T)
    assert np.all(np.linalg.eigvalsh(m + np.diag([0, 0, 0, 1e-9, 0, 0])) > 0)


# --- Damping -----------------------------------------------------------------


def test_hoerner_drag_for_streamlined_body():
    # Hoerner's formula at L/D = 7 with C_f = 0.0045: about 0.11
    assert axial_drag_coefficient(7.0) == pytest.approx(0.0045 * (21 + 4.5 / 7**0.5 + 21 / 49))
    assert axial_drag_coefficient(2.0) == pytest.approx(0.9)  # bluff
    assert axial_drag_coefficient(7.0, end_fraction=1.0) == pytest.approx(0.9)  # flat ends


def test_cylinder_crossflow_damping():
    length, r = 1.6, 0.1
    mesh = build_mesh(Geometry(primitive="cylinder", dims={"length": length, "radius": r}))
    d = estimate_damping(slice_hull(mesh), RHO)
    side = 2 * r * length
    assert d.quadratic.value[1] == pytest.approx(0.5 * RHO * 1.1 * side, rel=0.01)
    assert d.quadratic.value[2] == pytest.approx(d.quadratic.value[1], rel=1e-6)
    # yaw: 0.5 rho C_d 2r * integral |x|^3 dx = 0.5 rho C_d 2r * 2 (L/2)^4 / 4
    yaw = 0.5 * RHO * 1.1 * 2 * r * 2 * (length / 2) ** 4 / 4
    assert d.quadratic.value[5] == pytest.approx(yaw, rel=0.01)
    assert d.quadratic.value[3] == pytest.approx(0.0, abs=1e-9)  # round: no roll drag
    assert d.linear.value[3] > 0  # but some roll damping remains


def test_box_hull_has_square_crossflow_and_roll_drag(tmp_path):
    path = tmp_path / "box.stl"
    trimesh.creation.box(extents=[1.0, 0.4, 0.4]).export(path)
    sections = slice_hull(build_mesh(Geometry(file=path)))
    d = estimate_damping(sections, RHO)
    assert d.quadratic.value[1] == pytest.approx(0.5 * RHO * 2.0 * 0.4 * 1.0, rel=0.01)


# --- Thrusters ---------------------------------------------------------------


def four_thrusters():
    return make_vehicle(
        thrusters=[
            {
                "model": "x",
                "position": [-0.5, 0.2, 0],
                "direction": [1, 0, 0],
                "max_forward": 40,
                "max_reverse": 30,
            },
            {
                "model": "x",
                "position": [-0.5, -0.2, 0],
                "direction": [1, 0, 0],
                "max_forward": 40,
                "max_reverse": 30,
            },
            {
                "model": "x",
                "position": [0.3, 0, 0],
                "direction": [0, 0, 1],
                "max_forward": 40,
                "max_reverse": 30,
            },
            {
                "model": "x",
                "position": [-0.3, 0, 0],
                "direction": [0, 0, 1],
                "max_forward": 40,
                "max_reverse": 30,
            },
        ]
    ).thrusters


def test_allocation_pure_surge_and_yaw():
    t = four_thrusters()
    b = allocation_matrix(t)
    np.testing.assert_allclose(b @ [1, 1, 0, 0], [2, 0, 0, 0, 0, 0], atol=1e-12)
    # Right thruster (y = +0.2) pushing forward turns the nose left: negative yaw (z down).
    np.testing.assert_allclose(b @ [1, -1, 0, 0], [0, 0, 0, 0, 0, -0.4], atol=1e-12)
    # Front vertical thruster pushing down (+z) pitches the nose down: negative pitch.
    np.testing.assert_allclose(b @ [0, 0, 1, -1], [0, 0, 0, 0, -0.6, 0], atol=1e-12)


def test_thrust_curve_limits_and_asymmetry():
    t = four_thrusters()
    f = thrust(t, [1.0, -1.0, 0.5, 2.0])
    np.testing.assert_allclose(f, [40, -30, 10, 40])


def test_t200_preset_is_marked():
    v = make_vehicle(thrusters=[{"model": "T200", "position": [0, 0, 0], "direction": [1, 0, 0]}])
    assert v.thrusters[0].preset
    w = make_vehicle(
        thrusters=[
            {
                "model": "T200",
                "position": [0, 0, 0],
                "direction": [1, 0, 0],
                "max_forward": 45,
                "max_reverse": 35,
            }
        ]
    )
    assert not w.thrusters[0].preset and w.thrusters[0].max_forward == 45


def test_unknown_thruster_needs_limits():
    with pytest.raises(ValueError, match="max_forward"):
        make_vehicle(thrusters=[{"model": "mine", "position": [0, 0, 0], "direction": [1, 0, 0]}])


# --- Kinematics and Coriolis -------------------------------------------------


def test_rotation_is_orthonormal_and_yaw_turns_x_to_east():
    r = rotation_zyx(0.3, -0.2, 1.1)
    np.testing.assert_allclose(r @ r.T, np.eye(3), atol=1e-12)
    np.testing.assert_allclose(rotation_zyx(0, 0, np.pi / 2) @ [1, 0, 0], [0, 1, 0], atol=1e-12)
    np.testing.assert_allclose(euler_rate_matrix(0, 0), np.eye(3))


def test_coriolis_does_no_work():
    model = build_model(load_vehicle(EXAMPLES / "test_auv.yaml"))
    m = model.dynamics().mass_matrix
    rng = np.random.default_rng(0)
    for _ in range(5):
        nu = rng.normal(size=6)
        c = coriolis_matrix(m, nu)
        assert nu @ c @ nu == pytest.approx(0.0, abs=1e-9)
        np.testing.assert_allclose(c, -c.T, atol=1e-12)


def test_munk_moment_destabilises_yaw():
    """Moving forward with a sideslip, a slender hull turns further away (Munk moment).

    Fossen: C_A(nu) nu has yaw component (X_udot - Y_vdot) u v = (m22 - m11) u v, so the
    moment on the right-hand side is -(m22 - m11) u v. With u > 0 and v > 0 (velocity to the
    right of the nose) the nose turns left (r_dot < 0), away from the velocity: unstable.
    """
    m = np.diag([10, 10, 10, 1, 2, 2.0]) + np.diag([1, 8, 8, 0, 1, 1.0])
    nu = np.array([1.0, 0.1, 0, 0, 0, 0])
    yaw_moment = -(coriolis_matrix(m, nu) @ nu)[5]  # moved to the right-hand side
    assert yaw_moment == pytest.approx(-(18 - 11) * 1.0 * 0.1)


# --- Simulation --------------------------------------------------------------


def test_neutral_body_at_rest_stays_put():
    traj = simulate(simple_dynamics(), 60.0)
    assert np.max(np.abs(traj.eta[-1])) < 1e-3  # acceptance: < 1 mm in 60 s


def test_kinetic_energy_conserved_without_damping():
    dyn = simple_dynamics()
    nu0 = np.array([0.5, 0.2, -0.1, 0.3, 0.2, -0.4])
    traj = simulate(dyn, 20.0, state0=np.concatenate([np.zeros(6), nu0]))
    m = dyn.mass_matrix
    energy = 0.5 * np.einsum("ij,jk,ik->i", traj.nu, m, traj.nu)
    assert np.max(np.abs(energy / energy[0] - 1)) < 1e-3  # acceptance: < 0.1 %


def test_terminal_surge_speed():
    d_lin, d_quad, force = 2.0, 20.0, 30.0
    dyn = simple_dynamics(d_lin=np.full(6, d_lin), d_quad=np.full(6, d_quad))
    traj = simulate(dyn, 60.0, tau=np.array([force, 0, 0, 0, 0, 0]))
    expected = (-d_lin + np.sqrt(d_lin**2 + 4 * d_quad * force)) / (2 * d_quad)
    assert traj.nu[-1, 0] == pytest.approx(expected, rel=1e-3)  # acceptance: < 1 %
    assert traj.eta[-1, 0] > 0 and abs(traj.eta[-1, 1]) < 1e-9  # straight ahead, north


def test_heavy_body_sinks_with_expected_acceleration():
    dyn = simple_dynamics(weight=108.0, buoyancy=98.0)
    traj = simulate(dyn, 0.5, dt_out=0.01)
    accel = (108.0 - 98.0) / (10.0 + 5.0)  # added mass in heave counts too
    assert traj.eta[-1, 2] == pytest.approx(0.5 * accel * 0.5**2, rel=1e-3)  # z down


def test_roll_oscillation_period_from_metacentric_height():
    zg, w = 0.02, 98.0
    dyn = simple_dynamics(r_g=np.array([0, 0, zg]))
    phi0 = 0.05
    traj = simulate(dyn, 20.0, state0=np.array([0, 0, 0, phi0, 0, 0] + [0] * 6), dt_out=0.002)
    # Small-angle pendulum: period = 2 pi sqrt(I_roll / (zg W)), I_roll = 1 (+0 added).
    # The CG is not at the origin, so use the inertia about the CG's roll axis: I_o - m zg^2.
    inertia = 1.0 - 10.0 * zg**2
    period = 2 * np.pi * np.sqrt(inertia / (zg * w))
    phi = traj.eta[:, 3]
    crossings = traj.t[1:][(phi[:-1] > 0) & (phi[1:] <= 0)]
    assert np.mean(np.diff(crossings)) == pytest.approx(period, rel=0.01)


def test_example_auv_under_thrust_goes_forward():
    model = build_model(load_vehicle(EXAMPLES / "test_auv.yaml"))
    dyn = model.dynamics()
    tau = dyn.b_thr @ thrust(model.vehicle.thrusters, [0.5])
    traj = simulate(dyn, 30.0, tau=tau)
    assert traj.nu[-1, 0] > 0.2
    assert abs(traj.eta[-1, 1]) < 1e-6  # symmetric vehicle keeps its heading


def test_linearization_at_rest():
    dyn = simple_dynamics(d_lin=np.full(6, 2.0))
    a, b = linearize(dyn, np.zeros(12), np.zeros(6))
    np.testing.assert_allclose(a[:6, 6:], np.eye(6), atol=1e-8)  # eta_dot = nu at rest
    np.testing.assert_allclose(b[6:], np.linalg.inv(dyn.mass_matrix), atol=1e-8)
    np.testing.assert_allclose(a[6, 6], -2.0 / 11.0, rtol=1e-6)  # -d / (m + m_a)


# --- MATLAB export -----------------------------------------------------------


def test_matlab_export_files_and_values(tmp_path):
    from scipy.io import loadmat

    from robomodel.cli import main
    from robomodel.matlab_export import matlab_name

    assert (
        main(["matlab", str(EXAMPLES / "test_auv.yaml"), "-o", str(tmp_path), "--speed", "1.0"])
        == 0
    )
    names = {p.name for p in tmp_path.iterdir()}
    assert {
        "test_auv_params.m",
        "robomodel_dynamics.m",
        "robomodel_thrust.m",
        "test_auv_example.m",
        "test_auv_reference.csv",
        "test_auv_linear.mat",
    } <= names

    model = build_model(load_vehicle(EXAMPLES / "test_auv.yaml"))
    params = (tmp_path / "test_auv_params.m").read_text()
    assert f"{model.dynamics().weight:.17g}" in params  # full precision, no rounding

    lin = loadmat(tmp_path / "test_auv_linear.mat")
    assert lin["A"].shape == (12, 12) and lin["B"].shape == (12, 6)
    assert lin["Bu"].shape == (12, 1)
    ref = np.loadtxt(tmp_path / "test_auv_reference.csv", delimiter=",", skiprows=1)
    assert ref.shape[1] == 13 and ref[-1, 7] > 0.2  # moving forward at the end
    assert matlab_name("my-AUV 2") == "my_AUV_2" and matlab_name("2x") == "v_2x"


# --- Fins and measured values ------------------------------------------------


def tail_fins(control=True):
    """Cross-shaped tail: two vertical rudders (normal y) and two stern planes (normal z)."""
    x = -0.7
    common = {"area": 0.0066, "span": 0.08, "control": control}
    return [
        {"name": "rudder_top", "position": [x, 0, -0.14], "normal": [0, 1, 0], **common},
        {"name": "rudder_bottom", "position": [x, 0, 0.14], "normal": [0, 1, 0], **common},
        {"name": "plane_port", "position": [x, -0.14, 0], "normal": [0, 0, 1], **common},
        {"name": "plane_stbd", "position": [x, 0.14, 0], "normal": [0, 0, 1], **common},
    ]


def finned_auv(fins):
    data = {
        "name": "finned",
        "category": "underwater",
        "mass": 45.0,
        "water": "fresh",
        "geometry": {"primitive": "cylinder", "length": 1.6, "radius": 0.095},
        "parts": [{"mass": 6.0, "position": [0.0, 0.0, 0.04]}],
        "fins": fins,
    }
    return build_model(parse_vehicle(data))


def test_helmbold_lift_slope_limits():
    from robomodel.fins import lift_slope

    assert lift_slope(1e-6) == pytest.approx(np.pi / 2 * 1e-6, rel=1e-3)  # slender wing
    assert lift_slope(1e6) == pytest.approx(2 * np.pi, rel=1e-3)  # thin aerofoil theory
    assert 0 < lift_slope(2.0) < 2 * np.pi


def test_fin_damping_is_symmetric_and_dissipative():
    model = finned_auv(tail_fins())
    d = model.fins.d_fin.value
    np.testing.assert_allclose(d, d.T, atol=1e-12)
    assert np.all(np.linalg.eigvalsh(d) > -1e-12)


def test_tail_fins_damp_yaw_and_restore_heading():
    model = finned_auv(tail_fins())
    d = model.fins.d_fin.value
    assert d[5, 5] > 0 and d[4, 4] > 0  # yaw and pitch damping
    # Sideslip v > 0 at forward speed: tail lift turns the nose towards the velocity (+r).
    nu = np.array([1.0, 0.1, 0, 0, 0, 0])
    assert -(abs(nu[0]) * d @ nu)[5] > 0


def test_fins_make_the_hull_directionally_stable():
    """Bare hull: unstable in yaw (Munk moment). Big enough tail fins fix it."""
    speed = 1.0

    def slowest_pole(model):
        dyn = model.dynamics()
        state0 = np.zeros(12)
        state0[6] = speed
        tau0 = np.zeros(6)
        tau0[0] = dyn.d_lin[0] * speed + dyn.d_quad[0] * speed**2
        a, _ = linearize(dyn, state0, tau0)
        poles = np.linalg.eigvals(a[6:, 6:])  # velocity dynamics only
        return np.max(poles.real)

    bare = finned_auv([])
    big = [dict(f, area=0.03, span=0.15) for f in tail_fins(control=False)]
    assert slowest_pole(bare) > 0
    assert slowest_pole(finned_auv(big)) < 0


def test_rudder_deflection_turns_the_vehicle():
    model = finned_auv(tail_fins())
    dyn = model.dynamics()
    names = model.fins.control_names
    deflection = np.array([0.1 if n.startswith("rudder") else 0.0 for n in names])
    state0 = np.zeros(12)
    state0[6] = 1.0
    surge = dyn.d_lin[0] + dyn.d_quad[0]
    traj = simulate(
        dyn, 10.0, tau=np.array([surge, 0, 0, 0, 0, 0]), state0=state0, fin_deflection=deflection
    )
    # Rudders push the tail to +y, so the nose turns left: heading decreases.
    assert traj.eta[-1, 5] < -0.05
    no_turn = simulate(dyn, 10.0, tau=np.array([surge, 0, 0, 0, 0, 0]), state0=state0)
    assert abs(no_turn.eta[-1, 5]) < 1e-9


def test_measured_values_replace_cad_estimates():
    v = make_vehicle(measured={"r_g": [0.01, 0, 0.02], "inertia_g": [0.1, 0.5, 0.5]})
    props = build_model(v).mass
    np.testing.assert_allclose(props.r_g.value, [0.01, 0, 0.02])
    assert props.r_g.source.value == "measured"
    np.testing.assert_allclose(np.diag(props.inertia_g.value), [0.1, 0.5, 0.5])
    np.testing.assert_allclose(
        props.inertia_o.value, props.inertia_g.value + point_inertia(10.0, [0.01, 0, 0.02])
    )


def test_bad_measured_inertia_rejected():
    with pytest.raises(ValueError, match="inertia_g"):
        make_vehicle(measured={"inertia_g": [0.1, -0.5, 0.5]})


def test_body_lift_only_for_slender_hulls():
    slender = finned_auv([])
    assert slender.body_lift is not None
    b = slender.body_lift.value
    assert b[1, 1] > 0 and b[2, 2] > 0  # sideslip makes an opposing force
    assert b[1, 5] == 0  # rotation alone makes no body lift
    assert b[5, 1] < 0  # centre of pressure aft: the moment turns the nose into the flow
    stubby = build_model(make_vehicle())  # a sphere
    assert stubby.body_lift is None
