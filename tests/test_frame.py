"""Open-frame vehicles: voxel hulls of CAD assemblies, axes, frame hydrodynamics, BlueROV2."""

import sys
from pathlib import Path

import numpy as np
import pytest
import trimesh

from robomodel.model import build_model
from robomodel.vehicle import axes_matrix, parse_vehicle
from robomodel.voxel import voxelize

ROOT = Path(__file__).resolve().parent.parent


def two_box_assembly(path):
    """An 'ROV' made of two separate closed boxes and one open plate (not watertight)."""
    a = trimesh.creation.box(extents=[400, 100, 100])  # mm
    a.apply_translation([0, 150, 0])
    b = trimesh.creation.box(extents=[400, 100, 100])
    b.apply_translation([0, -150, 0])
    plate = trimesh.Trimesh(
        vertices=[[-200, -200, 60], [200, -200, 60], [200, 200, 60]], faces=[[0, 1, 2]]
    )
    trimesh.util.concatenate([a, b, plate]).export(path)


def test_voxel_hull_of_closed_boxes():
    box = trimesh.creation.box(extents=[0.4, 0.2, 0.1])
    hull = voxelize(box, 0.005)
    assert hull.volume == pytest.approx(0.4 * 0.2 * 0.1, rel=0.12)  # thickened surface
    np.testing.assert_allclose(hull.center_mass, 0, atol=0.003)
    assert hull.projection(2)[1].sum() == pytest.approx(0.4 * 0.2, rel=0.08)
    ixx = 0.4 * 0.2 * 0.1 * (0.2**2 + 0.1**2) / 12
    assert hull.moment_inertia[0, 0] == pytest.approx(ixx, rel=0.25)


def test_assembly_file_becomes_one_solid(tmp_path):
    path = tmp_path / "rov.stl"
    two_box_assembly(path)
    v = parse_vehicle(
        {
            "name": "rov",
            "category": "underwater",
            "mass": 8.0,
            "water": "fresh",
            "geometry": {"file": str(path), "unit": "mm", "assembly": True, "voxel": 5},
        }
    )
    model = build_model(v)
    assert model.hull.volume == pytest.approx(2 * 0.4 * 0.1 * 0.1, rel=0.2)
    assert model.body_lift is None  # frame method, not slender-body lift
    assert model.m_a.source.value in ("bem", "empirical")  # BEM if installed, else ellipsoid
    assert not any("longest along" in w for w in model.warnings)


def test_axes_rotate_the_cad_into_the_body_frame(tmp_path):
    path = tmp_path / "tall.stl"
    trimesh.creation.cylinder(radius=0.1, height=1.5).export(path)  # long axis = file z
    np.testing.assert_allclose(np.linalg.det(axes_matrix(("z", "-x", "-y"))), 1.0)
    v = parse_vehicle(
        {
            "name": "t",
            "category": "underwater",
            "mass": 47,
            "geometry": {"file": str(path), "axes": ["z", "x", "y"]},
        }
    )
    model = build_model(v)
    assert np.argmax(model.hull.extents) == 0
    assert not any("longest along" in w for w in model.warnings)
    with pytest.raises(ValueError, match="right-handed"):
        parse_vehicle(
            {
                "name": "t",
                "category": "underwater",
                "mass": 47,
                "geometry": {"file": str(path), "axes": ["z", "x", "-y"]},
            }
        )


def test_measured_volume_replaces_cad_volume():
    v = parse_vehicle(
        {
            "name": "s",
            "category": "underwater",
            "mass": 4.0,
            "geometry": {"primitive": "sphere", "radius": 0.1},
            "measured": {"volume": 0.004},
        }
    )
    statics = build_model(v).statics
    assert statics.volume.value == 0.004 and statics.volume.source.value == "measured"


def test_stubby_box_uses_frame_method(tmp_path):
    path = tmp_path / "box.stl"
    trimesh.creation.box(extents=[0.5, 0.4, 0.3]).export(path)
    model = build_model(
        parse_vehicle(
            {"name": "b", "category": "underwater", "mass": 60, "geometry": {"file": str(path)}}
        )
    )
    assert model.damping.axial_cd == pytest.approx(1.1)  # frame drag
    expected = 0.5 * 1000 * 1.1 * 0.4 * 0.3  # sea water default is 1025
    assert model.damping.quadratic.value[0] == pytest.approx(expected * 1.025, rel=0.08)


BLUEROV2 = ROOT / "examples" / "bluerov2_heavy.yaml"


@pytest.mark.skipif(
    not Path(
        "data/bluerov2/BROV2-HEAVY-ASM-BROV2-HEAVY-R1.STL"
    ).exists(),
    reason="BlueROV2 CAD (309 MB, Blue Robotics) not downloaded",
)
def test_bluerov2_geometry_matches_paper_and_damping_is_low():
    sys.path.insert(0, str(ROOT / "validation"))
    import bluerov2

    model, areas = bluerov2.run()
    for name, ours in areas.items():
        assert ours == pytest.approx(bluerov2.PAPER_AREAS[name], rel=0.10)
    quad = np.asarray(model.damping.quadratic.value)
    # Known result: CAD-only frame drag is 1.5-5x below the pool-identified values.
    ratio = quad / bluerov2.PAPER_QUAD
    assert np.all((ratio > 0.15) & (ratio < 0.8))


def test_model_does_not_depend_on_the_cad_origin(tmp_path):
    """The same hull with its CAD origin far away must behave the same (stress test bug)."""
    from robomodel.checks import FAIL, run_checks
    from robomodel.dynamics import linearize
    from robomodel.rigid_body import velocity_transform

    base = trimesh.creation.cylinder(radius=0.1, height=1.0)
    far = base.copy()
    far.apply_translation([25, 10, -5])
    poles = []
    for name, mesh in (("c0", base), ("c1", far)):
        path = tmp_path / f"{name}.stl"
        mesh.export(path)
        centre = np.asarray(mesh.center_mass)[[2, 0, 1]]
        v = parse_vehicle(
            {
                "name": name,
                "category": "underwater",
                "mass": 1000 * base.volume,
                "water": "fresh",
                "geometry": {"file": str(path), "axes": ["z", "x", "y"]},
                "measured": {"r_g": list(centre + [0, 0, 0.01])},
            }
        )
        dyn = build_model(v).dynamics()
        assert not [c for c in run_checks(dyn) if c.status == FAIL]
        state = np.zeros(12)
        state[6:] = np.linalg.solve(velocity_transform(dyn.r_hydro), [1.0, 0, 0, 0, 0, 0])
        a, _ = linearize(dyn, state, np.zeros(6))
        poles.append(np.sort(np.linalg.eigvals(a[6:, 6:]).real))
    np.testing.assert_allclose(poles[0], poles[1], atol=0.01)
