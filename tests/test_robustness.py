"""Odd and broken CAD files: the pipeline must repair, warn or explain, never give nonsense."""

import numpy as np
import pytest
import trimesh

from robomodel.checks import FAIL, run_checks
from robomodel.geometry import GeometryError
from robomodel.model import build_model
from robomodel.vehicle import parse_vehicle


def model_for(tmp_path, mesh, mass=None, **geometry):
    path = tmp_path / "cad.stl"
    mesh.export(path)
    if mass is None:
        mass = 1000 * abs(trimesh.load(path, force="mesh").convex_hull.volume) * 0.98
    v = parse_vehicle(
        {
            "name": "t",
            "category": "underwater",
            "mass": mass,
            "water": "fresh",
            "geometry": {"file": str(path), **geometry},
        }
    )
    return build_model(v)


def no_failures(model):
    return not [c for c in run_checks(model.dynamics()) if c.status == FAIL]


def test_open_shell_is_capped(tmp_path):
    shell = trimesh.creation.cylinder(radius=0.1, height=1.2)
    true_volume = shell.volume
    shell.update_faces(np.arange(len(shell.faces)) > 3)  # lose part of an end cap
    model = model_for(tmp_path, shell, axes=["z", "x", "y"])
    assert model.statics.volume.value == pytest.approx(true_volume, rel=0.03)
    assert any("closed holes" in w for w in model.warnings)
    assert no_failures(model)


def test_inverted_normals_are_fixed(tmp_path):
    mesh = trimesh.creation.cylinder(radius=0.1, height=1.2)
    true_volume = mesh.volume
    mesh.invert()
    model = model_for(tmp_path, mesh, axes=["z", "x", "y"])
    assert model.statics.volume.value == pytest.approx(true_volume, rel=1e-6)


def test_duplicate_faces_are_removed(tmp_path):
    s = trimesh.creation.icosphere(radius=0.2)
    dup = trimesh.Trimesh(
        np.vstack([s.vertices, s.vertices]),
        np.vstack([s.faces, s.faces + len(s.vertices)]),
        process=False,
    )
    model = model_for(tmp_path, dup)
    assert model.statics.volume.value == pytest.approx(s.volume, rel=1e-6)
    assert any("duplicate" in w for w in model.warnings)


def test_solid_inside_solid_is_not_counted_twice(tmp_path):
    outer = trimesh.creation.box(extents=[0.6, 0.4, 0.3])
    inner = trimesh.creation.box(extents=[0.3, 0.2, 0.1])  # e.g. a battery modelled inside
    model = model_for(tmp_path, trimesh.util.concatenate([outer, inner]))
    assert model.statics.volume.value == pytest.approx(outer.volume, rel=0.05)
    assert any("inside other solids" in w for w in model.warnings)


def test_flat_geometry_has_a_clear_error(tmp_path):
    flat = trimesh.Trimesh([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], [[0, 1, 2], [1, 3, 2]])
    with pytest.raises(GeometryError, match="no volume"):
        model_for(tmp_path, flat, mass=1.0)


def test_wrong_units_are_flagged(tmp_path):
    hull = trimesh.creation.cylinder(radius=95, height=1600)  # drawn in mm
    model = model_for(tmp_path, hull, mass=45.0, unit="m", axes=["z", "x", "y"])  # read as m
    assert any("units" in w for w in model.warnings)


def test_tilted_cad_is_flagged_and_rotate_fixes_it(tmp_path):
    hull = trimesh.creation.cylinder(radius=0.1, height=1.2)
    hull.apply_transform(trimesh.transformations.rotation_matrix(np.radians(90 - 20), [0, 1, 0]))
    tilted = model_for(tmp_path, hull)  # long axis 20 deg off body x
    assert any("tilted 20 deg" in w for w in tilted.warnings)
    straight = model_for(tmp_path, hull, rotate=[0, 20, 0])
    assert not any("tilted" in w for w in straight.warnings)


def test_separate_bodies_are_flagged(tmp_path):
    a = trimesh.creation.box(extents=[1.0, 0.1, 0.1])
    a.apply_translation([0, 0.3, 0])
    b = trimesh.creation.box(extents=[1.0, 0.1, 0.1])
    b.apply_translation([0, -0.3, 0])
    model = model_for(tmp_path, trimesh.util.concatenate([a, b]))
    assert any("2 separate bodies" in w for w in model.warnings)
    assert no_failures(model)


@pytest.mark.parametrize("shape", ["torus", "thin_plate", "noisy_scan", "tiny", "huge"])
def test_odd_shapes_give_valid_models(tmp_path, shape):
    rng = np.random.default_rng(0)
    mesh = {
        "torus": lambda: trimesh.creation.torus(major_radius=0.4, minor_radius=0.08),
        "thin_plate": lambda: trimesh.creation.box(extents=[0.3, 1.2, 0.004]),
        "noisy_scan": lambda: trimesh.creation.icosphere(radius=0.3, subdivisions=4),
        "tiny": lambda: trimesh.creation.icosphere(radius=0.02),
        "huge": lambda: trimesh.creation.cylinder(radius=1.5, height=20, sections=64),
    }[shape]()
    if shape == "noisy_scan":
        mesh.vertices += rng.normal(0, 0.01, mesh.vertices.shape)
    model = model_for(tmp_path, mesh)
    assert np.all(np.isfinite(model.dynamics().mass_matrix))
    assert no_failures(model)
