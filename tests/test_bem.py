"""Boundary element added mass (optional extra: pip install "robomodel[bem]")."""

import numpy as np
import pytest
import trimesh

from robomodel import bem
from robomodel.added_mass import ellipsoid_added_mass
from robomodel.geometry import build_mesh
from robomodel.model import build_model
from robomodel.vehicle import Geometry, parse_vehicle

pytestmark = pytest.mark.skipif(not bem.available(), reason="robomodel[bem] not installed")


@pytest.mark.parametrize("axes", [(0.1, 0.1, 0.1), (0.5, 0.1, 0.1), (0.5, 0.2, 0.08)])
def test_bem_matches_lamb_on_ellipsoids(axes):
    mesh = build_mesh(Geometry(primitive="ellipsoid", dims=dict(zip("abc", axes, strict=True))))
    m = np.diag(bem.bem_added_mass(mesh, 1000.0, mesh.center_mass))
    exact = ellipsoid_added_mass(*axes, 1000.0)
    for i in range(6):
        if exact[i] > 1e-6:
            assert m[i] == pytest.approx(exact[i], rel=0.05)


def test_sliver_cad_gets_even_panels():
    """A CAD cylinder side is a few 1.6 m long triangles; the panels must be even."""
    cyl = build_mesh(Geometry(primitive="cylinder", dims=dict(length=1.6, radius=0.095)))
    assert bem.panel_aspect(cyl) > bem.MAX_ASPECT
    panels = bem.bem_panels(cyl)
    assert bem.panel_aspect(panels) < bem.MAX_ASPECT
    assert panels.volume == pytest.approx(cyl.volume, rel=1e-6)
    assert len(panels.faces) <= bem.PANELS


def test_auto_uses_bem_for_box_like_cad(tmp_path):
    hull = trimesh.creation.box(extents=[0.6, 0.4, 0.3])
    path = tmp_path / "box.stl"
    hull.export(path)
    v = parse_vehicle(
        {
            "name": "b",
            "category": "underwater",
            "mass": 70.0,
            "water": "fresh",
            "geometry": {"file": str(path)},
        }
    )
    model = build_model(v)
    assert model.m_a.source.value == "bem"
    np.testing.assert_allclose(model.m_a.value, model.m_a.value.T, atol=1e-9)
    assert np.all(np.linalg.eigvalsh(model.m_a.value) > 0)
