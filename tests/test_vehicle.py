from pathlib import Path

import numpy as np
import pytest

from robomodel.cli import main
from robomodel.vehicle import VehicleError, load_vehicle, parse_vehicle

EXAMPLES = Path(__file__).parent.parent / "examples"


def minimal(**overrides):
    data = {
        "name": "v",
        "category": "underwater",
        "mass": 10.0,
        "geometry": {"primitive": "sphere", "radius": 0.1},
    }
    data.update(overrides)
    return data


@pytest.mark.parametrize("name", ["sphere.yaml", "test_auv.yaml"])
def test_examples_load(name):
    vehicle = load_vehicle(EXAMPLES / name)
    assert vehicle.mass > 0


def test_defaults_to_sea_water():
    assert parse_vehicle(minimal()).water_density == 1025.0


def test_fresh_water_and_numeric_density():
    assert parse_vehicle(minimal(water="fresh")).water_density == 1000.0
    assert parse_vehicle(minimal(water=1010)).water_density == 1010.0


def test_thruster_direction_is_normalised():
    v = parse_vehicle(
        minimal(thrusters=[{"model": "T200", "position": [0, 0, 0], "direction": [2, 0, 0]}])
    )
    np.testing.assert_allclose(v.thrusters[0].direction, [1, 0, 0])


def test_collects_all_problems_at_once():
    with pytest.raises(VehicleError) as e:
        parse_vehicle({"name": "", "category": "boat", "mass": -1})
    text = str(e.value)
    for key in ("'name'", "'category'", "'mass'", "'geometry'"):
        assert key in text
    assert len(e.value.problems) == 4


@pytest.mark.parametrize(
    "geometry",
    [
        {"primitive": "sphere"},  # missing radius
        {"primitive": "cube", "side": 1},  # unknown primitive
        {"primitive": "sphere", "radius": 0.1, "file": "x.stl"},  # both given
        {"file": "does_not_exist.stl"},
    ],
)
def test_bad_geometry(geometry):
    with pytest.raises(VehicleError):
        parse_vehicle(minimal(geometry=geometry))


def test_mesh_file_resolved_relative_to_yaml(tmp_path):
    (tmp_path / "hull.stl").write_text("solid empty\nendsolid empty\n")
    (tmp_path / "v.yaml").write_text(
        "name: v\ncategory: underwater\nmass: 5\ngeometry: {file: hull.stl, unit: mm}\n"
    )
    g = load_vehicle(tmp_path / "v.yaml").geometry
    assert g.file == (tmp_path / "hull.stl").resolve()
    assert g.unit_scale == 1e-3


def test_parts_heavier_than_vehicle_rejected():
    with pytest.raises(VehicleError, match="more than the vehicle mass"):
        parse_vehicle(minimal(parts=[{"mass": 11, "position": [0, 0, 0]}]))


def test_zero_thruster_direction_rejected():
    with pytest.raises(VehicleError, match="must not be zero"):
        parse_vehicle(
            minimal(thrusters=[{"model": "T200", "position": [0, 0, 0], "direction": [0, 0, 0]}])
        )


def test_cli_check(capsys):
    assert main(["check", str(EXAMPLES / "sphere.yaml")]) == 0
    assert "sphere: OK" in capsys.readouterr().out


def test_cli_check_reports_errors(tmp_path, capsys):
    bad = tmp_path / "bad.yaml"
    bad.write_text("name: x\n")
    assert main(["check", str(bad)]) == 1
    assert "'category'" in capsys.readouterr().err
