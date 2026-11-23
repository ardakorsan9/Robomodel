"""Mass spread over the parts of a CAD assembly by guessed material."""

import numpy as np
import pytest
import trimesh

from robomodel.mass_parts import DEFAULT_DENSITY, guess_material
from robomodel.model import build_model
from robomodel.vehicle import parse_vehicle


@pytest.mark.parametrize(
    ("name", "material"),
    [
        ("M3X0.5-16-SS316-SOCKET-HEAD-CAP-SCREW_27", "steel fastener"),
        ("T200-ASM-CCW-THRUSTER-R1_15", "motor / thruster"),
        ("LiPo3S_5000mAh", "battery"),
        ("ana_batarya", "battery"),
        ("spinner_cover", "plastic"),  # "pin" only as a whole word
        ("syntactic_foam_block", "foam / float"),
        ("lead_weight_200g", "lead ballast"),
        ("Part1", "unknown"),
    ],
)
def test_material_from_part_names(name, material):
    assert guess_material(name)[0] == material


def _assembly(path):
    """A frame plate, a battery off to one side and a foam float on the other."""
    scene = trimesh.Scene()
    plate = trimesh.creation.box([0.4, 0.4, 0.01])
    battery = trimesh.creation.box([0.15, 0.05, 0.03])
    battery.apply_translation([0.1, 0.0, 0.02])
    foam = trimesh.creation.box([0.2, 0.2, 0.05])
    foam.apply_translation([-0.1, 0.0, -0.03])
    for name, mesh in (("frame_plate", plate), ("lipo_battery", battery), ("foam_float", foam)):
        scene.add_geometry(mesh, geom_name=name, node_name=name)
    scene.export(path)
    return {"frame_plate": plate, "lipo_battery": battery, "foam_float": foam}


def _vehicle(path, **geometry):
    return parse_vehicle(
        {
            "name": "asm",
            "category": "underwater",
            "mass": 3.0,
            "geometry": {"file": str(path), "assembly": True, "voxel": 4, **geometry},
        }
    )


def test_assembly_mass_follows_the_materials(tmp_path):
    path = tmp_path / "asm.glb"
    meshes = _assembly(path)
    model = build_model(_vehicle(path))
    dist = model.mass.distribution
    assert dist is not None and len(dist.parts) == 3
    density = {n: guess_material(n)[1] for n in meshes}
    raw = {n: density[n] * m.volume for n, m in meshes.items()}
    scale = 3.0 / sum(raw.values())
    expected_cg = sum(raw[n] * scale * meshes[n].center_mass for n in meshes) / 3.0
    assert np.allclose(model.mass.r_g.value, expected_cg, atol=1e-9)
    by_name = {p.name.split()[0]: p for p in dist.parts}
    assert by_name["lipo_battery"].mass == pytest.approx(raw["lipo_battery"] * scale)
    assert dist.scale == pytest.approx(scale)
    # uniform density would put the CG at the volume centroid, on the foam's side
    uniform = build_model(_vehicle(path, hull_mass="solid"))
    assert uniform.mass.distribution is None
    assert uniform.mass.r_g.value[0] < model.mass.r_g.value[0]


def test_implausible_mass_is_flagged(tmp_path):
    path = tmp_path / "asm.glb"
    _assembly(path)
    data = _vehicle(path)
    from dataclasses import replace

    heavy = build_model(replace(data, mass=40.0))  # parts would weigh ~3 kg
    assert any("CAD parts would weigh" in w for w in heavy.warnings)
    assert not any("CAD parts would weigh" in w for w in build_model(data).warnings)


def test_single_body_cad_keeps_the_old_behaviour(tmp_path):
    path = tmp_path / "one.stl"
    trimesh.creation.box([0.4, 0.2, 0.2]).export(path)
    model = build_model(_vehicle(path))
    assert model.mass.distribution is None
    assert DEFAULT_DENSITY > 0
