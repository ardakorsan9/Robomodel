"""TurtleBot3 Burger: robomodel's ground model against the maker's published performance.

Run: uv run python validation/turtlebot3.py   (report: docs/dogrulama_turtlebot3.md)

Data (ROBOTIS, fetched 2026-11-24):
* turtlebot3_description (github.com/ROBOTIS-GIT/turtlebot3): burger_base.stl, wheel and
  caster positions, the maker's CAD inertia of the base (in data/turtlebot3/description, a
  link to ~/veri/veri/kaynaklar/turtlebot3);
* e-manual, TurtleBot3 features: max 0.22 m/s and 2.84 rad/s, 1 kg with SBC, battery and
  sensors; DYNAMIXEL XL430-W250: stall torque 1.4 N m and no-load speed 57 rpm at 11.1 V
  (61 rpm at 12 V), 57.2 g.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from robomodel.ground import build_ground_model, performance, simulate
from robomodel.vehicle import parse_vehicle

ROOT = Path(__file__).resolve().parents[1]
DESC = ROOT / "data/turtlebot3/description"
MESH_X = 0.032  # the base mesh sits 0.032 m behind base_link (URDF visual origin)


def burger(volts: float = 11.1, surface: str = "indoor"):
    rpm = {11.1: 57.0, 12.0: 61.0}[volts]
    stall = {11.1: 1.4, 12.0: 1.5}[volts]

    # URDF (base_link: x forward, y left, z up) -> body FRD, in the mesh's own frame (+0.032 x)
    def frd(x, y, z):
        return [x + MESH_X, -y, -z]

    motor = {"driven": True, "stall_torque": stall, "no_load_rpm": rpm}
    return parse_vehicle(
        {
            "name": "turtlebot3_burger",
            "category": "ground",
            "mass": 1.0,
            "surface": surface,
            "geometry": {
                "file": str(DESC / "meshes/bases/burger_base.stl"),
                "unit": "mm",
                "axes": ["x", "-y", "-z"],
            },
            "wheels": [
                {
                    "name": "left",
                    "position": frd(0, 0.08, 0.023),
                    "radius": 0.033,
                    "width": 0.018,
                    "mass": 0.0285 + 0.0572,
                    **motor,
                },
                {
                    "name": "right",
                    "position": frd(0, -0.08, 0.023),
                    "radius": 0.033,
                    "width": 0.018,
                    "mass": 0.0285 + 0.0572,
                    **motor,
                },
                {
                    "name": "caster",
                    "position": frd(-0.081, 0, -0.004),
                    "radius": 0.006,
                    "caster": True,
                },
            ],
        }
    )


def main() -> None:
    rows = []
    for volts in (11.1, 12.0):
        gm = build_ground_model(burger(volts))
        perf = performance(gm)
        spin = simulate(gm, 3.0, np.array([-1.0, 1.0]), dt_out=0.5).state[-1, 5]
        rows.append((volts, perf["top_speed"], abs(spin)))
    print("TurtleBot3 Burger, motors at full command (no firmware speed limit)")
    print(f"  {'battery':<10}{'top speed m/s':>16}{'spin rad/s':>14}")
    for volts, v, r in rows:
        print(f"  {volts:<10}{v:>16.4f}{r:>14.3f}")
    print(f"  {'ROBOTIS':<10}{0.22:>16.4f}{2.84:>14.3f}")
    for volts, v, r in rows:
        print(
            f"  {volts} V: speed {100 * (v / 0.22 - 1):+.1f} %, spin {100 * (r / 2.84 - 1):+.1f} %"
        )

    gm = build_ground_model(burger())
    print("\nBase link inertia (kg m^2): robomodel (uniform over the mesh) vs maker's CAD")
    import trimesh

    from robomodel.geometry import build_mesh

    hull = build_mesh(gm.vehicle.geometry)
    base_mass = 0.82573504  # maker's URDF base_link mass
    ours = np.diag(np.asarray(hull.moment_inertia)) * base_mass / hull.volume
    maker = np.array([2.2124416e-03, 2.1193702e-03, 2.0064271e-03])
    for axis, a, b in zip("xyz", ours, maker, strict=True):
        print(f"  I{axis}{axis}  {a:.4g}  maker {b:.4g}  {100 * (a / b - 1):+.0f} %")
    print(
        f"  mesh closed: {isinstance(hull, trimesh.Trimesh) and hull.is_watertight}, "
        f"CG height {gm.cg_height:.3f} m, tip angles {', '.join(f'{k} {v:.0f}' for k, v in __import__('robomodel.ground', fromlist=['tip_limits']).tip_limits(gm).items())} deg"
    )


if __name__ == "__main__":
    main()
