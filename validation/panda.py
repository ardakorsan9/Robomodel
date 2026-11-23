"""Franka Emika Panda: joint torques from robomodel's CAD-based model against the identified
model of Gaz et al. (2019), which reproduces torques measured on the real robot.

    uv run python validation/panda.py           # report: docs/dogrulama_panda.md

Data, in data/franka (links to ~/veri/veri/kaynaklar/franka):
* description: github.com/frankaemika/franka_description, robots/fer (the Panda): joint
  origins (kinematics.yaml), the maker's link inertials (inertials.yaml), collision meshes;
* C. Gaz, M. Cognetti, A. Oliva, P. Robuffo Giordano, A. De Luca, "Dynamic Identification
  of the Franka Emika Panda Robot With Retrieval of Feasible Parameters Using Penalty-Based
  Optimization", IEEE RA-L 2019; retrieved feasible parameters p-hat from Tables VIII and IX
  of the supplementary material (github.com/marcocognetti/FrankaEmikaPandaDynModel, pdf/),
  typed in below. Their inertias are about the centre of mass in link axes: with that
  reading the identified combinations of Table III come out right (e.g. J4yy + J3zz + ...
  = 0.1243 vs 0.1245 listed; about the link origin it would be 0.1041). c1z is not
  identifiable (joint 1 is vertical) and does not change any torque; the maker's value is used.

Three models of the same arm, compared at random poses inside the joint limits:
  reference  Gaz et al. (identified parameters)
  maker      Franka's inertials.yaml (mass, centre of mass, inertia per link)
  robomodel  Franka's collision meshes + only the link masses from inertials.yaml (as if
             weighed); centre of mass and inertia from the mesh, uniform density
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from robomodel.multibody.description import parse_robot

ROOT = Path(__file__).resolve().parents[1]
DESC = ROOT / "data/franka/description/robots/fer"
MESH = ROOT / "data/franka/description/meshes/robots/fer/collision"
POSES = 200

# Gaz et al. 2019, supplementary Tables VIII (mass kg, centre of mass m) and IX (inertia about
# the centre of mass, kg m^2: xx, xy, xz, yy, yz, zz), links 1-7.
GAZ_MASS = [4.970684, 0.646926, 3.228604, 3.587895, 1.225946, 1.666555, 0.735522]
GAZ_COM = [
    [3.875e-03, 2.081e-03, None],
    [-3.141e-03, -2.872e-02, 3.495e-03],
    [2.7518e-02, 3.9252e-02, -6.6502e-02],
    [-5.317e-02, 1.04419e-01, 2.7454e-02],
    [-1.1953e-02, 4.1065e-02, -3.8437e-02],
    [6.0149e-02, -1.4117e-02, -1.0517e-02],
    [1.0517e-02, -4.252e-03, 6.1597e-02],
]
GAZ_INERTIA = [
    [7.0337e-01, -1.3900e-04, 6.7720e-03, 7.0661e-01, 1.9169e-02, 9.1170e-03],
    [7.9620e-03, -3.9250e-03, 1.0254e-02, 2.8110e-02, 7.0400e-04, 2.5995e-02],
    [3.7242e-02, -4.7610e-03, -1.1396e-02, 3.6155e-02, -1.2805e-02, 1.0830e-02],
    [2.5853e-02, 7.7960e-03, -1.3320e-03, 1.9552e-02, 8.6410e-03, 2.8323e-02],
    [3.5549e-02, -2.1170e-03, -4.0370e-03, 2.9474e-02, 2.2900e-04, 8.6270e-03],
    [1.9640e-03, 1.0900e-04, -1.1580e-03, 4.3540e-03, 3.4100e-04, 5.4330e-03],
    [1.2516e-02, -4.2800e-04, -1.1960e-03, 1.0027e-02, -7.4100e-04, 4.8150e-03],
]


def _yaml(name):
    return yaml.safe_load((DESC / name).read_text())


def panda(source: str):
    kin, inert, limits = (
        _yaml("kinematics.yaml"),
        _yaml("inertials.yaml"),
        _yaml("joint_limits.yaml"),
    )
    links = []
    for i in range(8):
        data = inert[f"link{i}"]
        link = {"name": f"link{i}", "mass": float(data["mass"])}
        if source == "gaz" and i > 0:
            xx, xy, xz, yy, yz, zz = GAZ_INERTIA[i - 1]
            com = list(GAZ_COM[i - 1])
            if com[2] is None:  # not identifiable: any value gives the same torques
                com[2] = float(str(data["origin"]["xyz"]).split()[2])
            link.update(
                mass=GAZ_MASS[i - 1],
                com=com,
                inertia=[[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]],
            )
        elif source in ("maker", "gaz"):
            link["com"] = [float(x) for x in str(data["origin"]["xyz"]).split()]
            ii = data["inertia"]
            link["inertia"] = [
                [ii["xx"], ii["xy"], ii["xz"]],
                [ii["xy"], ii["yy"], ii["yz"]],
                [ii["xz"], ii["yz"], ii["zz"]],
            ]
        else:
            link["geometry"] = {"file": str(MESH / f"link{i}.stl"), "unit": "m"}
        links.append(link)
    joints = []
    for i in range(1, 8):
        k = kin[f"joint{i}"]["kinematic"]
        lim = limits[f"joint{i}"]["limit"]
        joints.append(
            {
                "name": f"joint{i}",
                "type": "revolute",
                "parent": f"link{i - 1}",
                "child": f"link{i}",
                "xyz": [k["x"], k["y"], k["z"]],
                "rpy": [k["roll"], k["pitch"], k["yaw"]],
                "axis": [0, 0, 1],
                "lower": lim["lower"],
                "upper": lim["upper"],
                "effort": lim["effort"],
            }
        )
    return parse_robot(
        {"name": f"panda_{source}", "root": "link0", "links": links, "joints": joints}
    )


def poses() -> np.ndarray:
    limits = _yaml("joint_limits.yaml")
    lo = [limits[f"joint{i}"]["limit"]["lower"] for i in range(1, 8)]
    hi = [limits[f"joint{i}"]["limit"]["upper"] for i in range(1, 8)]
    return np.random.default_rng(7).uniform(lo, hi, size=(POSES, 7))


def main() -> None:
    q = poses()
    zero = np.zeros(7)
    reference = panda("gaz")
    g_ref = np.array([reference.bias(qq, zero) for qq in q])
    m_ref = np.array([reference.mass_matrix(qq) for qq in q])
    models = {"maker": panda("maker"), "robomodel": panda("cad")}
    print(f"Panda, {POSES} random poses inside the joint limits; reference: Gaz et al. 2019\n")
    print("Gravity torque g(q): RMS error per joint, N m (and % of the reference RMS)")
    rms_ref = np.sqrt(np.mean(g_ref**2, axis=0))
    print(f"  {'':<11}" + "".join(f"{'j' + str(i + 1):>14}" for i in range(7)))
    print(f"  {'reference':<11}" + "".join(f"{x:>14.3f}" for x in rms_ref) + "  (RMS)")
    summary = {}
    for name, robot in models.items():
        g = np.array([robot.bias(qq, zero) for qq in q])
        err = np.sqrt(np.mean((g - g_ref) ** 2, axis=0))
        rel = err / rms_ref
        summary[name] = (err, rel)
        print(
            f"  {name:<11}"
            + "".join(f"{e:>8.3f} ({100 * r:3.0f}%)" for e, r in zip(err, rel, strict=True))
        )
    print("\nMass matrix M(q): median relative error ||M - M_ref|| / ||M_ref||")
    for name, robot in models.items():
        errs = [
            np.linalg.norm(robot.mass_matrix(qq) - mr) / np.linalg.norm(mr)
            for qq, mr in zip(q, m_ref, strict=True)
        ]
        diag = np.array(
            [
                np.diag(robot.mass_matrix(qq)) / np.diag(mr) - 1
                for qq, mr in zip(q, m_ref, strict=True)
            ]
        )
        print(
            f"  {name:<11} {100 * np.median(errs):5.1f} %   diagonal per joint, median: "
            + ", ".join(f"{100 * x:+.0f}%" for x in np.median(diag, axis=0))
        )
    total = {n: sum(lk.mass for lk in r.links.values()) for n, r in models.items()}
    print(f"\nmasses: maker/robomodel {total['maker']:.3f} kg (links 0-7, same for both)")


if __name__ == "__main__":
    main()
