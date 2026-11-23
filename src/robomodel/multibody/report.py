"""Summary and automatic checks for a multibody robot."""

from __future__ import annotations

import numpy as np

from robomodel.checks import FAIL, OK, WARN, Check
from robomodel.multibody.robot import Joint, Robot


def run_robot_checks(robot: Robot) -> list[Check]:
    out = []
    rng = np.random.default_rng(0)
    worst_eig, worst_asym, worst_id = np.inf, 0.0, 0.0
    for _ in range(20):
        q = rng.uniform(-np.pi, np.pi, robot.n)
        m = robot.mass_matrix(q)
        worst_asym = max(worst_asym, float(np.max(np.abs(m - m.T))))
        worst_eig = min(worst_eig, float(np.min(np.linalg.eigvalsh(m))))
        qd, qdd = rng.normal(size=(2, robot.n))
        tau = robot.inverse_dynamics(q, qd, qdd)
        worst_id = max(worst_id, float(np.max(np.abs(m @ qdd + robot.bias(q, qd) - tau))))
    out.append(
        Check(
            "mass matrix M(q) symmetric, positive definite (20 random poses)",
            OK if worst_asym < 1e-9 and worst_eig > 0 else FAIL,
            f"asymmetry {worst_asym:.1e}, smallest eigenvalue {worst_eig:.3g}",
        )
    )
    out.append(
        Check(
            "inverse and forward dynamics agree",
            OK if worst_id < 1e-8 else FAIL,
            f"largest difference {worst_id:.1e}",
        )
    )
    # Energy: without joint friction the total energy must stay constant.
    frictionless = Robot(
        list(robot.links.values()),
        [
            Joint(
                j.name, j.type, j.parent, j.child, j.xyz, j.rpy, j.axis, j.lower, j.upper, j.effort
            )
            for j in robot.joints
        ],
        root=robot.root,
        floating=robot.floating,
        gravity=robot.gravity,
    )
    q0 = rng.uniform(-0.5, 0.5, robot.n)
    try:
        _, q, qd = frictionless.simulate(2.0, q0=q0, dt_out=0.05)
        e = np.array([sum(frictionless.energy(a, b)) for a, b in zip(q, qd, strict=True)])
        drift = float(np.ptp(e) / max(abs(e).max(), 1e-12))
        out.append(
            Check(
                "energy conserved without friction (2 s simulation)",
                OK if drift < 1e-5 else FAIL,
                f"relative drift {drift:.1e}",
            )
        )
    except RuntimeError as e:
        out.append(Check("energy conserved without friction (2 s simulation)", FAIL, str(e)))
    light = [lk.name for lk in robot.links.values() if lk.mass < 1e-3]
    out.append(
        Check(
            "every link has a real mass",
            WARN if light else OK,
            f"very light links: {light}" if light else "all links over 1 g",
        )
    )
    return out


def robot_report(robot: Robot) -> str:
    from robomodel.checks import summary

    lines = [
        f"{getattr(robot, 'name', 'robot')}: {len(robot.links)} links, {robot.n} degrees of "
        f"freedom ({', '.join(robot.dof_names)})",
        "",
    ]
    total = sum(lk.mass for lk in robot.links.values())
    lines.append(f"total mass {total:.4g} kg" + (" (floating base)" if robot.floating else ""))
    lines.append("")
    lines.append(
        f"  {'link':<14}{'mass kg':>9}  {'centre of mass (m)':<28}{'Ixx, Iyy, Izz (kg m^2)'}"
    )

    def clean(v):
        v = np.asarray(v, dtype=float)
        return np.where(np.abs(v) < 1e-12 * max(np.abs(v).max(), 1.0), 0.0, v)

    for lk in robot.links.values():
        com = ", ".join(f"{x:.4g}" for x in clean(lk.com))
        ii = ", ".join(f"{x:.4g}" for x in np.diag(lk.inertia))
        src = "CAD" if lk.mesh is not None else "given"
        lines.append(f"  {lk.name:<14}{lk.mass:>9.4g}  [{com}]{'':<4}[{ii}]  {src}")
    lines += ["", "mass matrix M(q = 0):"]
    m = robot.mass_matrix(np.zeros(robot.n))
    lines += ["  " + "  ".join(f"{x:10.4g}" for x in row) for row in m]
    lines += ["", "Checks:", summary(run_robot_checks(robot))]
    lines += [
        "",
        "Not from CAD (needs data): joint friction, motor and gearbox, contacts "
        "(wheels, feet), and the environment (water, air).",
    ]
    return "\n".join(lines)
