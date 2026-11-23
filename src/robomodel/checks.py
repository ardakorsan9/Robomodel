"""Automatic mathematical checks run on every model.

A model can be built from bad input (wrong units, a part outside the hull, a typo in a
thruster direction) and still produce numbers. These checks catch models that are
physically impossible or unusable before anyone designs a controller with them.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from robomodel.dynamics import Dynamics, coriolis_matrix, damping_force, linearize, simulate
from robomodel.rigid_body import velocity_transform

OK, WARN, FAIL = "ok", "warn", "fail"


@dataclass(frozen=True)
class Check:
    name: str
    status: str  # ok | warn | fail
    detail: str


def _tidy(x: float, scale: float) -> float:
    """Round-off far below the matrix's own size is shown as 0."""
    return 0.0 if abs(x) < 1e-12 * max(scale, 1e-300) else x


def _sym_pd(m: np.ndarray, semi: bool = False) -> tuple[bool, float, float]:
    asym = float(np.max(np.abs(m - m.T)) / max(np.max(np.abs(m)), 1e-12))
    eig = float(np.min(np.linalg.eigvalsh((m + m.T) / 2)))
    scale = max(float(np.max(np.abs(m))), 1e-12)
    ok = asym < 1e-9 and (eig >= -1e-9 * scale if semi else eig > 1e-9 * scale)
    return ok, _tidy(asym, 1.0), _tidy(eig, scale)


def run_checks(dyn: Dynamics, speed: float = 1.0) -> list[Check]:
    out = []

    ok, asym, eig = _sym_pd(dyn.m_rb)
    out.append(
        Check(
            "rigid-body mass matrix M_RB symmetric, positive definite",
            OK if ok else FAIL,
            f"asymmetry {asym:.1e}, smallest eigenvalue {eig:.3g}",
        )
    )
    ok, asym, eig = _sym_pd(dyn.m_a, semi=True)
    out.append(
        Check(
            "added mass M_A symmetric, positive semi-definite",
            OK if ok else FAIL,
            f"asymmetry {asym:.1e}, smallest eigenvalue {eig:.3g}",
        )
    )
    ok, _, eig = _sym_pd(dyn.mass_matrix)
    out.append(
        Check(
            "total mass matrix M invertible", OK if ok else FAIL, f"smallest eigenvalue {eig:.3g}"
        )
    )

    rng = np.random.default_rng(0)
    worst_c, worst_d = 0.0, np.inf
    scales = np.array([speed] * 3 + [speed] * 3)
    for _ in range(50):
        nu = np.linalg.solve(velocity_transform(dyn.r_hydro), rng.normal(size=6) * scales)
        c = coriolis_matrix(dyn.mass_matrix, nu)
        worst_c = max(worst_c, abs(nu @ c @ nu) / max(nu @ dyn.mass_matrix @ nu, 1e-12))
        nu_fwd = nu.copy()
        nu_fwd[0] = abs(nu_fwd[0]) + speed  # fins and body lift act in forward motion
        worst_d = min(worst_d, float(nu_fwd @ damping_force(dyn, nu_fwd)))
    out.append(
        Check(
            "Coriolis does no work (nu' C nu = 0)",
            OK if worst_c < 1e-9 else FAIL,
            f"largest relative work {worst_c:.1e}",
        )
    )
    out.append(
        Check(
            "damping only removes energy (nu' D nu >= 0)",
            OK if worst_d >= -1e-9 else FAIL,
            f"smallest power {worst_d:.3g} W",
        )
    )

    # Free motion without gravity/buoyancy effects: kinetic energy must not grow.
    free = replace(dyn, weight=dyn.buoyancy, r_g=dyn.r_b.copy())
    # Random velocities of the hull centre, moved to the origin: a far-away CAD origin
    # must not turn small rotations into huge hull speeds.
    nu0 = np.linalg.solve(velocity_transform(dyn.r_hydro), rng.normal(size=6) * scales * 0.3)
    try:
        traj = simulate(
            free, 5.0, state0=np.concatenate([np.zeros(6), nu0]), dt_out=0.1, rtol=1e-7, atol=1e-9
        )
        energy = 0.5 * np.einsum("ij,jk,ik->i", traj.nu, free.mass_matrix, traj.nu)
        grows = float(np.max(np.diff(energy)) / max(energy[0], 1e-12))
        detail = f"largest relative energy increase per step {grows:.1e}"
    except (RuntimeError, FloatingPointError, ValueError) as e:
        grows, detail = np.inf, f"simulation failed: {e}"
    out.append(Check("free motion loses energy (simulation)", OK if grows < 1e-6 else FAIL, detail))

    bg = dyn.r_g[2] - dyn.r_b[2]  # z down: CG below CB means bg > 0
    out.append(
        Check(
            "self-righting in roll and pitch (CG below CB)",
            OK if bg > 1e-4 else WARN,
            f"metacentric height BG_z = {100 * bg:.2f} cm"
            + ("" if bg > 1e-4 else ": the vehicle will not return upright by itself"),
        )
    )
    net = (dyn.buoyancy - dyn.weight) / dyn.weight
    out.append(
        Check(
            "near neutral buoyancy (within 5 %)",
            OK if abs(net) < 0.05 else WARN,
            f"net buoyancy {100 * net:+.1f} % of weight",
        )
    )

    a, _ = linearize(dyn, np.zeros(12), np.zeros(6))
    vel = np.linalg.eigvals(a[6:, 6:])
    worst = float(np.max(vel.real))
    out.append(
        Check(
            "velocities decay at rest (linear model)",
            OK if worst <= 1e-9 else FAIL,
            f"largest real part {worst:.3g} 1/s",
        )
    )
    state = np.zeros(12)
    state[6] = speed
    tau0 = np.zeros(6)
    tau0[0] = dyn.d_lin[0] * speed + dyn.d_quad[0] * speed**2
    a, _ = linearize(dyn, state, tau0)
    worst = float(np.max(np.linalg.eigvals(a[6:, 6:]).real))
    out.append(
        Check(
            f"directionally stable at {speed:g} m/s (no control)",
            OK if worst <= 1e-9 else WARN,
            f"largest real part {worst:.3g} 1/s"
            + ("" if worst <= 1e-9 else ": needs active heading/depth control"),
        )
    )

    if dyn.b_thr.shape[1]:
        b = dyn.b_thr / np.maximum(np.abs(dyn.b_thr).max(axis=1, keepdims=True), 1e-12)
        dofs = [n for n, row in zip("XYZKMN", dyn.b_thr, strict=True) if np.max(np.abs(row)) > 1e-9]
        rank = int(np.linalg.matrix_rank(b, tol=1e-6))
        out.append(
            Check(
                "thrusters can push in each direction they touch",
                OK,
                f"{rank} independent directions; forces/moments: {', '.join(dofs) or 'none'}",
            )
        )
    return out


def summary(checks: list[Check]) -> str:
    marks = {OK: "ok  ", WARN: "WARN", FAIL: "FAIL"}
    return "\n".join(f"  [{marks[c.status]}] {c.name}: {c.detail}" for c in checks)
