"""Fossen 6-DOF equations of motion and a simple simulator.

    eta_dot = J(eta) nu
    M nu_dot + C(nu) nu + D(nu) nu + g(eta) = tau

eta = [x, y, z, phi, theta, psi] in NED (z down), nu = [u, v, w, p, q, r] in the body
frame, M = M_RB + M_A, C = C_RB + C_A from M (Kirchhoff form, valid for any symmetric M).
The ocean current is zero.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp

from robomodel.hydrostatics import restoring_forces
from robomodel.rigid_body import skew


@dataclass(frozen=True)
class Dynamics:
    """Plain numbers of a model, everything the equations need."""

    m_rb: np.ndarray  # 6x6
    m_a: np.ndarray  # 6x6, positive added mass
    d_lin: np.ndarray  # (6,), positive
    d_quad: np.ndarray  # (6,), positive
    weight: float  # N
    buoyancy: float  # N
    r_g: np.ndarray  # (3,)
    r_b: np.ndarray  # (3,)
    b_thr: np.ndarray  # 6 x n thruster allocation

    @property
    def mass_matrix(self) -> np.ndarray:
        return self.m_rb + self.m_a


def coriolis_matrix(m: np.ndarray, nu: np.ndarray) -> np.ndarray:
    """Coriolis-centripetal matrix from a symmetric 6x6 mass matrix (Fossen, Theorem 3.2).

    C(nu) = [[0, -S(M11 nu1 + M12 nu2)], [-S(M11 nu1 + M12 nu2), -S(M21 nu1 + M22 nu2)]]
    It is skew-symmetric, so it does no work: nu^T C(nu) nu = 0.
    """
    nu1, nu2 = nu[:3], nu[3:]
    a = m[:3, :3] @ nu1 + m[:3, 3:] @ nu2
    b = m[3:, :3] @ nu1 + m[3:, 3:] @ nu2
    return np.block([[np.zeros((3, 3)), -skew(a)], [-skew(a), -skew(b)]])


def rotation_zyx(phi: float, theta: float, psi: float) -> np.ndarray:
    """Body-to-NED rotation matrix for z-y-x Euler angles."""
    cph, sph = np.cos(phi), np.sin(phi)
    cth, sth = np.cos(theta), np.sin(theta)
    cps, sps = np.cos(psi), np.sin(psi)
    return np.array(
        [
            [cps * cth, -sps * cph + cps * sth * sph, sps * sph + cps * cph * sth],
            [sps * cth, cps * cph + sph * sth * sps, -cps * sph + sth * sps * cph],
            [-sth, cth * sph, cth * cph],
        ]
    )


def euler_rate_matrix(phi: float, theta: float) -> np.ndarray:
    """T(Theta): body angular rates -> Euler angle rates. Singular at theta = +-90 deg."""
    cph, sph = np.cos(phi), np.sin(phi)
    cth, tth = np.cos(theta), np.tan(theta)
    return np.array([[1.0, sph * tth, cph * tth], [0.0, cph, -sph], [0.0, sph / cth, cph / cth]])


def damping_force(dyn: Dynamics, nu: np.ndarray) -> np.ndarray:
    """D(nu) nu."""
    return dyn.d_lin * nu + dyn.d_quad * np.abs(nu) * nu


def state_derivative(
    dyn: Dynamics, state: np.ndarray, tau: np.ndarray, m_inv: np.ndarray | None = None
) -> np.ndarray:
    eta, nu = state[:6], state[6:]
    phi, theta, psi = eta[3:]
    m = dyn.mass_matrix
    if m_inv is None:
        m_inv = np.linalg.inv(m)
    g = restoring_forces(phi, theta, dyn.weight, dyn.buoyancy, dyn.r_g, dyn.r_b)
    nu_dot = m_inv @ (tau - coriolis_matrix(m, nu) @ nu - damping_force(dyn, nu) - g)
    eta_dot = np.concatenate(
        [rotation_zyx(phi, theta, psi) @ nu[:3], euler_rate_matrix(phi, theta) @ nu[3:]]
    )
    return np.concatenate([eta_dot, nu_dot])


@dataclass(frozen=True)
class Trajectory:
    t: np.ndarray  # (n,)
    eta: np.ndarray  # (n, 6)
    nu: np.ndarray  # (n, 6)


def simulate(
    dyn: Dynamics,
    t_end: float,
    tau: Callable[[float, np.ndarray], np.ndarray] | np.ndarray | None = None,
    state0: np.ndarray | None = None,
    dt_out: float = 0.05,
    rtol: float = 1e-8,
    atol: float = 1e-10,
) -> Trajectory:
    """Integrate the equations of motion.

    tau: constant generalised force (6,), or a function tau(t, state), or None for zero.
    """
    state0 = np.zeros(12) if state0 is None else np.asarray(state0, dtype=float)
    if tau is None:
        tau = np.zeros(6)
    if callable(tau):
        tau_fn = tau
    else:
        constant = np.asarray(tau, dtype=float)

        def tau_fn(_t, _x):
            return constant

    m_inv = np.linalg.inv(dyn.mass_matrix)

    def rhs(t, x):
        return state_derivative(dyn, x, tau_fn(t, x), m_inv)

    t_eval = np.arange(0.0, t_end + dt_out / 2, dt_out)
    sol = solve_ivp(rhs, (0.0, t_end), state0, method="RK45", t_eval=t_eval, rtol=rtol, atol=atol)
    if not sol.success:
        raise RuntimeError(f"simulation failed: {sol.message}")
    return Trajectory(sol.t, sol.y[:6].T, sol.y[6:].T)


def linearize(dyn: Dynamics, state0: np.ndarray, tau0: np.ndarray, eps: float = 1e-6):
    """A = d f / d x and B = d f / d tau at (state0, tau0), by central differences."""
    state0, tau0 = np.asarray(state0, float), np.asarray(tau0, float)
    m_inv = np.linalg.inv(dyn.mass_matrix)

    def f(x, u):
        return state_derivative(dyn, x, u, m_inv)

    a = np.zeros((12, 12))
    for i in range(12):
        d = np.zeros(12)
        d[i] = eps
        a[:, i] = (f(state0 + d, tau0) - f(state0 - d, tau0)) / (2 * eps)
    b = np.zeros((12, 6))
    for i in range(6):
        d = np.zeros(6)
        d[i] = eps
        b[:, i] = (f(state0, tau0 + d) - f(state0, tau0 - d)) / (2 * eps)
    return a, b
