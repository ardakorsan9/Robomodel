"""Fossen 6-DOF equations of motion and a simple simulator.

    eta_dot = J(eta) nu
    M nu_dot + C(nu) nu + D(nu) nu + g(eta) = tau

eta = [x, y, z, phi, theta, psi] in NED (z down), nu = [u, v, w, p, q, r] in the body
frame, M = M_RB + M_A, C = C_RB + C_A from M (Kirchhoff form, valid for any symmetric M).

Ocean current: a constant, irrotational current v_c (NED, m/s). Hydrodynamic terms (added
mass, damping, lift) see the velocity relative to the water nu_r = nu - nu_c (Fossen 2021,
ch. 10):
    M_RB nu_dot + C_RB(nu) nu + M_A nu_r_dot + C_A(nu_r) nu_r + D(nu_r) nu_r + g = tau
with nu_c = [R^T v_c; 0] and nu_c_dot = [-S(w) R^T v_c; 0].
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
from scipy.integrate import solve_ivp

from robomodel.hydrostatics import restoring_forces
from robomodel.rigid_body import skew, velocity_transform


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
    d_fin: np.ndarray = field(default_factory=lambda: np.zeros((6, 6)))  # x |u|
    b_fin: np.ndarray = field(default_factory=lambda: np.zeros((6, 0)))  # x u|u| delta
    fin_max: np.ndarray = field(default_factory=lambda: np.zeros(0))  # rad per control fin
    current_ned: np.ndarray = field(default_factory=lambda: np.zeros(3))  # m/s, constant
    # d_lin / d_quad act at this point (the hull's centre), not at the body origin
    r_hydro: np.ndarray = field(default_factory=lambda: np.zeros(3))
    # Surface vessels: g(eta) = G (eta - eta_rest), roll from the GZ curve (roll_table rows:
    # heel angle rad, roll restoring N m). None: fully submerged vehicle (W, B, r_g, r_b).
    g_matrix: np.ndarray | None = None
    eta_rest: np.ndarray = field(default_factory=lambda: np.zeros(6))
    roll_table: np.ndarray = field(default_factory=lambda: np.zeros((2, 0)))
    kind: str = "underwater"  # underwater | surface | aerial (which checks apply)
    thrust_max: np.ndarray = field(default_factory=lambda: np.zeros(0))  # N per thruster

    @property
    def mass_matrix(self) -> np.ndarray:
        return self.m_rb + self.m_a

    @property
    def rest_state(self) -> np.ndarray:
        """State at rest: floating position for a surface vessel, zero otherwise."""
        return np.concatenate([self.eta_rest, np.zeros(6)])


def restoring(dyn: Dynamics, eta: np.ndarray) -> np.ndarray:
    """g(eta): hydrostatic restoring forces (the negative of gravity + buoyancy)."""
    phi, theta = eta[3], eta[4]
    if dyn.g_matrix is None:
        return restoring_forces(phi, theta, dyn.weight, dyn.buoyancy, dyn.r_g, dyn.r_b)
    d_eta = eta - dyn.eta_rest
    g = dyn.g_matrix @ d_eta
    if dyn.roll_table.shape[1]:  # large heel: the GZ curve instead of the slope GM
        angles, moments = dyn.roll_table
        g[3] += np.interp(phi, angles, moments) - dyn.g_matrix[3, 3] * d_eta[3]
    return g


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
    """D(nu) nu, including the speed-dependent fin lift.

    The hull drag is defined at the hull centre r_hydro: its velocity there is H nu, and the
    force moves back to the origin with H^T. For r_hydro = 0 this is the plain diagonal form.
    """
    lift = abs(nu[0]) * (dyn.d_fin @ nu)
    if not np.any(dyn.r_hydro):
        return dyn.d_lin * nu + dyn.d_quad * np.abs(nu) * nu + lift
    h = velocity_transform(dyn.r_hydro)
    nu_c = h @ nu
    return h.T @ (dyn.d_lin * nu_c + dyn.d_quad * np.abs(nu_c) * nu_c) + lift


def fin_control_force(dyn: Dynamics, nu: np.ndarray, deflection: np.ndarray) -> np.ndarray:
    """tau from control-fin deflections (rad); fins only work while moving."""
    if dyn.b_fin.shape[1] == 0:
        return np.zeros(6)
    delta = np.asarray(deflection, dtype=float)
    if dyn.fin_max.size:
        delta = np.clip(delta, -dyn.fin_max, dyn.fin_max)  # stall: lift stops growing
    return nu[0] * abs(nu[0]) * (dyn.b_fin @ delta)


def relative_velocity(dyn: Dynamics, state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(nu_r, nu_c_dot): velocity relative to the water and the current's rate in body axes."""
    nu = state[6:]
    if not np.any(dyn.current_ned):
        return nu, np.zeros(6)
    phi, theta, psi = state[3:6]
    v_c = rotation_zyx(phi, theta, psi).T @ dyn.current_ned
    nu_c = np.concatenate([v_c, np.zeros(3)])
    nu_c_dot = np.concatenate([-np.cross(nu[3:], v_c), np.zeros(3)])
    return nu - nu_c, nu_c_dot


def state_derivative(
    dyn: Dynamics, state: np.ndarray, tau: np.ndarray, m_inv: np.ndarray | None = None
) -> np.ndarray:
    eta, nu = state[:6], state[6:]
    phi, theta, psi = eta[3:]
    m = dyn.mass_matrix
    if m_inv is None:
        m_inv = np.linalg.inv(m)
    g = restoring(dyn, eta)
    if np.any(dyn.current_ned):
        nu_r, nu_c_dot = relative_velocity(dyn, state)
        rhs = (
            tau
            - coriolis_matrix(dyn.m_rb, nu) @ nu
            - coriolis_matrix(dyn.m_a, nu_r) @ nu_r
            - damping_force(dyn, nu_r)
            - g
            + dyn.m_a @ nu_c_dot
        )
    else:  # same equations; C is linear in M, so one call does both parts
        rhs = tau - coriolis_matrix(m, nu) @ nu - damping_force(dyn, nu) - g
    nu_dot = m_inv @ rhs
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
    fin_deflection: np.ndarray | None = None,
) -> Trajectory:
    """Integrate the equations of motion.

    tau: constant generalised force (6,), or a function tau(t, state), or None for zero.
    fin_deflection: constant deflection (rad) of each control fin.
    """
    state0 = dyn.rest_state if state0 is None else np.asarray(state0, dtype=float)
    if tau is None:
        tau = np.zeros(6)
    if callable(tau):
        tau_fn = tau
    else:
        constant = np.asarray(tau, dtype=float)

        def tau_fn(_t, _x):
            return constant

    m_inv = np.linalg.inv(dyn.mass_matrix)

    deflection = None if fin_deflection is None else np.asarray(fin_deflection, dtype=float)

    def rhs(t, x):
        tau_t = tau_fn(t, x)
        if deflection is not None:
            tau_t = tau_t + fin_control_force(dyn, relative_velocity(dyn, x)[0], deflection)
        return state_derivative(dyn, x, tau_t, m_inv)

    t_eval = np.arange(0.0, t_end + dt_out / 2, dt_out)
    t_eval = t_eval[t_eval <= t_end]  # round-off: an odd dt_out must not step past t_end
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
