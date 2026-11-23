"""Calibrate a model to test data (pool runs, sea trials, or a simulated "true" vehicle).

The CAD model gives every uncertain term a value and an uncertainty. Calibration fits scale
factors s (1 = CAD value) for chosen groups of terms so that the simulated motion matches
the logged motion, given the logged inputs:

    minimise  sum_k ((y_sim - y_meas) / noise)^2  +  sum_i (ln s_i / prior_i)^2

(a maximum a posteriori fit with Gaussian sensor noise; noise is the 1-sigma per channel)

The second sum keeps the fit near the CAD value when the data says little about a term
(a Bayesian prior on ln s). Uncertainty of the result comes from the fit's Jacobian.

Log format (CSV with a header row): column `t` (s), inputs and measured states.
  inputs:  thr_<i>   thruster i command in [-1, 1]
           fin_<name> control fin deflection (rad)
           tau_x, tau_y, tau_z, tau_k, tau_m, tau_n   extra generalised force (N, N m)
  states:  any of x y z phi theta psi (NED, rad) and u v w p q r (body, m/s, rad/s)
Inputs are held constant between samples (zero-order hold).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares

from robomodel.dynamics import fin_control_force, relative_velocity, state_derivative
from robomodel.model import VehicleModel, apply_calibration
from robomodel.thrusters import thrust
from robomodel.vehicle import Calibration

STATES = ("x", "y", "z", "phi", "theta", "psi", "u", "v", "w", "p", "q", "r")
TAU_COLUMNS = ("tau_x", "tau_y", "tau_z", "tau_k", "tau_m", "tau_n")
ANGLES = {"phi", "theta", "psi"}
# Typical sensor noise (1 sigma), used to weight the channels.
DEFAULT_NOISE = {
    "x": 0.05, "y": 0.05, "z": 0.02, "phi": 0.01, "theta": 0.01, "psi": 0.01,
    "u": 0.02, "v": 0.02, "w": 0.02, "p": 0.01, "q": 0.01, "r": 0.01,
}  # fmt: skip
# Prior width on ln(scale): CAD damping was ~10x too low for REMUS, so it is wide.
PRIOR_LOG_SIGMA = {
    "damping": 1.5, "added_mass": 0.5, "fin_lift": 0.5, "body_lift": 1.0, "thrust": 0.3,
}  # fmt: skip
GROUPS = ("damping", "added_mass", "fin_lift", "body_lift", "thrust")
DEFAULT_GROUPS = ("damping", "fin_lift", "body_lift")
DOF = ("surge", "sway", "heave", "roll", "pitch", "yaw")
BOUNDS = (np.log(0.02), np.log(50.0))


@dataclass(frozen=True)
class Log:
    t: np.ndarray
    inputs: dict[str, np.ndarray]
    states: dict[str, np.ndarray]
    name: str = "log"


def load_log(path: str | Path) -> Log:
    path = Path(path)
    data = np.genfromtxt(path, delimiter=",", names=True, dtype=float, encoding="utf-8")
    names = data.dtype.names or ()
    if "t" not in names:
        raise ValueError(f"{path}: needs a 't' column (time in seconds)")
    t = np.asarray(data["t"], dtype=float)
    if np.any(np.diff(t) <= 0):
        raise ValueError(f"{path}: time must increase")
    inputs = {
        n: np.asarray(data[n], float)
        for n in names
        if n.startswith(("thr_", "fin_")) or n in TAU_COLUMNS
    }
    states = {}
    for n in names:
        if n in STATES:
            column = np.atleast_1d(np.asarray(data[n], float))
            if np.all(np.isnan(column)):
                continue  # an empty template column: not measured
            if np.any(np.isnan(column)):
                raise ValueError(f"{path}: column {n!r} has empty cells; fill or delete it")
            states[n] = column
    if not states:
        raise ValueError(f"{path}: no measured states (columns {', '.join(STATES)})")
    unknown = set(names) - {"t"} - set(inputs) - set(STATES)
    if unknown:
        raise ValueError(f"{path}: unknown columns {sorted(unknown)}")
    return Log(t, inputs, states, path.stem)


def save_log(log: Log, path: str | Path) -> None:
    columns = {"t": log.t, **log.inputs, **log.states}
    header = ",".join(columns)
    np.savetxt(
        path,
        np.column_stack(list(columns.values())),
        delimiter=",",
        header=header,
        comments="",
        fmt="%.9g",
    )


def _input_function(model: VehicleModel, log: Log):
    """tau(t, state) from the logged inputs, zero-order hold."""
    vehicle = model.vehicle
    n_thr = len(vehicle.thrusters)
    thr = np.zeros((len(log.t), n_thr))
    for i in range(n_thr):
        if f"thr_{i}" in log.inputs:
            thr[:, i] = log.inputs[f"thr_{i}"]
    names = model.fins.control_names
    fins = np.zeros((len(log.t), len(names)))
    for j, name in enumerate(names):
        if f"fin_{name}" in log.inputs:
            fins[:, j] = log.inputs[f"fin_{name}"]
    extra = np.zeros((len(log.t), 6))
    for j, col in enumerate(TAU_COLUMNS):
        if col in log.inputs:
            extra[:, j] = log.inputs[col]
    unknown_fins = {k for k in log.inputs if k.startswith("fin_")} - {f"fin_{n}" for n in names}
    unknown_thr = {k for k in log.inputs if k.startswith("thr_")} - {
        f"thr_{i}" for i in range(n_thr)
    }
    if unknown_fins or unknown_thr:
        raise ValueError(
            f"{log.name}: inputs not in the vehicle: {sorted(unknown_fins | unknown_thr)}"
        )
    base = extra.copy()
    if n_thr:
        forces = np.array([thrust(vehicle.thrusters, row) for row in thr])
        base += forces @ model.b_thr.T

    def tau(t, state, dyn):
        k = min(max(int(np.searchsorted(log.t, t, side="right")) - 1, 0), len(log.t) - 1)
        out = base[k]
        if fins.shape[1]:
            out = out + fin_control_force(dyn, relative_velocity(dyn, state)[0], fins[k])
        return out

    return tau


def initial_state(log: Log, rest: np.ndarray | None = None) -> np.ndarray:
    """Logged first values; states not in the log start at rest (floating position)."""
    state = np.zeros(12) if rest is None else np.array(rest, dtype=float)
    for i, name in enumerate(STATES):
        if name in log.states:
            state[i] = log.states[name][0]
    return state


def simulate_log(model: VehicleModel, log: Log, rtol: float = 1e-6) -> np.ndarray:
    """Simulate the model with the logged inputs; returns states (n, 12) at log.t."""
    dyn = model.dynamics()
    m_inv = np.linalg.inv(dyn.mass_matrix)
    tau = _input_function(model, log)
    dt = float(np.min(np.diff(log.t)))

    def rhs(t, x):
        return state_derivative(dyn, x, tau(t, x, dyn), m_inv)

    sol = solve_ivp(
        rhs,
        (log.t[0], log.t[-1]),
        initial_state(log, dyn.rest_state),
        t_eval=log.t,
        rtol=rtol,
        atol=1e-8,
        max_step=dt,
    )
    if not sol.success or sol.y.shape[1] != len(log.t):
        return np.full((len(log.t), 12), np.nan)
    return sol.y.T


def _channel_error(name: str, sim: np.ndarray, meas: np.ndarray) -> np.ndarray:
    err = sim - meas
    if name in ANGLES:
        err = (err + np.pi) % (2 * np.pi) - np.pi
    return err


def fit_percent(model: VehicleModel, log: Log) -> dict[str, float]:
    """MATLAB-style fit per channel: 100 (1 - |y - y_sim| / |y - mean(y)|)."""
    sim = simulate_log(model, log)
    out = {}
    for name, meas in log.states.items():
        err = _channel_error(name, sim[:, STATES.index(name)], meas)
        spread = np.linalg.norm(meas - meas.mean())
        out[name] = float(100 * (1 - np.linalg.norm(err) / spread)) if spread > 1e-12 else 0.0
    return out


@dataclass(frozen=True)
class CalibrationResult:
    calibration: Calibration
    model: VehicleModel
    fit_before: dict[str, dict[str, float]] = field(default_factory=dict)  # log -> channel
    fit_after: dict[str, dict[str, float]] = field(default_factory=dict)
    validation_before: dict[str, dict[str, float]] = field(default_factory=dict)
    validation_after: dict[str, dict[str, float]] = field(default_factory=dict)
    cost: float = 0.0


def _layout(groups: tuple[str, ...]) -> list[tuple[str, int]]:
    out = []
    for g in groups:
        if g not in GROUPS:
            raise ValueError(f"unknown group {g!r}; choose from {list(GROUPS)}")
        out += [(g, i) for i in range(6)] if g in ("damping", "added_mass") else [(g, 0)]
    return out


def _to_calibration(
    layout, theta: np.ndarray, sigma: np.ndarray | None = None, note: str = ""
) -> Calibration:
    values = {"damping": np.ones(6), "added_mass": np.ones(6), "fin_lift": 1.0, "body_lift": 1.0,
              "thrust": 1.0}  # fmt: skip
    sig = {"damping": np.zeros(6), "added_mass": np.zeros(6), "fin_lift": 0.0, "body_lift": 0.0,
           "thrust": 0.0}  # fmt: skip
    for (group, i), th, s in zip(
        layout, theta, sigma if sigma is not None else np.zeros(len(theta)), strict=True
    ):
        if group in ("damping", "added_mass"):
            values[group][i] = np.exp(th)
            sig[group][i] = s
        else:
            values[group] = float(np.exp(th))
            sig[group] = float(s)
    return Calibration(
        damping=values["damping"],
        added_mass=values["added_mass"],
        fin_lift=values["fin_lift"],
        body_lift=values["body_lift"],
        thrust=values["thrust"],
        damping_sigma=sig["damping"],
        added_mass_sigma=sig["added_mass"],
        fin_lift_sigma=sig["fin_lift"],
        body_lift_sigma=sig["body_lift"],
        thrust_sigma=sig["thrust"],
        note=note,
    )


def calibrate(
    model: VehicleModel,
    logs: list[Log],
    groups: tuple[str, ...] = DEFAULT_GROUPS,
    validation: list[Log] | None = None,
    noise: dict[str, float] | None = None,
    max_evaluations: int = 400,
    progress=None,
) -> CalibrationResult:
    """Fit scale factors for `groups` to `logs`; report fit on `validation` logs too.

    progress(n_simulations, cost) is called after every model evaluation, if given.
    """
    if not logs:
        raise ValueError("calibration needs at least one log")
    noise = {**DEFAULT_NOISE, **(noise or {})}
    layout = _layout(groups)
    if model.body_lift is None:
        layout = [(g, i) for g, i in layout if g != "body_lift"]
    if not model.fins.control_names and not model.vehicle.fins:
        layout = [(g, i) for g, i in layout if g != "fin_lift"]
    if not model.vehicle.thrusters:
        layout = [(g, i) for g, i in layout if g != "thrust"]
    prior = np.array([PRIOR_LOG_SIGMA[g] for g, _ in layout])

    def model_for(theta):
        return apply_calibration(model, _to_calibration(layout, theta))

    calls = [0]

    def residuals(theta):
        calls[0] += 1
        candidate = model_for(theta)
        parts = []
        for log in logs:
            sim = simulate_log(candidate, log)
            for name, meas in log.states.items():
                err = _channel_error(name, sim[:, STATES.index(name)], meas)
                parts.append(np.nan_to_num(err / noise[name], nan=1e3))
        parts.append(theta / prior)
        out = np.concatenate(parts)
        if progress is not None:
            progress(calls[0], float(0.5 * out @ out))
        return out

    fit = least_squares(
        residuals,
        np.zeros(len(layout)),
        bounds=BOUNDS,
        x_scale=1.0,
        diff_step=1e-3,
        max_nfev=max_evaluations,
    )
    try:
        cov = np.linalg.inv(fit.jac.T @ fit.jac)
        log_sigma = np.sqrt(np.clip(np.diag(cov), 0, None))
    except np.linalg.LinAlgError:
        log_sigma = np.full(len(layout), np.inf)
    rel_sigma = np.expm1(log_sigma)  # relative 1-sigma of the scale
    names = ", ".join(log.name for log in logs)
    cal = _to_calibration(layout, fit.x, rel_sigma, note=f"fitted to {names}")
    calibrated = apply_calibration(model, cal)
    validation = validation or []
    return CalibrationResult(
        calibration=cal,
        model=calibrated,
        fit_before={log.name: fit_percent(model, log) for log in logs},
        fit_after={log.name: fit_percent(calibrated, log) for log in logs},
        validation_before={log.name: fit_percent(model, log) for log in validation},
        validation_after={log.name: fit_percent(calibrated, log) for log in validation},
        cost=float(fit.cost),
    )


def calibration_to_yaml(cal: Calibration) -> dict:
    return {
        "damping": [round(float(x), 6) for x in cal.damping],
        "damping_sigma": [round(float(x), 6) for x in cal.damping_sigma],
        "added_mass": [round(float(x), 6) for x in cal.added_mass],
        "added_mass_sigma": [round(float(x), 6) for x in cal.added_mass_sigma],
        "fin_lift": round(cal.fin_lift, 6),
        "fin_lift_sigma": round(cal.fin_lift_sigma, 6),
        "body_lift": round(cal.body_lift, 6),
        "body_lift_sigma": round(cal.body_lift_sigma, 6),
        "thrust": round(cal.thrust, 6),
        "thrust_sigma": round(cal.thrust_sigma, 6),
        "note": cal.note,
    }


def summary(result: CalibrationResult) -> str:
    cal = result.calibration
    lines = ["Calibration (scale x CAD value, ± relative 1 sigma)"]
    for i, dof in enumerate(DOF):
        if cal.damping[i] != 1 or cal.damping_sigma[i]:
            lines.append(f"  damping {dof:<6} x{cal.damping[i]:.3g} ± {cal.damping_sigma[i]:.0%}")
    for i, dof in enumerate(DOF):
        if cal.added_mass[i] != 1:
            lines.append(
                f"  added mass {dof:<6} x{cal.added_mass[i]:.3g} ± {cal.added_mass_sigma[i]:.0%}"
            )
    if cal.fin_lift != 1:
        lines.append(f"  fin lift       x{cal.fin_lift:.3g} ± {cal.fin_lift_sigma:.0%}")
    if cal.body_lift != 1:
        lines.append(f"  body lift      x{cal.body_lift:.3g} ± {cal.body_lift_sigma:.0%}")
    if cal.thrust != 1:
        lines.append(f"  thrust         x{cal.thrust:.3g} ± {cal.thrust_sigma:.0%}")

    def table(title, before, after):
        if not before:
            return
        lines.extend(["", title + " (fit %, 100 = perfect)"])
        for log in before:
            pairs = ", ".join(
                f"{ch} {before[log][ch]:.0f}->{after[log][ch]:.0f}" for ch in before[log]
            )
            lines.append(f"  {log}: {pairs}")

    table("Calibration logs", result.fit_before, result.fit_after)
    table("Validation logs", result.validation_before, result.validation_after)
    return "\n".join(lines)
