"""Ground robots (wheeled, skid-steer, tracked): planar dynamics with wheel-ground contact.

State x = [x, y, psi, u, v, r]: position and heading on flat ground (x north, y east, psi
from north to the right) and the body velocities of the centre of gravity.

    m (u_dot - v r) = sum Fx,   m (v_dot + u r) = sum Fy,   I_z r_dot = sum (x_i Fy_i - y_i Fx_i)

From the CAD and the parts list: mass, centre of gravity (also its height, which sets the
tip-over limits), yaw inertia, wheel layout and how the weight is shared between the wheels.

Contact forces at each wheel, in the wheel's own frame (turned by its steering angle):
* drive: a DC motor's straight torque-speed line, tau = tau_stall (command - omega / omega_0),
  with the wheel rolling without slip (omega = v_long / R);
* rolling resistance: C_rr N, smoothed around zero speed;
* side force: friction mu N, growing with the slip angle (peak near SLIP_ANGLE) and smoothed
  around zero speed (regularised Coulomb friction); a caster gives none;
* the total is held inside the friction circle mu N.
Not from the CAD: mu and C_rr (surface; typical values with a range), motor data (datasheet).
Not modelled: load transfer while accelerating, wheel spin inertia, suspension, slopes.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np
from scipy.integrate import solve_ivp
from scipy.spatial import ConvexHull, QhullError

from robomodel.checks import FAIL, OK, WARN, Check
from robomodel.geometry import build_mesh
from robomodel.hydrostatics import GRAVITY
from robomodel.rigid_body import MassProperties, mass_properties
from robomodel.vehicle import Vehicle

CONTACTS_PER_TRACK = 5
SLIP_SPEED = 0.02  # m/s: below this, friction is smoothed (regularised Coulomb), assumed
SLIP_ANGLE = 0.1  # rad (~6 deg): side force near its peak at this slip angle (rubber), assumed
GOOD_TIP_ANGLE = 20.0  # deg

# Model range (low = least grip, most resistance, least motor; high = the opposite)
RANGE = {"low": (0.7, 2.0, 0.85), "high": (1.2, 0.5, 1.15)}  # mu, C_rr, motor torque


class GroundError(ValueError):
    pass


@dataclass(frozen=True)
class GroundModel:
    vehicle: Vehicle
    hull: object
    mass: MassProperties
    ground_z: float  # body z of the ground plane (z down), m
    cg_height: float  # CG above the ground, m
    contact_xy: np.ndarray  # (k, 2) contact points relative to the CG, body axes, m
    contact_wheel: np.ndarray  # (k,) wheel index of each contact
    normal: np.ndarray  # (k,) normal load at rest, N
    stall_force: np.ndarray  # (k,) drive force at full command, standstill, N (0: not driven)
    free_speed: np.ndarray  # (k,) ground speed at full command, no load, m/s
    drive_input: np.ndarray  # (k,) index into the drive commands, -1 if not driven
    steer_input: np.ndarray  # (k,) index into the steering angles, -1 if not steered
    caster: np.ndarray  # (k,) bool
    mu: float
    rolling: float
    label: str = "nominal"
    warnings: list[str] = field(default_factory=list)

    @property
    def m(self) -> float:
        return float(self.mass.mass.value)

    @property
    def izz(self) -> float:
        return float(self.mass.inertia_g.value[2, 2])

    @property
    def weight(self) -> float:
        return self.m * GRAVITY

    @property
    def driven_wheels(self) -> list[int]:
        return [i for i, w in enumerate(self.vehicle.wheels) if w.driven]

    @property
    def steered_wheels(self) -> list[int]:
        return [i for i, w in enumerate(self.vehicle.wheels) if w.steered]

    @property
    def n_inputs(self) -> int:
        return len(self.driven_wheels) + len(self.steered_wheels)


def build_ground_model(vehicle: Vehicle) -> GroundModel:
    if not vehicle.wheels:
        raise GroundError("a ground robot needs wheels (or tracks): list them under 'wheels'")
    hull = build_mesh(vehicle.geometry)
    props = mass_properties(vehicle, hull)
    r_g = np.asarray(props.r_g.value, dtype=float)
    wheels = vehicle.wheels
    ground_z = max(float(w.position[2] + w.radius) for w in wheels)
    xy, owner, share = [], [], []
    for i, w in enumerate(wheels):
        if w.track_length > 0:
            offsets = np.linspace(-w.track_length / 2, w.track_length / 2, CONTACTS_PER_TRACK)
        else:
            offsets = np.zeros(1)
        for dx in offsets:
            xy.append([w.position[0] + dx - r_g[0], w.position[1] - r_g[1]])
            owner.append(i)
            share.append(1.0 / len(offsets))
    xy, owner, share = np.array(xy), np.array(owner), np.array(share)
    weight = props.mass.value * GRAVITY
    normal = _normal_loads(xy, weight)

    driven = [i for i, w in enumerate(wheels) if w.driven]
    steered = [i for i, w in enumerate(wheels) if w.steered]
    stall = np.array(
        [share[k] * wheels[i].stall_torque / wheels[i].radius for k, i in enumerate(owner)]
    )
    free = np.array([wheels[i].radius * wheels[i].no_load_speed for i in owner])
    drive_in = np.array([driven.index(i) if i in driven else -1 for i in owner])
    steer_in = np.array([steered.index(i) if i in steered else -1 for i in owner])
    _, mu, crr = vehicle.surface
    model = GroundModel(
        vehicle=vehicle,
        hull=hull,
        mass=props,
        ground_z=ground_z,
        cg_height=ground_z - float(r_g[2]),
        contact_xy=xy,
        contact_wheel=owner,
        normal=normal,
        stall_force=stall,
        free_speed=free,
        drive_input=drive_in,
        steer_input=steer_in,
        caster=np.array([wheels[i].caster for i in owner]),
        mu=mu,
        rolling=crr,
    )
    return replace(model, warnings=_warnings(model))


def _normal_loads(xy: np.ndarray, weight: float) -> np.ndarray:
    """Static wheel loads: vertical force and both tilting moments balance; with more than
    three contacts the loads are statically indeterminate, so the smoothest (least-squares)
    sharing is taken, which is exact for a rigid, level chassis on symmetric wheels."""
    a = np.vstack([np.ones(len(xy)), xy[:, 0], xy[:, 1]])
    b = np.array([weight, 0.0, 0.0])
    loads, *_ = np.linalg.lstsq(a, b, rcond=None)
    if np.max(np.abs(a @ loads - b)) > 1e-6 * weight:
        raise GroundError(
            "the wheels cannot hold the robot up: they touch the ground in one line (or one "
            "point) that does not pass under the centre of gravity; add a caster or wheel"
        )
    return loads


# ---------- equations of motion ----------


def contact_forces(gm: GroundModel, nu: np.ndarray, inputs: np.ndarray) -> np.ndarray:
    """Body-frame force (Fx, Fy) at every contact, shape (k, 2)."""
    u, v, r = nu
    x, y = gm.contact_xy[:, 0], gm.contact_xy[:, 1]
    n_drive = len(gm.driven_wheels)
    commands = np.clip(inputs[:n_drive], -1.0, 1.0)
    steer_cmd = inputs[n_drive:]
    delta = np.zeros(len(x))
    s_idx = gm.steer_input >= 0
    if s_idx.any():
        limits = np.array([gm.vehicle.wheels[i].max_steer for i in gm.steered_wheels])
        angles = np.clip(steer_cmd, -limits, limits)
        delta[s_idx] = angles[gm.steer_input[s_idx]]
    vx, vy = u - r * y, v + r * x
    c, s = np.cos(delta), np.sin(delta)
    v_long, v_lat = c * vx + s * vy, -s * vx + c * vy
    n = gm.normal
    grip = gm.mu * n
    f_long = -gm.rolling * n * np.tanh(v_long / SLIP_SPEED)
    d_idx = gm.drive_input >= 0
    if d_idx.any():
        cmd = commands[gm.drive_input[d_idx]]
        f_long[d_idx] += gm.stall_force[d_idx] * (cmd - v_long[d_idx] / gm.free_speed[d_idx])
    f_lat = -grip * np.tanh(v_lat / (SLIP_SPEED + SLIP_ANGLE * np.abs(v_long)))
    f_lat = np.where(gm.caster, 0.0, f_lat)
    total = np.hypot(f_long, f_lat)
    scale = np.where(total > grip, grip / np.maximum(total, 1e-12), 1.0)  # friction circle
    f_long, f_lat = f_long * scale, f_lat * scale
    return np.column_stack([c * f_long - s * f_lat, s * f_long + c * f_lat])


def state_derivative(gm: GroundModel, state: np.ndarray, inputs: np.ndarray) -> np.ndarray:
    psi, u, v, r = state[2], state[3], state[4], state[5]
    f = contact_forces(gm, state[3:], np.asarray(inputs, dtype=float))
    fx, fy = f[:, 0].sum(), f[:, 1].sum()
    mz = float(np.sum(gm.contact_xy[:, 0] * f[:, 1] - gm.contact_xy[:, 1] * f[:, 0]))
    return np.array(
        [
            u * np.cos(psi) - v * np.sin(psi),
            u * np.sin(psi) + v * np.cos(psi),
            r,
            fx / gm.m + v * r,
            fy / gm.m - u * r,
            mz / gm.izz,
        ]
    )


@dataclass(frozen=True)
class GroundTrajectory:
    t: np.ndarray
    state: np.ndarray  # (n, 6)


def simulate(
    gm: GroundModel,
    t_end: float,
    inputs=None,
    state0=None,
    dt_out: float = 0.05,
    rtol: float = 1e-8,
    atol: float = 1e-10,
) -> GroundTrajectory:
    """inputs: constant vector [drive commands..., steering angles (rad)...] or f(t, x)."""
    state0 = np.zeros(6) if state0 is None else np.asarray(state0, dtype=float)
    if inputs is None:
        inputs = np.zeros(gm.n_inputs)
    if callable(inputs):
        fn = inputs
    else:
        constant = np.asarray(inputs, dtype=float)

        def fn(_t, _x):
            return constant

    t_eval = np.arange(0.0, t_end + dt_out / 2, dt_out)
    t_eval = t_eval[t_eval <= t_end]
    # Friction near standstill is stiff: an implicit solver keeps the steps reasonable.
    sol = solve_ivp(
        lambda t, x: state_derivative(gm, x, fn(t, x)),
        (0.0, t_end),
        state0,
        method="LSODA",
        t_eval=t_eval,
        rtol=rtol,
        atol=atol,
    )
    if not sol.success:
        raise RuntimeError(f"simulation failed: {sol.message}")
    return GroundTrajectory(sol.t, sol.y.T)


# ---------- stability, performance, checks ----------


def tip_limits(gm: GroundModel) -> dict[str, float]:
    """Static tip-over angle (deg) towards each side of the support polygon, and the matching
    acceleration limit (m/s^2): a = g d / h, d = distance from the CG to that edge."""
    try:
        hull = ConvexHull(gm.contact_xy)
    except QhullError:
        return {}
    out: dict[str, float] = {}
    h = max(gm.cg_height, 1e-9)
    for eq in hull.equations:  # n . p + c <= 0 inside; the CG is at the origin
        normal, offset = eq[:2], eq[2]
        d = -offset  # distance from the CG to the edge line
        side = (
            ("front" if normal[0] > 0 else "rear")
            if abs(normal[0]) >= abs(normal[1])
            else ("right" if normal[1] > 0 else "left")
        )
        angle = float(np.degrees(np.arctan2(d, h)))
        if angle < out.get(side, np.inf):
            out[side] = angle
    return out


def performance(gm: GroundModel) -> dict[str, float]:
    """Plain numbers: top speed, acceleration, climbable slope, turning."""
    out: dict[str, float] = {}
    driven = gm.drive_input >= 0
    if not driven.any():
        return out
    w = gm.weight
    f_stall = float(gm.stall_force[driven].sum())
    f_roll = gm.rolling * w
    # straight line at full command: sum F_s (1 - v / v_0) = C_rr W
    slope = float(np.sum(gm.stall_force[driven] / gm.free_speed[driven]))
    out["top_speed"] = max((f_stall - f_roll) / slope, 0.0)
    grip_driven = gm.mu * float(gm.normal[driven].sum())
    out["max_acceleration"] = max(min(f_stall, grip_driven) - f_roll, 0.0) / gm.m
    # climbing a slope: traction (driven wheels' load share, level loads) and motor force
    share = float(gm.normal[driven].sum()) / w
    traction = np.degrees(np.arctan(gm.mu * share))
    motor = np.degrees(np.arcsin(min(1.0, f_stall / w))) if f_stall < w else 90.0
    out["max_slope"] = float(min(traction, motor))
    # turning: spin in place (driven wheels on both sides) or tightest steering radius
    left = driven & (gm.contact_xy[:, 1] < -1e-9)
    right = driven & (gm.contact_xy[:, 1] > 1e-9)
    if left.any() and right.any():
        n_drive = len(gm.driven_wheels)
        cmd = np.zeros(gm.n_inputs)
        for k in np.flatnonzero(driven):
            cmd[gm.drive_input[k]] = 1.0 if gm.contact_xy[k, 1] < 0 else -1.0
        cmd[n_drive:] = 0.0
        traj = simulate(gm, 6.0, cmd, dt_out=0.5, rtol=1e-6, atol=1e-8)
        out["turn_rate"] = float(abs(traj.state[-1, 5]))
    if gm.steered_wheels:
        steered = gm.steer_input >= 0
        x_steer = float(
            np.mean(np.abs(gm.contact_xy[steered, 0] - gm.contact_xy[~steered, 0].mean()))
        )
        dmax = min(gm.vehicle.wheels[i].max_steer for i in gm.steered_wheels)
        if x_steer > 0 and (~steered).any():
            out["min_turn_radius"] = float(x_steer / np.tan(dmax))
    return out


def run_ground_checks(gm: GroundModel) -> list[Check]:
    out = [
        Check(
            "mass and yaw inertia positive",
            OK if gm.m > 0 and gm.izz > 0 else FAIL,
            f"m = {gm.m:.4g} kg, I_z = {gm.izz:.4g} kg m^2",
        )
    ]
    low = float(gm.normal.min())
    out.append(
        Check(
            "stands on its wheels (CG inside the wheel base)",
            OK if low > 0 else FAIL,
            "wheel loads "
            + ", ".join(f"{x:.3g}" for x in gm.normal)
            + " N"
            + ("" if low > 0 else ": a load is negative, the robot tips over"),
        )
    )
    tips = tip_limits(gm)
    if tips:
        worst = min(tips, key=tips.get)
        out.append(
            Check(
                f"tip-over margin at least {GOOD_TIP_ANGLE:g} deg",
                OK if tips[worst] >= GOOD_TIP_ANGLE else WARN,
                ", ".join(f"{k} {v:.0f} deg" for k, v in tips.items())
                + ("" if tips[worst] >= GOOD_TIP_ANGLE else f": tips {worst} easily"),
            )
        )
    driven = gm.drive_input >= 0
    out.append(
        Check(
            "has driven wheels",
            OK if driven.any() else FAIL,
            f"{len(gm.driven_wheels)} driven wheel(s)",
        )
    )
    y = gm.contact_xy[:, 1]
    turns = bool(gm.steered_wheels) or (
        np.any(driven & (y < -1e-9)) and np.any(driven & (y > 1e-9))
    )
    out.append(
        Check(
            "can turn (steering or driven wheels on both sides)",
            OK if turns else FAIL,
            "steered" if gm.steered_wheels else ("differential drive" if turns else "cannot turn"),
        )
    )
    # Coasting must lose energy: the contact forces only dissipate when no motor drives.
    state0 = np.array([0, 0, 0, 1.0, 0.2, 0.5])
    traj = simulate(
        gm, 2.0, np.zeros(gm.n_inputs), state0=state0, dt_out=0.05, rtol=1e-7, atol=1e-9
    )
    nu = traj.state[:, 3:]
    energy = 0.5 * gm.m * (nu[:, 0] ** 2 + nu[:, 1] ** 2) + 0.5 * gm.izz * nu[:, 2] ** 2
    grows = float(np.max(np.diff(energy)) / energy[0])
    out.append(
        Check(
            "coasting loses energy (simulation)",
            OK if grows < 1e-6 else FAIL,
            f"largest relative energy increase per step {grows:.1e}",
        )
    )
    return out


def _warnings(gm: GroundModel) -> list[str]:
    from robomodel import mass_parts

    v = gm.vehicle
    out = list(getattr(gm.hull, "metadata", {}).get("warnings", []))
    out += mass_parts.warnings(gm.mass.distribution)
    extents = np.asarray(gm.hull.extents)
    if extents.max() > 20 or extents.max() < 0.02:
        out.append(f"the robot is {extents.max():.3g} m long: check the file units (mm / m)")
    if gm.cg_height <= 0:
        out.append("the centre of gravity is below the ground: check wheel radii and positions")
    if not v.parts and gm.mass.distribution is None:
        out.append(
            "no parts listed: the mass is spread over the CAD volume; list the battery and "
            "motors (and give wheel mass) for a correct CG height and inertia"
        )
    if v.measured.r_g is None:
        out.append(
            "tip-over limits depend on the CG height: measure it (or list heavy parts) if the "
            "robot is tall or drives on slopes"
        )
    out.append(
        f"surface '{v.surface[0]}': mu = {gm.mu:g}, rolling resistance {gm.rolling:g} are "
        "typical values, not measured; the low/high models span 0.7-1.2 x mu"
    )
    out.append(
        "not modelled: load transfer while accelerating or turning, wheel spin inertia, "
        "suspension, slopes and bumps"
    )
    return out


def model_range(gm: GroundModel) -> dict[str, GroundModel]:
    out = {"nominal": gm}
    for label, (k_mu, k_crr, k_motor) in RANGE.items():
        out[label] = replace(
            gm,
            mu=gm.mu * k_mu,
            rolling=gm.rolling * k_crr,
            stall_force=gm.stall_force * k_motor,
            label=label,
        )
    return {k: out[k] for k in ("low", "nominal", "high")}


def behaviour(gm: GroundModel) -> dict:
    out: dict = {}
    try:
        for label, variant in model_range(gm).items():
            for key, value in performance(variant).items():
                out.setdefault(key, {})[label] = float(value)
        for key, value in tip_limits(gm).items():
            out.setdefault(f"tip_{key}", {})["nominal"] = value
    except Exception as e:  # a summary must never break the build
        out["error"] = str(e)
    return out


def report(gm: GroundModel) -> str:
    v = gm.vehicle
    lines = [f"{v.name}  (ground)", ""]
    if gm.mass.distribution is not None:
        from robomodel.mass_parts import summary as parts_summary

        lines += ["Mass over the CAD parts (material guessed from the part names)"]
        lines += ["  " + x for x in parts_summary(gm.mass.distribution)] + [""]
    lines.append("Mass and contact")
    lines.append(f"  {'mass':<18}{gm.m:.4g} kg")
    lines.append(f"  {'CG r_g':<18}[" + ", ".join(f"{x:.4g}" for x in gm.mass.r_g.value) + "] m")
    lines.append(f"  {'CG height':<18}{gm.cg_height:.4g} m above the ground")
    lines.append(f"  {'yaw inertia I_z':<18}{gm.izz:.4g} kg m^2")
    lines.append(f"  {'surface':<18}{v.surface[0]}: mu {gm.mu:g}, rolling {gm.rolling:g}")
    lines += ["", "Wheels (contact points relative to the CG)"]
    for k, (xy, n) in enumerate(zip(gm.contact_xy, gm.normal, strict=True)):
        w = v.wheels[gm.contact_wheel[k]]
        kind = "caster" if w.caster else ("driven" if w.driven else "free")
        if w.steered:
            kind += ", steered"
        lines.append(
            f"  {w.name:<10} x {xy[0]:+.3f} y {xy[1]:+.3f} m  load {n:.4g} N  {kind}"
            + (
                f"  stall force {gm.stall_force[k]:.3g} N, free speed {gm.free_speed[k]:.3g} m/s"
                if w.driven
                else ""
            )
        )
    tips = tip_limits(gm)
    if tips:
        lines += ["", "Tip-over (static angle; acceleration limit g tan(angle))"]
        for side, ang in tips.items():
            lines.append(
                f"  {side:<8}{ang:5.1f} deg   {GRAVITY * np.tan(np.radians(ang)):.3g} m/s^2"
            )
    perf = performance(gm)
    if perf:
        lines += ["", "Performance (nominal)"]
        units = {
            "top_speed": "m/s",
            "max_acceleration": "m/s^2",
            "max_slope": "deg",
            "turn_rate": "rad/s",
            "min_turn_radius": "m",
        }
        lines += [f"  {k:<18}{val:.4g} {units[k]}" for k, val in perf.items()]
    from robomodel.checks import summary

    lines += ["", "Checks:", summary(run_ground_checks(gm))]
    if gm.warnings:
        lines += ["", "Warnings:"] + [f"  ! {w}" for w in gm.warnings]
    lines += [
        "",
        "Inputs: ["
        + ", ".join(
            [f"drive {v.wheels[i].name} (-1..1)" for i in gm.driven_wheels]
            + [f"steer {v.wheels[i].name} (rad)" for i in gm.steered_wheels]
        )
        + "]",
    ]
    return "\n".join(lines)


def to_dict(gm: GroundModel) -> dict:
    from robomodel.provenance import provenance

    v = gm.vehicle
    return {
        "name": v.name,
        "category": "ground",
        "frame": "planar: x forward, y right, z down; state [x, y, psi, u, v, r] at the CG",
        "mass": gm.m,
        "r_g": np.asarray(gm.mass.r_g.value).tolist(),
        "cg_height": gm.cg_height,
        "I_z": gm.izz,
        "inertia_g": np.asarray(gm.mass.inertia_g.value).tolist(),
        "surface": {"name": v.surface[0], "mu": gm.mu, "rolling": gm.rolling},
        "contacts": [
            {
                "wheel": v.wheels[i].name,
                "xy_from_cg": xy.tolist(),
                "normal_load": float(n),
                "stall_force": float(f),
                "free_speed": float(s),
            }
            for i, xy, n, f, s in zip(
                gm.contact_wheel,
                gm.contact_xy,
                gm.normal,
                gm.stall_force,
                gm.free_speed,
                strict=True,
            )
        ],
        "inputs": [f"drive:{v.wheels[i].name}" for i in gm.driven_wheels]
        + [f"steer:{v.wheels[i].name}" for i in gm.steered_wheels],
        "tip_limits_deg": tip_limits(gm),
        "performance": performance(gm),
        "warnings": gm.warnings,
        "checks": [
            {"name": c.name, "status": c.status, "detail": c.detail} for c in run_ground_checks(gm)
        ],
        "provenance": provenance(v),
    }
