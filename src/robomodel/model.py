"""Assemble the vehicle model from its description, and report it."""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np
import trimesh

from robomodel.added_mass import (
    ellipsoid_added_mass_parameter,
    strip_theory_parameter,
)
from robomodel.body_lift import body_lift
from robomodel.damping import Damping, estimate_damping
from robomodel.dynamics import Dynamics
from robomodel.fins import FinModel, fin_model
from robomodel.geometry import build_mesh
from robomodel.hydrostatics import Hydrostatics, hydrostatics
from robomodel.parameter import Parameter, Source
from robomodel.rigid_body import MassProperties, mass_properties, rigid_body_mass_matrix
from robomodel.sections import Sections, slice_hull
from robomodel.thrusters import allocation_matrix, thrust_rel_sigma
from robomodel.vehicle import Calibration, Vehicle

NEUTRAL_BUOYANCY_TOLERANCE = 0.02  # |W - B| / B above this gets a warning


@dataclass(frozen=True)
class VehicleModel:
    vehicle: Vehicle
    hull: trimesh.Trimesh
    sections: Sections
    mass: MassProperties
    statics: Hydrostatics
    m_rb: np.ndarray  # 6x6, about CO
    m_a: Parameter  # 6x6 added mass (positive)
    damping: Damping
    b_thr: np.ndarray  # 6 x n thruster allocation (per newton of thrust)
    fins: FinModel
    body_lift: Parameter | None = None  # 6x6, multiply by |u|; None for stubby hulls
    warnings: list[str] = field(default_factory=list)

    def dynamics(self) -> Dynamics:
        return Dynamics(
            m_rb=self.m_rb,
            m_a=np.asarray(self.m_a.value),
            d_lin=np.asarray(self.damping.linear.value),
            d_quad=np.asarray(self.damping.quadratic.value),
            weight=float(self.statics.weight.value),
            buoyancy=float(self.statics.buoyancy.value),
            r_g=np.asarray(self.mass.r_g.value),
            r_b=np.asarray(self.statics.r_b.value),
            b_thr=self.b_thr,
            d_fin=np.asarray(self.fins.d_fin.value)
            + (0 if self.body_lift is None else np.asarray(self.body_lift.value)),
            b_fin=self.fins.b_fin,
        )


def build_model(vehicle: Vehicle) -> VehicleModel:
    hull = build_mesh(vehicle.geometry)
    sections = slice_hull(hull)
    props = mass_properties(vehicle, hull)
    statics = hydrostatics(hull, props.mass, vehicle.water_density)
    rho = vehicle.water_density

    g = vehicle.geometry
    if g.primitive == "sphere":
        r = g.dims["radius"]
        m_a = ellipsoid_added_mass_parameter(r, r, r, rho)
    elif g.primitive == "ellipsoid":
        m_a = ellipsoid_added_mass_parameter(g.dims["a"], g.dims["b"], g.dims["c"], rho)
    else:
        m_a = strip_theory_parameter(sections, float(hull.volume), rho)

    damping = estimate_damping(sections, rho)
    fins = fin_model(vehicle.fins, rho)
    if vehicle.fins:  # fin added mass adds to the hull's, with the hull's uncertainty
        m_a = Parameter(
            m_a.value + fins.m_a,
            m_a.sigma + 0.3 * np.abs(fins.m_a),
            m_a.source,
            m_a.unit,
            m_a.note + "; fins as flat plates",
        )
    model = VehicleModel(
        vehicle=vehicle,
        hull=hull,
        sections=sections,
        mass=props,
        statics=statics,
        m_rb=rigid_body_mass_matrix(props),
        m_a=m_a,
        damping=damping,
        b_thr=allocation_matrix(vehicle.thrusters),
        fins=fins,
        body_lift=body_lift(sections, float(hull.bounds[1, 0]), rho),
        warnings=_warnings(vehicle, props, statics),
    )
    if not vehicle.calibration.is_identity:
        model = apply_calibration(model, vehicle.calibration)
    return model


def _scaled(param: Parameter, scale, rel_sigma, note: str) -> Parameter:
    """Multiply a parameter by fitted scale factors; it becomes a calibrated value."""
    value = np.asarray(param.value, dtype=float) * scale
    sigma = np.abs(value) * rel_sigma
    if np.ndim(value) == 0:
        value, sigma = float(value), float(sigma)
    return Parameter(value, sigma, Source.CALIBRATED, param.unit, note)


def apply_calibration(model: VehicleModel, cal: Calibration) -> VehicleModel:
    """Apply scale factors from test data. Cheap: no geometry is recomputed."""
    note = "calibrated to test data" + (f" ({cal.note})" if cal.note else "")
    d = model.damping
    damping = replace(
        d,
        linear=_scaled(d.linear, cal.damping, cal.damping_sigma, note),
        quadratic=_scaled(d.quadratic, cal.damping, cal.damping_sigma, note),
    )
    m_a = model.m_a
    if not np.allclose(cal.added_mass, 1):
        diag = np.sqrt(cal.added_mass)  # scale rows and columns: diagonal x s, couplings x sqrt
        m_a = _scaled(
            m_a,
            np.outer(diag, diag),
            np.sqrt(np.add.outer(cal.added_mass_sigma, cal.added_mass_sigma) / 2),
            note,
        )
    fins = model.fins
    if cal.fin_lift != 1:
        fins = replace(
            fins,
            d_fin=_scaled(fins.d_fin, cal.fin_lift, cal.fin_lift_sigma, note),
            b_fin=fins.b_fin * cal.fin_lift,
        )
    lift = model.body_lift
    if lift is not None and cal.body_lift != 1:
        lift = _scaled(lift, cal.body_lift, cal.body_lift_sigma, note)
    return replace(
        model,
        damping=damping,
        m_a=m_a,
        fins=fins,
        body_lift=lift,
        vehicle=replace(model.vehicle, calibration=cal),
    )


def _warnings(vehicle: Vehicle, props: MassProperties, statics: Hydrostatics) -> list[str]:
    out = []
    w, b = statics.weight.value, statics.buoyancy.value
    if abs(w - b) / b > NEUTRAL_BUOYANCY_TOLERANCE:
        kind = "sinks" if w > b else "floats up"
        out.append(f"not neutrally buoyant: W - B = {w - b:+.3g} N, the vehicle {kind}")
    bg = props.r_g.value[2] - statics.r_b.value[2]  # z down: CG below CB means bg > 0
    if bg <= 0:
        out.append(
            f"centre of gravity is not below the centre of buoyancy (BG_z = {bg:.3g} m): "
            "roll and pitch are not self-righting"
        )
    if not vehicle.thrusters:
        out.append("no thrusters: the model has no control inputs")
    elif any(t.preset for t in vehicle.thrusters):
        out.append("thrust limits come from a preset; measure them or give max_forward/max_reverse")
    return out


def _p(param: Parameter) -> dict:
    return {
        "value": np.asarray(param.value).tolist(),
        "sigma": np.asarray(param.sigma).tolist(),
        "source": param.source.value,
        "unit": param.unit,
        **({"note": param.note} if param.note else {}),
    }


def to_dict(model: VehicleModel) -> dict:
    """Plain data for saving as YAML/JSON: every value with sigma and source."""
    v = model.vehicle
    return {
        "name": v.name,
        "frame": "Fossen body frame: x forward, y right, z down; about CO (docs/eksenler.md)",
        "equations": "M nu_dot + C(nu) nu + D(nu) nu + g(eta) = tau, M = M_RB + M_A",
        "mass": _p(model.mass.mass),
        "r_g": _p(model.mass.r_g),
        "inertia_o": _p(model.mass.inertia_o),
        "inertia_g": _p(model.mass.inertia_g),
        "volume": _p(model.statics.volume),
        "r_b": _p(model.statics.r_b),
        "weight": _p(model.statics.weight),
        "buoyancy": _p(model.statics.buoyancy),
        "M_RB": np.round(model.m_rb, 12).tolist(),
        "M_A": _p(model.m_a),
        "D_linear": _p(model.damping.linear),
        "D_quadratic": _p(model.damping.quadratic),
        "thrusters": [
            {
                "model": t.model,
                "position": t.position.tolist(),
                "direction": t.direction.tolist(),
                "max_forward": t.max_forward,
                "max_reverse": t.max_reverse,
                "rel_sigma": thrust_rel_sigma(t),
            }
            for t in v.thrusters
        ],
        "B_thr": np.round(model.b_thr, 12).tolist(),
        "D_fin": _p(model.fins.d_fin),
        "B_fin": np.round(model.fins.b_fin, 12).tolist(),
        "fin_controls": list(model.fins.control_names),
        "D_body_lift": _p(model.body_lift) if model.body_lift is not None else None,
        "warnings": model.warnings,
    }


def _clean(values: np.ndarray) -> np.ndarray:
    values = np.ravel(values)
    scale = max(float(np.abs(values).max(initial=0.0)), 1.0)
    return np.where(np.abs(values) < 1e-12 * scale, 0.0, values)


def report(model: VehicleModel) -> str:
    """Human-readable summary for the terminal."""
    m, s, d = model.mass, model.statics, model.damping
    lines = [f"{model.vehicle.name}  ({model.vehicle.category})", ""]

    def row(label: str, param: Parameter, diag: bool = False) -> None:
        value, sigma = param.value, param.sigma
        if diag:
            value, sigma = np.diag(value), np.diag(sigma)
        if np.ndim(value) == 0:
            lines.append(f"  {label:<16}{param.format()}")
            return
        vals = ", ".join(f"{x:.4g}" for x in _clean(value))
        sig = ", ".join(f"{x:.2g}" for x in np.ravel(sigma))
        lines.append(f"  {label:<16}[{vals}] ± [{sig}] {param.unit} [{param.source.value}]")

    lines.append("Mass and statics")
    row("mass", m.mass)
    row("volume", s.volume)
    row("weight W", s.weight)
    row("buoyancy B", s.buoyancy)
    row("CG r_g", m.r_g)
    row("CB r_b", s.r_b)
    row("I_g diag", m.inertia_g, diag=True)
    lines += ["", "Hydrodynamics (order: surge, sway, heave, roll, pitch, yaw)"]
    row("M_A diag", model.m_a, diag=True)
    row("D linear", d.linear)
    row("D quadratic", d.quadratic)
    lines.append(f"  {'surge C_D':<16}{d.axial_cd:.3g} (on frontal area)")

    if model.vehicle.fins:
        lines += ["", "Fins (lift grows with speed: force = D_fin x |u| x velocity)"]
        for fin, cla in zip(model.vehicle.fins, model.fins.lift_slopes, strict=True):
            kind = "control" if fin.control else "fixed"
            lines.append(
                f"  {fin.name:<12} {kind:<8} S = {fin.area:g} m^2, C_L_alpha = {cla:.3g} /rad"
            )

    if model.vehicle.thrusters:
        lines += ["", "Thrusters"]
        for i, t in enumerate(model.vehicle.thrusters):
            col = ", ".join(f"{x:.3g}" for x in _clean(model.b_thr[:, i]))
            lines.append(
                f"  {i}: {t.model:<8} +{t.max_forward:g} / -{t.max_reverse:g} N  B column [{col}]"
            )

    params = (m.r_g, s.volume, model.m_a, d.linear, d.quadratic, model.body_lift)
    notes = {p.note for p in params if p is not None and p.note}
    if notes:
        lines += ["", "Notes:"] + [f"  - {n}" for n in sorted(notes)]
    if model.warnings:
        lines += ["", "Warnings:"] + [f"  ! {w}" for w in model.warnings]
    return "\n".join(lines)
