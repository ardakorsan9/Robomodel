"""What the robot NEEDS: thrust and torque computed from the model, for builders who do not
know their actuators yet ("I don't know" is a valid answer; this module turns it into numbers).

Every entry is {"unit", "low", "nominal", "high"}: "nominal" from the CAD model, "low" and "high"
the smallest and largest need over the model range (damping, rolling resistance), so a motor
sized for "high" still works if the CAD estimate was too optimistic.

* underwater and surface: total forward thrust to hold 0.5, 1 and 2 m/s, F = D_lin v + D_quad v^2
  (surge), and, underwater, the vertical thrust that cancels net buoyancy;
* aerial: hover thrust (= weight), recommended total maximum (2 x weight) and per rotor;
* ground: per driven wheel, the torque to climb a 20 % slope and to accelerate at 1 m/s^2,
  and the tip-over-limited acceleration.
"""

from __future__ import annotations

import numpy as np

from robomodel.hydrostatics import GRAVITY

HOLD_SPEEDS = (0.5, 1.0, 2.0)  # m/s
SLOPE_GRADE = 0.20  # 20 % = rise 0.2 per 1.0 run
ACCELERATION = 1.0  # m/s^2
MAX_THRUST_TO_WEIGHT = 2.0  # recommended total maximum thrust of a drone
DEFAULT_ROTORS = 4


def _entry(unit: str, values: dict[str, float], **extra) -> dict:
    nominal = float(values["nominal"])
    allv = [float(v) for v in values.values()]
    return {
        "unit": unit,
        "low": min(allv),
        "nominal": nominal,
        "high": max(allv),
        **extra,
    }


def _speed_key(v: float) -> str:
    return "thrust_" + f"{v:g}".replace(".", "_") + "_ms"


def requirements(model, n_rotors: int | None = None) -> dict:
    """Plain numbers for a VehicleModel (underwater, surface, aerial) or a GroundModel."""
    from robomodel.ground import GroundModel

    if isinstance(model, GroundModel):
        return _ground(model)
    if model.vehicle.category == "aerial":
        return _aerial(model, n_rotors)
    return _marine(model)


def _marine(model) -> dict:
    from robomodel.model_range import model_range

    variants = {label: v.dynamics() for label, v in model_range(model).items()}
    items: dict[str, dict] = {}
    for speed in HOLD_SPEEDS:
        forces = {
            label: float(dyn.d_lin[0] * speed + dyn.d_quad[0] * speed**2)
            for label, dyn in variants.items()
        }
        items[_speed_key(speed)] = _entry("N", forces, speed=speed)
    notes = []
    dyn = variants["nominal"]
    if model.vehicle.category == "underwater":
        net = float(dyn.weight - dyn.buoyancy)  # > 0: sinks, thrust must push up
        items["vertical_thrust"] = _entry(
            "N",
            {"nominal": abs(net)},
            direction="up" if net > 1e-9 else ("down" if net < -1e-9 else "none"),
        )
    else:
        notes.append("surface_froude")
    return {"category": model.vehicle.category, "items": items, "notes": notes}


def _aerial(model, n_rotors: int | None) -> dict:
    weight = float(model.mass.mass.value) * GRAVITY
    given = len(model.vehicle.thrusters)
    n = n_rotors or given or DEFAULT_ROTORS
    total_max = MAX_THRUST_TO_WEIGHT * weight
    items = {
        "hover_thrust": _entry("N", {"nominal": weight}),
        "max_thrust_total": _entry("N", {"nominal": total_max}),
        "hover_per_rotor": _entry("N", {"nominal": weight / n}, rotors=n),
        "max_thrust_per_rotor": _entry("N", {"nominal": total_max / n}, rotors=n),
    }
    notes = [] if given or n_rotors else ["rotors_assumed"]
    return {"category": "aerial", "items": items, "notes": notes}


def _ground(gm) -> dict:
    from robomodel.ground import model_range, tip_limits

    wheels = [w for w in gm.vehicle.wheels if not w.caster]
    driven = [w for w in wheels if w.driven]
    used = driven or wheels
    n = max(len(used), 1)
    radius = float(np.mean([w.radius for w in used])) if used else 0.0
    theta = np.arctan(SLOPE_GRADE)
    slope, accel = {}, {}
    for label, variant in model_range(gm).items():
        # force along the ground that the wheels must supply: weight component down the
        # slope plus rolling resistance (normal load W cos(theta)); or m a plus rolling.
        slope[label] = gm.weight * (np.sin(theta) + variant.rolling * np.cos(theta))
        accel[label] = gm.m * ACCELERATION + variant.rolling * gm.weight
    items = {
        "slope_force": _entry("N", {k: v / n for k, v in slope.items()}, wheels=n),
        "accel_force": _entry("N", {k: v / n for k, v in accel.items()}, wheels=n),
    }
    if radius > 0:
        items["slope_torque"] = _entry(
            "N m", {k: v / n * radius for k, v in slope.items()}, wheels=n, radius=radius
        )
        items["accel_torque"] = _entry(
            "N m", {k: v / n * radius for k, v in accel.items()}, wheels=n, radius=radius
        )
    tips = tip_limits(gm)
    out_tips = {
        side: float(GRAVITY * np.tan(np.radians(angle)))
        for side, angle in tips.items()
        if side in ("front", "rear")
    }
    if out_tips:
        items["tip_acceleration"] = _entry(
            "m/s^2", {"nominal": min(out_tips.values())}, sides=out_tips
        )
    notes = ["wheels_driven" if driven else "wheels_all"]
    return {"category": "ground", "items": items, "notes": notes}


def format_requirements(req: dict) -> str:
    """Plain-text block for report.txt (English; the web page translates the cards itself)."""
    labels = {
        "vertical_thrust": "vertical thrust to cancel net buoyancy",
        "hover_thrust": "total thrust to hover (= weight)",
        "max_thrust_total": "recommended total maximum thrust (2 x weight)",
        "hover_per_rotor": "hover thrust per rotor",
        "max_thrust_per_rotor": "recommended maximum thrust per rotor",
        "slope_force": "force per wheel at the ground, 20 % slope",
        "accel_force": "force per wheel at the ground, 1 m/s^2",
        "slope_torque": "torque per wheel, 20 % slope",
        "accel_torque": "torque per wheel, 1 m/s^2",
        "tip_acceleration": "tip-over limit of the acceleration",
    }
    lines = ["What the robot needs (low / nominal / high):"]
    for key, e in req["items"].items():
        if key.startswith("thrust_"):
            label = f"total forward thrust to hold {e['speed']:g} m/s"
        else:
            label = labels.get(key, key)
        extra = f" ({e['direction']})" if e.get("direction") not in (None, "none") else ""
        lines.append(
            f"  {label:<50}{e['low']:.3g} / {e['nominal']:.3g} / {e['high']:.3g} {e['unit']}{extra}"
        )
    if "rotors_assumed" in req["notes"]:
        lines.append(f"  (no rotors given: per-rotor values assume {DEFAULT_ROTORS} rotors)")
    if "wheels_all" in req["notes"]:
        lines.append(
            "  (no wheel marked as driven: per-wheel values share the load over all wheels)"
        )
    return "\n".join(lines)
