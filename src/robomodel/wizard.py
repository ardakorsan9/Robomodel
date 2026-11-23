"""Measurement wizard (O4): which measurement would improve the model the most?

Each uncertain parameter is moved by its own 1-sigma uncertainty, and we look at how much
the vehicle's behaviour changes. Behaviour = a few numbers a user cares about: top speed,
turn rate, roll and pitch periods, net buoyancy. Parameters are ranked by their effect and
each comes with the cheapest test that pins it down. Already measured or calibrated values
have small sigmas, so they drop to the bottom by themselves.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from robomodel.dynamics import Dynamics, simulate
from robomodel.model import VehicleModel
from robomodel.thrusters import thrust, thrust_rel_sigma

DOF_TR = ("ileri", "yan", "dikey", "yuvarlanma", "yunuslama", "sapma")
POOL_TEST = {
    0: "1_ileri_ve_sonumlenme",
    1: "2_zikzak",
    2: "3_dalis",
    3: "4_serbest_salinim",
    4: "3_dalis",
    5: "2_zikzak",
}
TURN_FIN_ANGLE = np.radians(5.0)


@dataclass(frozen=True)
class Effect:
    parameter: str
    rel_sigma: float  # the parameter's own uncertainty
    changes: dict[str, float]  # metric -> relative change for +1 sigma
    test: str  # how to measure it (Turkish)
    sigma_text: str = ""  # how to show the uncertainty, if not as a percentage

    @property
    def worst(self) -> float:
        return max((abs(v) for v in self.changes.values()), default=0.0)


def _metrics(model: VehicleModel, dyn: Dynamics, thrust_scale: float = 1.0) -> dict[str, float]:
    v = model.vehicle
    out = {}
    forward = [i for i, t in enumerate(v.thrusters) if abs(t.direction[0]) > 0.7]
    if forward:
        commands = np.zeros(len(v.thrusters))
        commands[forward] = np.sign([v.thrusters[i].direction[0] for i in forward])
        force = float((dyn.b_thr @ thrust(v.thrusters, commands))[0]) * thrust_scale
        a, b = dyn.d_quad[0], dyn.d_lin[0]
        out["son hız"] = (-b + np.sqrt(b * b + 4 * a * force)) / (2 * a) if a > 0 else force / b
    rudders = [
        n
        for n, f in zip(model.fins.control_names, [f for f in v.fins if f.control], strict=True)
        if abs(f.normal[1]) > 0.7
    ]
    if rudders:
        speed = out.get("son hız", 1.0) * 0.8
        deflection = np.array(
            [TURN_FIN_ANGLE if n in rudders else 0.0 for n in model.fins.control_names]
        )
        hold = dyn.d_lin[0] * speed + dyn.d_quad[0] * speed**2

        def tau(_t, x):
            return np.array([hold + 50.0 * (speed - x[6]), 0, 0, 0, 0, 0])

        state0 = dyn.rest_state.copy()
        state0[6] = speed
        traj = simulate(
            dyn,
            30.0,
            tau=tau,
            state0=state0,
            fin_deflection=deflection,
            dt_out=0.5,
            rtol=1e-6,
            atol=1e-8,
        )
        out["dönüş hızı (5° dümen)"] = float(abs(traj.nu[-1, 5]))
    elif v.thrusters:
        # ROV-style: turn and dive with thrusters at half command (steady state)
        axes = ((5, "dönüş hızı (%50 itici)"),)
        if dyn.g_matrix is None:  # a floating boat does not dive
            axes += ((2, "dikey hız (%50 itici)"),)
        for axis, name in axes:
            cols = [
                i
                for i in range(dyn.b_thr.shape[1])
                if abs(dyn.b_thr[axis, i]) > 1e-6 and i not in forward
            ]
            if not cols and axis == 5:
                cols = [i for i in forward if abs(dyn.b_thr[axis, i]) > 1e-6]
            if not cols:
                continue
            commands = np.zeros(len(v.thrusters))
            commands[cols] = 0.5 * np.sign(dyn.b_thr[axis, cols])
            force = float((dyn.b_thr @ thrust(v.thrusters, commands))[axis]) * thrust_scale
            if axis == 2:
                force += dyn.weight - dyn.buoyancy  # net weight helps or fights the dive
            a, b = dyn.d_quad[axis], dyn.d_lin[axis]
            sign = np.sign(force)
            force = abs(force)
            speed = (-b + np.sqrt(b * b + 4 * a * force)) / (2 * a) if a > 0 else force / b
            out[name] = float(sign * speed) if axis == 2 else float(speed)
    if dyn.g_matrix is not None:  # surface vessel: periods from the hydrostatic springs
        from robomodel.surface import natural_periods

        periods = natural_periods(dyn.mass_matrix, dyn.g_matrix)
        for key, name in (
            ("heave", "dalga periyodu (dikey)"),
            ("roll", "yuvarlanma periyodu"),
            ("pitch", "yunuslama periyodu"),
        ):
            if np.isfinite(periods[key]):
                out[name] = periods[key]
        return out
    bg = dyn.r_g[2] - dyn.r_b[2]  # z down: CG below CB means bg > 0
    if bg > 0:
        m = dyn.mass_matrix
        out["yuvarlanma periyodu"] = 2 * np.pi * np.sqrt(m[3, 3] / (bg * dyn.weight))
        out["yunuslama periyodu"] = 2 * np.pi * np.sqrt(m[4, 4] / (bg * dyn.weight))
    out["net kaldırma"] = dyn.buoyancy - dyn.weight
    return out


def _relative(after: dict, before: dict) -> dict[str, float]:
    out = {}
    for k, b in before.items():
        if k.startswith("_"):
            continue
        if k == "net kaldırma":
            scale = max(abs(b), 0.02 * before.get("_weight", 1.0), 1e-9)
            out[k] = (after[k] - b) / scale
        else:
            out[k] = (after[k] - b) / b if abs(b) > 1e-12 else 0.0
    return out


def analyse(model: VehicleModel) -> tuple[dict[str, float], list[Effect]]:
    dyn = model.dynamics()
    base = _metrics(model, dyn)
    base_w = {**base, "_weight": dyn.weight}

    def effect(name, rel, changed: Dynamics, test, thrust_scale=1.0, sigma_text=""):
        after = _metrics(model, changed, thrust_scale)
        return Effect(name, rel, _relative(after, base_w), test, sigma_text)

    effects = []
    s = model.statics
    w_rel = float(s.weight.sigma / s.weight.value)
    effects.append(
        effect(
            "ağırlık",
            w_rel,
            replace(dyn, weight=dyn.weight * (1 + w_rel)),
            "Aracı tart (banyo tartısı), 5 dk",
        )
    )
    b_rel = float(s.buoyancy.sigma / s.buoyancy.value)
    effects.append(
        effect(
            "kaldırma kuvveti (hacim)",
            b_rel,
            replace(dyn, buoyancy=dyn.buoyancy * (1 + b_rel)),
            "Suda tart: el kantarıyla suya sarkıt, B = W - okunan, 10 dk",
        )
    )
    zg_sigma = float(model.mass.r_g.sigma[2])
    r_g = dyn.r_g.copy()
    r_g[2] += zg_sigma
    scale = max(abs(dyn.r_g[2] - dyn.r_b[2]), 1e-3)
    effects.append(
        effect(
            "kütle merkezi yüksekliği",
            zg_sigma / scale,
            replace(dyn, r_g=r_g),
            "Denge testi: iki noktadan asıp ip doğrultularını çiz, 20 dk",
            sigma_text=f"±{100 * zg_sigma:.1f} cm",
        )
    )
    inertia = model.mass.inertia_o
    for i, axis in ((3, "x"), (4, "y"), (5, "z")):
        rel = float(inertia.sigma[i - 3, i - 3] / inertia.value[i - 3, i - 3])
        m_rb = dyn.m_rb.copy()
        m_rb[i, i] *= 1 + rel
        effects.append(
            effect(
                f"eylemsizlik I{axis}{axis}",
                rel,
                replace(dyn, m_rb=m_rb),
                "İki ipli sarkaç (bifilar): salınım süresini ölç, 30 dk",
            )
        )
    ma = model.m_a
    for i in range(6):
        if ma.value[i, i] <= 0:
            continue
        rel = float(ma.sigma[i, i] / ma.value[i, i])
        m_a = dyn.m_a.copy()
        m_a[i, i] *= 1 + rel
        effects.append(
            effect(
                f"eklenmiş kütle, {DOF_TR[i]}",
                rel,
                replace(dyn, m_a=m_a),
                f"Havuz: {POOL_TEST[i]} (hızlanma kısmı)",
            )
        )
    d = model.damping
    for i in range(6):
        if d.quadratic.value[i] <= 0 and d.linear.value[i] <= 0:
            continue
        rel = float(
            max(
                d.quadratic.sigma[i] / max(d.quadratic.value[i], 1e-12)
                if d.quadratic.value[i] > 0
                else 0,
                d.linear.sigma[i] / max(d.linear.value[i], 1e-12),
            )
        )
        factor = np.ones(6)
        factor[i] = 1 + rel
        effects.append(
            effect(
                f"sönüm, {DOF_TR[i]}",
                rel,
                replace(dyn, d_lin=dyn.d_lin * factor, d_quad=dyn.d_quad * factor),
                f"Havuz: {POOL_TEST[i]}, sonra robomodel calibrate",
            )
        )
    if model.vehicle.fins:
        rel = float(np.max(model.fins.d_fin.sigma) / max(np.max(model.fins.d_fin.value), 1e-12))
        fin_d = np.asarray(model.fins.d_fin.value)
        effects.append(
            effect(
                "kanat kaldırması",
                rel,
                replace(dyn, d_fin=dyn.d_fin + rel * fin_d, b_fin=dyn.b_fin * (1 + rel)),
                "Havuz: 2_zikzak, sonra robomodel calibrate",
            )
        )
    if model.body_lift is not None:
        rel = float(
            np.max(model.body_lift.sigma) / max(np.max(np.abs(model.body_lift.value)), 1e-12)
        )
        effects.append(
            effect(
                "gövde kaldırması",
                rel,
                replace(dyn, d_fin=dyn.d_fin + rel * np.asarray(model.body_lift.value)),
                "Havuz: 2_zikzak, sonra robomodel calibrate",
            )
        )
    if model.vehicle.thrusters:
        rel = max(thrust_rel_sigma(t) for t in model.vehicle.thrusters)
        effects.append(
            effect(
                "itki",
                rel,
                dyn,
                "Çekme testi: iticiyi el kantarına bağla, komuta karşı ölç, 20 dk",
                thrust_scale=1 + rel,
            )
        )
    effects.sort(key=lambda e: e.worst, reverse=True)
    return base, effects


def accuracy(effects: list[Effect]) -> dict[str, float]:
    """Expected 1-sigma relative error of each metric (root sum of squares)."""
    metrics = {k for e in effects for k in e.changes}
    return {k: float(np.sqrt(sum(e.changes.get(k, 0.0) ** 2 for e in effects))) for k in metrics}


UNITS = {
    "son hız": ("m/s", 1.0),
    "dönüş hızı (5° dümen)": ("°/s", np.degrees(1.0)),
    "dönüş hızı (%50 itici)": ("°/s", np.degrees(1.0)),
    "dikey hız (%50 itici)": ("m/s", 1.0),
    "yuvarlanma periyodu": ("s", 1.0),
    "yunuslama periyodu": ("s", 1.0),
    "net kaldırma": ("N", 1.0),
}


def report(model: VehicleModel, top: int = 8) -> str:
    base, effects = analyse(model)
    acc = accuracy(effects)
    lines = [
        f"Ölçüm sihirbazı: {model.vehicle.name}",
        "",
        "Davranış tahmini ve beklenen hata (1σ):",
    ]
    for k, value in base.items():
        unit, factor = UNITS[k]
        err = acc.get(k, 0.0)
        extra = "" if k != "net kaldırma" else " (fark, ağırlığın %2'sine göre)"
        lines.append(f"  {k:<22} {value * factor:8.3g} {unit:<4} ± %{100 * err:.0f}{extra}")
    lines += ["", "Önce bunları ölç (etkisi büyükten küçüğe):"]
    for n, e in enumerate([e for e in effects if e.worst > 0.01][:top], 1):
        worst_metric = max(e.changes, key=lambda k: abs(e.changes[k]))
        sigma = e.sigma_text or f"±%{100 * e.rel_sigma:.0f}"
        lines.append(
            f"  {n}. {e.parameter:<28} belirsizliği {sigma} -> "
            f"{worst_metric} ±%{100 * abs(e.changes[worst_metric]):.0f}"
        )
        lines.append(f"     nasıl: {e.test}")
    small = [e.parameter for e in effects if e.worst <= 0.01]
    if small:
        lines += ["", "Etkisi %1'in altında (şimdilik ölçmeye gerek yok): " + ", ".join(small)]
    return "\n".join(lines)
