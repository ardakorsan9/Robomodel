"""Virtual pool test: can calibration recover REMUS's real damping from motion data?

"Truth" is REMUS as Prestero tuned it to sea trials (Appendix C), written as scale factors
on robomodel's CAD values. We simulate test runs of that truth with sensor noise, calibrate
the CAD-only model on two runs, and check it on a third run it has not seen, and on the
sea-trial turn rate (about 10 deg/s at 4 deg rudder, Prestero Sec. 7.5.1).

Run:  uv run python validation/calibration_remus.py   (writes docs/dogrulama_kalibrasyon.md)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import remus100  # noqa: E402

from robomodel.calibration import (  # noqa: E402
    DEFAULT_NOISE,
    DOF,
    Log,
    calibrate,
    fit_percent,
    simulate_log,
    summary,
)
from robomodel.model import apply_calibration, build_model  # noqa: E402
from robomodel.vehicle import Calibration, load_vehicle  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
REPORT = ROOT / "docs" / "dogrulama_kalibrasyon.md"
MEASURED = ("z", "phi", "theta", "psi", "u", "v", "w", "p", "q", "r")  # DVL + IMU + depth
RUDDERS = ("rudder_top", "rudder_bottom")
PLANES = ("plane_port", "plane_stbd")


def truth_scales(cad) -> Calibration:
    """Prestero's sea-tuned REMUS (Tables C.1-C.4, C.10) relative to our CAD values."""
    dq = np.asarray(cad.damping.quadratic.value)
    tuned = np.array([0.5 * 1030 * 0.20 * 2.85e-2, 1310, 1310, dq[3], 117.5, 94.0])
    damping = tuned / np.where(dq > 0, dq, 1)
    damping[3] = 1.0  # roll: no usable reference (ours is a floor value)
    fin = 9.64 / float(cad.fins.d_fin.value[1, 1])
    body = 28.6 / float(cad.body_lift.value[1, 1])
    return Calibration(damping=damping, fin_lift=fin, body_lift=body, note="truth")


def make_log(truth, name, duration, rudder, planes, rng, dt=0.1) -> Log:
    """Simulate a test run of the truth at cruise speed and add sensor noise."""
    t = np.arange(0.0, duration + dt / 2, dt)
    dyn = truth.dynamics()
    surge = dyn.d_lin[0] * remus100.SPEED + dyn.d_quad[0] * remus100.SPEED**2  # propeller
    inputs = {"tau_x": np.full_like(t, surge)}
    for fin in RUDDERS:
        inputs[f"fin_{fin}"] = rudder(t)
    for fin in PLANES:
        inputs[f"fin_{fin}"] = planes(t)
    states0 = {"u": np.full_like(t, remus100.SPEED)}
    clean = simulate_log(truth, Log(t, inputs, states0, name), rtol=1e-8)
    index = ("x", "y", "z", "phi", "theta", "psi", "u", "v", "w", "p", "q", "r")
    states = {
        s: clean[:, index.index(s)] + rng.normal(0, DEFAULT_NOISE[s], len(t)) for s in MEASURED
    }
    states["u"][0] = remus100.SPEED  # start the fit from the known cruise speed
    for s in MEASURED:
        if s != "u":
            states[s][0] = 0.0
    return Log(t, inputs, states, name)


def square(amplitude_deg, period, start=2.0):
    a = np.radians(amplitude_deg)
    return lambda t: np.where(
        t < start, 0.0, a * np.sign(np.sin(2 * np.pi * (t - start) / period) + 1e-9)
    )


def steps(pattern_deg, step):
    values = np.radians(np.asarray(pattern_deg, float))
    return lambda t: values[np.minimum((t // step).astype(int), len(values) - 1)]


def run(duration=40.0, seed=1):
    rng = np.random.default_rng(seed)
    cad = build_model(load_vehicle(remus100.VEHICLE))
    truth_cal = truth_scales(cad)
    truth = apply_calibration(cad, truth_cal)
    zero = steps([0], duration)
    logs = [
        make_log(truth, "zigzag", duration, square(8, 12), zero, rng),
        make_log(truth, "dalis", duration, zero, steps([0, 6, -6, 0, -4, 4, 0], 6), rng),
    ]
    check = make_log(
        truth,
        "dogrulama",
        duration,
        steps([0, 4, 4, -6, -6, 0], 7),
        steps([0, -3, 3, 0, 5, -5], 7),
        rng,
    )
    result = calibrate(cad, logs, validation=[check])
    ceiling = fit_percent(truth, check)  # the best any model can do on noisy data
    return cad, truth, truth_cal, result, ceiling


def report(cad, truth, truth_cal, result, ceiling) -> str:
    cal = result.calibration
    turn_cad = remus100.steady_turn(cad)
    turn_cal = remus100.steady_turn(result.model)
    turn_truth = remus100.steady_turn(truth)

    def pct(a, b):
        return f"{100 * (a / b - 1):+.0f}%"

    lines = [
        "# Kalibrasyon doğrulaması: sanal havuz (REMUS 100)",
        "",
        "Bu dosya `validation/calibration_remus.py` ile üretilir, elle düzenleme.",
        "",
        "**Soru:** Araç hareket kayıtlarından, CAD'in yanlış tahmin ettiği sönümü geri bulabilir miyiz?",
        "",
        "- **Gerçek araç (sanal):** REMUS 100, Prestero'nun deniz testlerine göre ayarladığı",
        "  katsayılarla (Ek C). robomodel'in CAD değerlerine göre çarpan olarak yazıldı.",
        "- **Ölçüm:** bu araçla iki test (zikzak dümen, dalış dümeni), 40 s, 10 Hz; derinlik,",
        "  açılar, hızlar ve açısal hızlara tipik sensör gürültüsü eklendi.",
        "- **Kalibrasyon:** sadece CAD'den çıkan model, bu iki kayda uyduruldu.",
        "- **Kontrol:** kalibrasyonda kullanılmayan üçüncü bir kayıt ve deniz testindeki dönüş hızı.",
        "",
        "## Bulunan çarpanlar",
        "",
        "| Terim | Gerçek | Kalibrasyonla bulunan | Fark | Belirsizlik (1σ) |",
        "|---|---|---|---|---|",
    ]
    for i, dof in enumerate(DOF):
        if i == 3:
            continue  # roll is not excited by these runs
        lines.append(
            f"| Sönüm, {dof} | ×{truth_cal.damping[i]:.3g} | ×{cal.damping[i]:.3g} | "
            f"{pct(cal.damping[i], truth_cal.damping[i])} | ±{cal.damping_sigma[i]:.0%} |"
        )
    lines.append(
        f"| Kanat kaldırması | ×{truth_cal.fin_lift:.3g} | ×{cal.fin_lift:.3g} | "
        f"{pct(cal.fin_lift, truth_cal.fin_lift)} | ±{cal.fin_lift_sigma:.0%} |"
    )
    lines.append(
        f"| Gövde kaldırması | ×{truth_cal.body_lift:.3g} | ×{cal.body_lift:.3g} | "
        f"{pct(cal.body_lift, truth_cal.body_lift)} | ±{cal.body_lift_sigma:.0%} |"
    )
    lines += [
        "",
        "Yuvarlanma sönümü bu testlerde uyarılmadığı için CAD değerinde kaldı (tabloda yok).",
        "",
        "## Kalibrasyonda kullanılmayan kayıtta uyum (%, 100 = birebir)",
        "",
        "| Kanal | Sadece CAD | Kalibrasyon sonrası | Tavan (gerçek aracın kendisi) |",
        "|---|---|---|---|",
    ]
    before = result.validation_before["dogrulama"]
    after = result.validation_after["dogrulama"]
    for ch in before:
        lines.append(f"| `{ch}` | {before[ch]:.0f} | {after[ch]:.0f} | {ceiling[ch]:.0f} |")
    key = ("psi", "r", "z", "theta")
    mean_before = np.mean([before[c] for c in key])
    mean_after = np.mean([after[c] for c in key])
    mean_ceiling = np.mean([ceiling[c] for c in key])
    lines += [
        f"| **ortalama (psi, r, z, theta)** | **{mean_before:.0f}** | **{mean_after:.0f}** | "
        f"**{mean_ceiling:.0f}** |",
        "",
        "Tavan: gerçek aracın kendi modeli bile gürültülü ölçüme bundan iyi uyamaz. Hareket",
        "küçük olan kanallarda (`phi`, `p`, `v`, `w`) sinyal gürültüden küçük olduğu için yüzde",
        "düşük çıkar; bu kanallar ortalamaya alınmadı.",
        "",
        "## Dönüş testi: 4° dümen, 1,54 m/s",
        "",
        "| Model | Dönüş hızı (°/s) | Sanal gerçek araca göre |",
        "|---|---|---|",
        f"| Sanal gerçek araç | {turn_truth:.1f} | – |",
        f"| Sadece CAD | {turn_cad:.1f} | {pct(turn_cad, turn_truth)} |",
        f"| Kalibre edilmiş | {turn_cal:.1f} | {pct(turn_cal, turn_truth)} |",
        "",
        "**Bu test neyi gösterir, neyi göstermez:** kalibrasyon algoritmasının, bilinen bir",
        "gerçeği gürültülü hareket verisinden geri bulduğunu gösterir. Gerçek REMUS'u göstermez:",
        f"Prestero'nun ayarlı katsayıları bizim denklemlerimizde {turn_truth:.1f}°/s veriyor,",
        "denizde ölçülen ~10°/s değil (onun modelinde bizde olmayan çapraz terimler var ve",
        "kanat tablosu kendi formülüyle tutarsız). Gerçek araçta kalibrasyon havuz verisiyle",
        "yapılmalı; o zaman model yapısının kendi eksikleri de sonuca girer.",
        "",
        "```",
        summary(result),
        "```",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    text = report(*run())
    REPORT.write_text(text, encoding="utf-8")
    print(text)
    print(f"written: {REPORT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
