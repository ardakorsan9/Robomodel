"""Compare robomodel's REMUS 100 model with Prestero (2001).

Reference: T. Prestero, "Verification of a Six-Degree of Freedom Simulation Model for the
REMUS Autonomous Underwater Vehicle", MIT/WHOI MSc thesis, 2001.
https://dspace.mit.edu/handle/1721.1/65068

Prestero gives two sets of values:
  theory   - his own handbook/strip-theory estimates (Tables 4.2, 4.3), the same kind of
             estimate robomodel makes, so this checks our method;
  adjusted - after tuning some terms to sea trials (Table 8.2, Appendix C);
  measured - tow tank, scale or sea data (Tables 2.3, 2.4, Section 7.5).

Run:  uv run python validation/remus100.py   (writes docs/dogrulama_remus100.md)
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from robomodel.dynamics import simulate
from robomodel.model import VehicleModel, build_model
from robomodel.vehicle import load_vehicle

ROOT = Path(__file__).resolve().parent.parent
VEHICLE = ROOT / "examples" / "remus100.yaml"
REPORT = ROOT / "docs" / "dogrulama_remus100.md"

SPEED = 1.54  # m/s, REMUS cruise speed in the sea trials (Prestero Table 8.1)
RUDDER = np.radians(4.0)  # Section 7.5.1
FIELD_YAW_RATE = 10.0  # deg/s, "approximately 10 degrees/second" (Section 7.5.1)
CB_FROM_NOSE_MEASURED = 0.611  # m, Table 2.4


@dataclass(frozen=True)
class Row:
    group: str
    name: str  # Prestero / Fossen symbol
    ours: float | None
    theory: float | None
    adjusted: float | None
    source: str
    note: str = ""

    def error(self, ref: float | None) -> float | None:
        """Relative difference in magnitude: +10% means ours is 10% larger."""
        if self.ours is None or ref in (None, 0):
            return None
        return abs(self.ours) / abs(ref) - 1

    def sign_ok(self, ref: float | None) -> bool:
        return (
            self.ours is None
            or ref in (None, 0)
            or self.ours == 0
            or (np.sign(self.ours) == np.sign(ref))
        )


def _fin_cols(model: VehicleModel, prefix: str) -> np.ndarray:
    names = model.fins.control_names
    return model.fins.b_fin[:, [i for i, n in enumerate(names) if n.startswith(prefix)]]


def comparison(model: VehicleModel) -> list[Row]:
    ma = np.asarray(model.m_a.value)
    dq = np.asarray(model.damping.quadratic.value)
    df = np.asarray(model.fins.d_fin.value)
    rudder = _fin_cols(model, "rudder").sum(axis=1)
    planes = -_fin_cols(model, "plane").sum(axis=1)  # Prestero's delta_s has the other sign
    s = model.statics
    cb_from_nose = float(model.hull.bounds[1, 0])  # origin is at our CB, nose at +x
    # Fin lift from Prestero's own formula (eq. 4.44-4.45) and his parameters (Table 2.2):
    # rho * CLa * S_fin, with rho = 1030, CLa = 3.12, S_fin = 6.65e-3, x_fin = -0.638.
    k = 1030 * 3.12 * 6.65e-3
    x_fin = -0.638
    fin_note = "teori sütunu: tezin formülü (denk. 4.44-4.45); tablo bu formülden 2,2 kat küçük, tezde nedeni açıklanmıyor"

    return [
        # Hydrostatics: Tables 2.3, 2.4, 2.7
        Row(
            "Hidrostatik",
            "V (m^3)",
            float(s.volume.value),
            3.15e-2,
            None,
            "Tablo 2.7",
            "Prestero'nun gövde hacmi tahmini",
        ),
        Row(
            "Hidrostatik",
            "B (N)",
            float(s.buoyancy.value),
            3.17e2,
            3.06e2,
            "Tablo 2.3, 2.7",
            "ayarlı sütunu = ölçülen kaldırma; kanat, sonar ve su dolan burun hesapta yok",
        ),
        Row(
            "Hidrostatik",
            "x_cb burundan (m)",
            cb_from_nose,
            None,
            CB_FROM_NOSE_MEASURED,
            "Tablo 2.4",
            "ayarlı sütunu = ölçülen",
        ),
        # Added mass: Table 4.2/4.3 (no adjustment except K_pdot)
        Row("Eklenmiş kütle", "X_udot", -ma[0, 0], -9.30e-1, -9.30e-1, "Tablo 4.2, C.5"),
        Row("Eklenmiş kütle", "Y_vdot", -ma[1, 1], -3.55e1, -3.55e1, "Tablo 4.2, C.5"),
        Row("Eklenmiş kütle", "Z_wdot", -ma[2, 2], -3.55e1, -3.55e1, "Tablo 4.2, C.5"),
        Row(
            "Eklenmiş kütle",
            "K_pdot",
            -ma[3, 3],
            -1.41e-2,
            -7.04e-2,
            "Tablo 4.3, C.5",
            "Prestero denize göre 5 kat büyüttü",
        ),
        Row("Eklenmiş kütle", "M_qdot", -ma[4, 4], -4.88, -4.88, "Tablo 4.3, C.5"),
        Row("Eklenmiş kütle", "N_rdot", -ma[5, 5], -4.88, -4.88, "Tablo 4.3, C.5"),
        Row("Eklenmiş kütle", "Y_rdot = N_vdot", -ma[1, 5], 1.93, 1.93, "Tablo 4.2, C.5"),
        Row("Eklenmiş kütle", "Z_qdot = M_wdot", -ma[2, 4], -1.93, -1.93, "Tablo 4.2, C.5"),
        # Damping: Tables 4.2/4.3 and C.1-C.3, adjustments Table 8.2
        Row(
            "Sönümleme",
            "eksenel C_d",
            model.damping.axial_cd,
            0.12,
            0.20,
            "Bölüm 4.2.1",
            "teori: el kitabı formülleri 0,11-0,13; ayarlı sütunu: denizde ölçülen 0,2 "
            "(havuzda 0,27, kanat ve sonar dahil)",
        ),
        Row(
            "Sönümleme",
            "X_u|u|",
            -dq[0],
            -1.62,
            -0.5 * 1030 * 0.20 * 2.85e-2,
            "Tablo 4.2, C.1",
            "tablo C_d ≈ 0,11'e karşılık geliyor; ayarlı sütunu: C_d = 0,2",
        ),
        Row(
            "Sönümleme",
            "Y_v|v|",
            -dq[1],
            -1.31e2,
            -1.31e3,
            "Tablo 4.2, C.2",
            "Prestero denize göre 10 kat büyüttü (Tablo 8.2)",
        ),
        Row(
            "Sönümleme",
            "Z_w|w|",
            -dq[2],
            -1.31e2,
            -1.31e3,
            "Tablo 4.2, 8.2",
            "Tablo C.2'de -131 yazıyor, Tablo 8.2'ye göre 10 kat olmalı: tezde tutarsızlık",
        ),
        Row(
            "Sönümleme",
            "K_p|p|",
            -dq[3],
            -1.30e-3,
            -1.30e-1,
            "Tablo 4.3, C.3",
            "bizde dairesel gövdede 0; kanatların çapraz akış sürüklemesi yok",
        ),
        Row(
            "Sönümleme",
            "M_q|q|",
            -dq[4],
            -9.40,
            -1.175e2,
            "Tablo 4.3, 8.2",
            "Prestero 12,5 kat büyüttü; Tablo C.2'de -188 yazıyor",
        ),
        Row(
            "Sönümleme",
            "N_r|r|",
            -dq[5],
            -9.40,
            -9.40e1,
            "Tablo 4.3, C.2",
            "Prestero 10 kat büyüttü",
        ),
        # Fin lift: Table C.10 (not adjusted). Prestero's eq. 4.44 gives rho*CLa*S = 21.4,
        # the table has 9.64: the thesis is inconsistent here.
        Row("Kanat kaldırması", "Y_uvf", -df[1, 1], -k, -9.64, "denk. 4.44, Tablo C.10", fin_note),
        Row("Kanat kaldırması", "Y_urf", -df[1, 5], -k * x_fin, 6.15, "denk. 4.44, Tablo C.10"),
        Row("Kanat kaldırması", "N_uvf", -df[5, 1], -k * x_fin, 6.15, "denk. 4.45, Tablo C.10"),
        Row("Kanat kaldırması", "N_urf", -df[5, 5], -k * x_fin**2, -3.93, "denk. 4.45, Tablo C.10"),
        Row("Kanat kaldırması", "M_uqf", -df[4, 4], -k * x_fin**2, -3.93, "denk. 4.45, Tablo C.10"),
        Row("Kontrol", "Y_uudr", float(rudder[1]), k, 9.64, "denk. 4.44, Tablo C.10"),
        Row("Kontrol", "N_uudr", float(rudder[5]), k * x_fin, -6.15, "denk. 4.45, Tablo C.10"),
        Row("Kontrol", "Z_uuds", float(planes[2]), -k, -9.64, "denk. 4.44, Tablo C.10"),
        Row("Kontrol", "M_uuds", float(planes[4]), k * x_fin, -6.15, "denk. 4.45, Tablo C.10"),
        Row(
            "Gövde kaldırması",
            "Y_uvl",
            None,
            -1.90e1,
            -1.90e1,
            "Tablo 4.2 - C.10",
            "robomodel'de yok (Hoerner gövde kaldırması); Y_uv = -28,6 toplamından kanat payı çıkarıldı",
        ),
    ]


def steady_turn(model: VehicleModel, damping_scale: np.ndarray | None = None) -> float:
    """Steady yaw rate (deg/s) at SPEED with both rudders at RUDDER."""
    dyn = model.dynamics()
    if damping_scale is not None:
        from dataclasses import replace

        dyn = replace(dyn, d_quad=dyn.d_quad * damping_scale)
    names = model.fins.control_names
    deflection = np.array([RUDDER if n.startswith("rudder") else 0.0 for n in names])
    surge_force = dyn.d_lin[0] * SPEED + dyn.d_quad[0] * SPEED**2

    def tau(_t, x):  # hold the surge speed, like REMUS's speed controller
        u = x[6]
        return np.array([surge_force + 50.0 * (SPEED - u), 0, 0, 0, 0, 0])

    state0 = np.zeros(12)
    state0[6] = SPEED
    traj = simulate(dyn, 40.0, tau=tau, state0=state0, fin_deflection=deflection, dt_out=0.1)
    return float(np.degrees(abs(traj.nu[-1, 5])))


def _fmt(x: float | None) -> str:
    return "–" if x is None else f"{x:+.3g}"


def _pct(x: float | None) -> str:
    return "–" if x is None else f"{100 * x:+.0f}%"


def _diff(row: Row, ref: float | None) -> str:
    text = _pct(row.error(ref))
    return text if row.sign_ok(ref) else text + " **işaret ters**"


def write_report(rows: list[Row], turn: float, turn_adjusted: float) -> str:
    lines = [
        "# REMUS 100 doğrulaması: robomodel ve Prestero (2001)",
        "",
        "Bu dosya `validation/remus100.py` ile üretilir, elle düzenleme.",
        "",
        "Kaynak: T. Prestero, *Verification of a Six-Degree of Freedom Simulation Model for the",
        "REMUS Autonomous Underwater Vehicle*, MIT/WHOI yüksek lisans tezi, 2001.",
        "[MIT DSpace](https://dspace.mit.edu/handle/1721.1/65068)",
        "",
        "- **Teori:** Prestero'nun el kitabı ve şerit teorisi tahmini (Tablo 4.2, 4.3).",
        "  robomodel'in yaptığı tahminle aynı türden; yöntemimizi test eder.",
        "- **Ayarlı / ölçülen:** deniz testlerine göre düzeltilmiş (Tablo 8.2, Ek C) ya da",
        "  ölçülmüş değerler. Gerçeğe en yakın referans bu sütun.",
        "- İşaretler Fossen/Prestero notasyonunda. Kütle, kütle merkezi ve eylemsizlik tezden",
        "  ölçülmüş değer olarak girildi; karşılaştırmada sadece hesapladığımız terimler var.",
        "- **Fark** büyüklük farkıdır: +%10, robomodel'in değeri %10 büyük demek. İşaret",
        "  uyuşmazsa ayrıca yazılır.",
        "",
        "| Grup | Katsayı | robomodel | Prestero teori | Fark | Prestero ayarlı / ölçülen | Fark | Kaynak | Not |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r.group} | `{r.name}` | {_fmt(r.ours)} | {_fmt(r.theory)} | "
            f"{_diff(r, r.theory)} | {_fmt(r.adjusted)} | {_diff(r, r.adjusted)} | "
            f"{r.source} | {r.note} |"
        )
    lines += [
        "",
        "## Davranış: deniz testi",
        "",
        f"1,54 m/s hızda, iki dümen {np.degrees(RUDDER):.0f}° açıyla, sabit dönüş hızı:",
        "",
        "| Model | Dönüş hızı (°/s) | Deniz testine göre fark |",
        "|---|---|---|",
        f"| Deniz testi (Prestero 7.5.1, yaklaşık) | {FIELD_YAW_RATE:.1f} | – |",
        f"| robomodel, sadece CAD | {turn:.1f} | {_pct(turn / FIELD_YAW_RATE - 1)} |",
        f"| robomodel, Prestero'nun sönüm düzeltmeleriyle (Tablo 8.2) | {turn_adjusted:.1f} | "
        f"{_pct(turn_adjusted / FIELD_YAW_RATE - 1)} |",
        "",
        'Deniz testindeki değer bir grafikten okunmuş yaklaşık değerdir ("approximately 10',
        'degrees/second"); ±%20 civarında belirsizlik kabul edilmeli.',
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    model = build_model(load_vehicle(VEHICLE))
    rows = comparison(model)
    turn = steady_turn(model)
    scale = np.array([1, 10, 10, 1, 12.5, 10.0])  # Prestero Table 8.2 (K_p|p| x100 skipped: 0)
    turn_adjusted = steady_turn(model, scale)
    text = write_report(rows, turn, turn_adjusted)
    REPORT.write_text(text, encoding="utf-8")
    print(text)
    print(f"written: {REPORT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
