"""Compare robomodel's BlueROV2 Heavy model with von Benzon et al. (2022).

Reference: F. von Benzon et al., "An Open-Source Benchmark Simulator: Control of a BlueROV2
Underwater Robot", J. Mar. Sci. Eng. 10 (2022) 1898, Table A1 and Tables 4-5.
Damping there was identified in a pool and validated within about 10-15 %; added mass is an
empirical estimate (Eidsvik), with 10-20 % (translation) and 30-100 % (rotation) error.

Run:  uv run python validation/bluerov2.py   (writes docs/dogrulama_bluerov2.md)
Needs the Blue Robotics CAD file named in examples/bluerov2_heavy.yaml.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from robomodel.model import build_model
from robomodel.vehicle import load_vehicle

ROOT = Path(__file__).resolve().parent.parent
VEHICLE = ROOT / "examples" / "bluerov2_heavy.yaml"
REPORT = ROOT / "docs" / "dogrulama_bluerov2.md"
DOF = ("ileri (u)", "yan (v)", "dikey (w)", "yuvarlanma (p)", "yunuslama (q)", "sapma (r)")

# Table A1 (Fossen sign convention would be negative; given here as magnitudes)
PAPER_QUAD = np.array([141.0, 217.0, 190.0, 1.19, 0.47, 1.5])  # experimental
PAPER_LIN = np.array([13.7, 0.0, 33.0, 0.0, 0.8, 0.0])  # experimental
PAPER_ADDED = np.array([6.36, 7.12, 18.68, 0.189, 0.135, 0.222])  # Eidsvik, empirical
PAPER_AREAS = {"A_u (önden)": 0.0877, "A_v (yandan)": 0.1131, "A_w (üstten)": 0.2049}  # Table 5
PAPER_VOLUME = 0.0134  # experimental
SURGE_FORCE = 85.0  # N, Table 4
PAPER_MAX_SURGE = 0.72  # m/s, Table 4


def steady_speed(lin: float, quad: float, force: float) -> float:
    return (-lin + np.sqrt(lin * lin + 4 * quad * force)) / (2 * quad)


def run():
    model = build_model(load_vehicle(VEHICLE))
    hull = model.hull
    areas = {
        name: hull.projection(axis)[1].sum()
        for name, axis in zip(PAPER_AREAS, (0, 1, 2), strict=True)
    }
    return model, areas


def report(model, areas) -> str:
    quad = np.asarray(model.damping.quadratic.value)
    lin = np.asarray(model.damping.linear.value)
    added = np.diag(np.asarray(model.m_a.value))
    cad_volume = float(model.hull.volume)

    def pct(a, b):
        return "–" if b == 0 else f"{100 * (a / b - 1):+.0f}%"

    lines = [
        "# BlueROV2 Heavy doğrulaması: robomodel ve von Benzon vd. (2022)",
        "",
        "Bu dosya `validation/bluerov2.py` ile üretilir, elle düzenleme.",
        "",
        "Kaynak: F. von Benzon vd., *An Open-Source Benchmark Simulator: Control of a BlueROV2",
        "Underwater Robot*, J. Mar. Sci. Eng. 10 (2022) 1898, Tablo A1, 4, 5.",
        "CAD: Blue Robotics'in açık BlueROV2 Heavy montajı (1142 parça, açık kasa).",
        "",
        "Makalede **sönüm havuzda deneyle bulunmuş** (doğrulama hatası ~%10–15); eklenmiş kütle",
        "ampirik (Eidsvik). Kütle, hacim, kütle merkezi ve eylemsizlik makaleden ölçülmüş değer",
        "olarak girildi; karşılaştırmada sadece CAD'den hesapladıklarımız var.",
        "",
        "## Geometri (CAD işleme)",
        "",
        "| Büyüklük | robomodel | Makale | Fark |",
        "|---|---|---|---|",
    ]
    for name, ours in areas.items():
        lines.append(
            f"| {name} | {ours:.4f} m² | {PAPER_AREAS[name]:.4f} m² | "
            f"{pct(ours, PAPER_AREAS[name])} |"
        )
    lines += [
        f"| Hacim (CAD'den, voksel) | {cad_volume:.4f} m³ | {PAPER_VOLUME:.4f} m³ (ölçülen) | "
        f"{pct(cad_volume, PAPER_VOLUME)} |",
        "",
        "## Sönüm (makalede deneyle ölçülmüş)",
        "",
        "| Eksen | robomodel karesel | Makale karesel | Fark | robomodel doğrusal | Makale doğrusal |",
        "|---|---|---|---|---|---|",
    ]
    for i, dof in enumerate(DOF):
        lines.append(
            f"| {dof} | {quad[i]:.3g} | {PAPER_QUAD[i]:.3g} | {pct(quad[i], PAPER_QUAD[i])}"
            f" | {lin[i]:.3g} | {PAPER_LIN[i]:.3g} |"
        )
    lines += [
        "",
        "Birimler: karesel N s²/m² (dönmede N m s²/rad²), doğrusal N s/m (N m s/rad).",
        "Makalenin doğrusal terimleri ölçülmüş; bizimki karesel × 0,1 m/s varsayımı.",
        "",
        "## Eklenmiş kütle (makalede de ampirik tahmin)",
        "",
        f"| Eksen | robomodel ({'BEM' if model.m_a.source.value == 'bem' else 'saran elipsoit'}) "
        "| Makale (Eidsvik) | Fark |",
        "|---|---|---|---|",
    ]
    for i, dof in enumerate(DOF):
        lines.append(
            f"| {dof} | {added[i]:.3g} | {PAPER_ADDED[i]:.3g} | {pct(added[i], PAPER_ADDED[i])} |"
        )
    ours_speed = steady_speed(lin[0], quad[0], SURGE_FORCE)
    paper_speed = steady_speed(PAPER_LIN[0], PAPER_QUAD[0], SURGE_FORCE)
    lines += [
        "",
        "İkisi de tahmin; hangisinin doğru olduğunu bu veriyle bilemeyiz.",
        "",
        f"## Davranış: {SURGE_FORCE:.0f} N ileri itkide son hız",
        "",
        "| Model | Son hız (m/s) | Makaleye göre |",
        "|---|---|---|",
        f"| Makale (havuzda ölçülen sönümle, Tablo 4: {PAPER_MAX_SURGE} m/s) | {paper_speed:.2f} | – |",
        f"| robomodel, sadece CAD | {ours_speed:.2f} | {pct(ours_speed, paper_speed)} |",
        "",
        "## Yorum",
        "",
        "- CAD işleme iyi: izdüşüm alanları makalenin CAD değerleriyle birkaç yüzde içinde.",
        "- Hacim CAD'den iyi çıkmıyor (vokseller ince plakaları kalınlaştırıyor, tüplerin",
        "  içindeki hava CAD'de parça olarak yok): açık kasalı araçta suda tartmak şart.",
        "- Sönüm REMUS'taki gibi CAD'den düşük çıkıyor: C_D = 1,1 ile 1,6–4,6 kat az. Makalenin",
        "  değerleri izdüşüm alanına göre C_D ≈ 1,9 (dikey) – 3,8 (yan) demek; iticiler, kablolar",
        "  ve kasa içinden geçen akış bunu büyütüyor. Doğru değer için havuz testi ve",
        "  `robomodel calibrate` gerekli.",
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
