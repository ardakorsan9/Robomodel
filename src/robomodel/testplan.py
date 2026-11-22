"""Pool test plan: which manoeuvres to run so calibration can identify the uncertain terms.

Writes one CSV per manoeuvre with the input columns filled in (the format calibration reads)
and empty state columns to fill from the vehicle's log, plus a Turkish instruction sheet.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from robomodel.model import VehicleModel

DT = 0.1  # s
LOGGED = ("z", "phi", "theta", "psi", "u", "v", "w", "p", "q", "r")


@dataclass(frozen=True)
class Manoeuvre:
    name: str
    purpose: str  # what it identifies (Turkish)
    how: str  # what to do in the pool (Turkish)
    t: np.ndarray
    inputs: dict[str, np.ndarray]


def _square(t, amplitude, period, start):
    return np.where(
        t < start, 0.0, amplitude * np.sign(np.sin(2 * np.pi * (t - start) / period) + 1e-9)
    )


def _forward_thrusters(model: VehicleModel) -> list[int]:
    return [i for i, th in enumerate(model.vehicle.thrusters) if abs(th.direction[0]) > 0.7]


def plan(model: VehicleModel) -> list[Manoeuvre]:
    v = model.vehicle
    out = []
    rudders = [f.name for f in v.fins if f.control and abs(f.normal[1]) > 0.7]
    planes = [f.name for f in v.fins if f.control and abs(f.normal[2]) > 0.7]
    forward = _forward_thrusters(model)
    b = model.b_thr

    t = np.arange(0, 35 + DT / 2, DT)
    if forward:
        cmd = np.where(t < 20, 0.6, 0.0)
        out.append(
            Manoeuvre(
                "1_ileri_ve_sonumlenme",
                "ileri sönüm (surge): hızlanma ve motor kesilince yavaşlama",
                "Duruştan başla, ileri iticiler %60 ile 20 s git, sonra iticileri kapat ve 15 s "
                "süzülmesini kaydet. Havuz duvarına çarpmadan bitecek bir yön seç.",
                t,
                {f"thr_{i}": cmd * np.sign(v.thrusters[i].direction[0]) for i in forward},
            )
        )

    t = np.arange(0, 40 + DT / 2, DT)
    if rudders:
        delta = _square(t, np.radians(8), 12, 2)
        out.append(
            Manoeuvre(
                "2_zikzak",
                "sapma ve yan kayma sönümü, kanat ve gövde kaldırması",
                "Seyir hızında git (ileri itki sabit), 2 s sonra dümeni ±8° olarak her 6 s'de bir "
                "ters çevir. İleri itki komutunu da `tau_x` ya da `thr_` sütununa yaz.",
                t,
                {f"fin_{n}": delta for n in rudders},
            )
        )
    else:
        yaw = [i for i in range(b.shape[1]) if abs(b[5, i]) > 1e-6 and i not in forward] or [
            i for i in forward if abs(b[5, i]) > 1e-6
        ]
        if yaw:
            sign = np.sign(b[5, yaw])
            cmd = _square(t, 0.4, 12, 2)
            out.append(
                Manoeuvre(
                    "2_donus_zikzak",
                    "sapma sönümü ve dönme eklenmiş kütlesi",
                    "Yerinde dur, iticilerle ±%40 dönme komutunu her 6 s'de bir ters çevir.",
                    t,
                    {f"thr_{i}": cmd * s for i, s in zip(yaw, sign, strict=True)},
                )
            )

    t = np.arange(0, 36 + DT / 2, DT)
    if planes:
        pattern = np.radians([0, 6, -6, 0, -4, 4])
        delta = pattern[np.minimum((t // 6).astype(int), len(pattern) - 1)]
        out.append(
            Manoeuvre(
                "3_dalis",
                "dalma ve yunuslama sönümü, kanat kaldırması",
                "Seyir hızında, derinlik dümenini 6 s aralıklarla 0, +6°, -6°, 0, -4°, +4° yap. "
                "Havuz derinliği en az 2 m olmalı; sığsa açıları yarıya indir.",
                t,
                {f"fin_{n}": delta for n in planes},
            )
        )
    else:
        vertical = [i for i in range(b.shape[1]) if abs(b[2, i]) > 1e-6]
        if vertical:
            cmd = np.where((t > 2) & (t < 10), 0.4, 0.0)
            out.append(
                Manoeuvre(
                    "3_dikey",
                    "dalma sönümü ve eklenmiş kütlesi, net kaldırma",
                    "Yüzeye yakın başla, dikey iticileri %40 ile 8 s aşağı bas, sonra kapat ve "
                    "aracın yükselmesini kaydet.",
                    t,
                    {f"thr_{i}": cmd * np.sign(b[2, i]) for i in vertical},
                )
            )

    t = np.arange(0, 20 + DT / 2, DT)
    out.append(
        Manoeuvre(
            "4_serbest_salinim",
            "yuvarlanma ve yunuslama sönümü, kütle merkezi yüksekliği",
            "İticiler kapalıyken aracı elle yaklaşık 20° yana yatır ve bırak; salınımı 20 s "
            "kaydet. Sonra aynısını burun yukarı yatırarak yap (ayrı kayıt).",
            t,
            {},
        )
    )
    return out


def write_plan(model: VehicleModel, out_dir: str | Path) -> list[Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written = []
    sheet = [
        f"# Havuz test planı: {model.vehicle.name}",
        "",
        "Her test için bir CSV şablonu var. Giriş sütunları (thr_, fin_) dolu; durum",
        f"sütunlarını ({', '.join(LOGGED)}) aracın kaydından doldur. Birimler: m, m/s, rad,",
        "rad/s; z aşağı pozitif (derinlik), açılar NED. Kayıt sıklığı en az 10 Hz.",
        "",
        "Ölçemediğin sütunu sil (boş bırakma). Testleri iki kez yap: birini kalibrasyona,",
        "birini kontrole ayır.",
        "",
        "**Güvenlik:** testleri araç başında, havuzda biri varken yap. İticiler dönerken",
        "aracı elle tutma; serbest salınım testinde iticiler kapalı olmalı.",
        "",
        "| Test | Neyi ölçer | Nasıl yapılır |",
        "|---|---|---|",
    ]
    for m in plan(model):
        path = out / f"{m.name}.csv"
        columns = {"t": m.t, **m.inputs}
        header = ",".join(list(columns) + list(LOGGED))
        data = np.column_stack(list(columns.values()))
        with path.open("w", encoding="utf-8") as f:
            f.write(header + "\n")
            for row in data:
                f.write(",".join(f"{x:.6g}" for x in row) + "," * len(LOGGED) + "\n")
        written.append(path)
        sheet.append(f"| `{m.name}.csv` | {m.purpose} | {m.how} |")
    sheet += [
        "",
        "Kalibrasyon:",
        "",
        "```bash",
        "uv run robomodel calibrate ARAC.yaml 1_*.csv 2_*.csv 3_*.csv --validate KONTROL.csv",
        "```",
    ]
    readme = out / "TEST_PLANI.md"
    readme.write_text("\n".join(sheet) + "\n", encoding="utf-8")
    written.append(readme)
    return written
