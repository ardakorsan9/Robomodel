"""Spread the weighed mass over the parts of a CAD assembly by their likely material.

A CAD file holds shapes, not materials. With one uniform density, a battery and a foam float
of the same size weigh the same, which puts the centre of gravity and the inertia in the
wrong place. A STEP assembly, however, keeps every part separate and usually named
("LiPo_3S", "T200_thruster", "M3x10_screw", "frame_plate"). So:

1. each part's material is guessed from its name (English and Turkish keywords); parts with
   no telling name get a neutral default;
2. each part's mass follows from its volume and that density (open, sheet-like parts: their
   area times a wall thickness);
3. all part masses are scaled together so that they add up to the weighed mass.

Step 3 makes the result depend only on the density *ratios*, which are far more certain than
the absolute values, and the scale itself is a check: far from 1 means the mass, the units
or hollow parts need a look.

Densities are typical values (kg/m^3), assumed; effective ones for parts that are not solid
(a motor or a battery pack is metal, plastic and air).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace

import numpy as np
import trimesh

SHELL_THICKNESS = 0.002  # m, wall of a part whose surface is not closed (sheet, open shell)
DEFAULT_DENSITY = 1500.0  # unknown part: between plastics and aluminium
SCALE_WARN = (0.33, 3.0)  # weighed / estimated mass outside this: say so

# (material, density, keywords). The first match wins, so specific names come first.
MATERIALS: list[tuple[str, float, tuple[str, ...]]] = [
    ("lead ballast", 11300.0, ("lead", "ballast", "kursun", "kurşun", "safra")),
    (
        "battery",
        1600.0,
        (
            "battery",
            "batt",
            "lipo",
            "li-ion",
            "liion",
            "18650",
            "21700",
            "cell",
            "akku",
            "pil",
            "batarya",
            "aku",
        ),
    ),
    (
        "motor / thruster",
        3500.0,
        (
            "motor",
            "thruster",
            "servo",
            "dynamixel",
            "stepper",
            "bldc",
            "gearbox",
            "gearmotor",
            "reducer",
            "itici",
            "redüktör",
        ),
    ),
    (
        "steel fastener",
        7850.0,
        (
            "screw",
            "bolt",
            "nut",
            "washer",
            "vida",
            "somun",
            "pul",
            "rivet",
            "pin",
            "dowel",
            "standoff",
        ),
    ),
    (
        "steel",
        7850.0,
        (
            "steel",
            "stainless",
            "celik",
            "çelik",
            "shaft",
            "mil",
            "bearing",
            "rulman",
            "spring",
            "yay",
        ),
    ),
    ("brass / copper", 8600.0, ("brass", "copper", "pirinc", "pirinç", "bakir", "bakır")),
    (
        "circuit board",
        1850.0,
        (
            "pcb",
            "board",
            "kart",
            "raspberry",
            "arduino",
            "jetson",
            "esc",
            "controller",
            "pixhawk",
            "navigator",
        ),
    ),
    (
        "electronics",
        1000.0,
        (
            "electronics",
            "elektronik",
            "module",
            "modul",
            "modül",
            "sensor",
            "sensör",
            "camera",
            "kamera",
            "gps",
            "imu",
            "sonar",
            "lidar",
        ),
    ),
    ("foam / float", 300.0, ("foam", "float", "buoyancy", "kopuk", "köpük", "flotation")),
    ("glass", 2500.0, ("glass", "cam_", "lens")),
    ("acrylic", 1190.0, ("acrylic", "pmma", "dome", "akrilik", "plexi", "window")),
    ("carbon fibre", 1600.0, ("carbon", "cfrp", "karbon")),
    (
        "aluminium",
        2700.0,
        (
            "aluminium",
            "aluminum",
            "alu",
            "6061",
            "7075",
            "5083",
            "extrusion",
            "profile",
            "profil",
            "sigma",
        ),
    ),
    (
        "rubber",
        1150.0,
        ("rubber", "tire", "tyre", "lastik", "o-ring", "oring", "gasket", "seal", "conta"),
    ),
    (
        "printed plastic",
        900.0,
        ("pla", "petg", "abs", "print", "baski", "baskı", "nylon", "naylon"),
    ),  # solid ~1250, typical infill lower
    (
        "plastic",
        1250.0,
        (
            "plastic",
            "plastik",
            "hdpe",
            "pom",
            "delrin",
            "pvc",
            "polycarbonate",
            "housing",
            "fairing",
            "shroud",
            "cover",
            "kapak",
            "enclosure",
            "tube",
            "tup",
            "tüp",
        ),
    ),
    (
        "structure (metal or plastic)",
        2000.0,  # aluminium or plastic (HDPE, POM) frames are both common
        (
            "frame",
            "plate",
            "panel",
            "bracket",
            "arm",
            "chassis",
            "sasi",
            "şasi",
            "govde",
            "gövde",
            "plaka",
        ),
    ),
]


@dataclass(frozen=True)
class CadPart:
    name: str
    volume: float  # m^3 of material (closed parts: solid volume; open: area x wall)
    centroid: np.ndarray  # body frame, m
    inertia_unit: np.ndarray  # 3x3 about the centroid per kg of the part's mass
    closed: bool
    material: str
    density: float
    by_name: bool  # material from the part's name (else the default)
    mass: float = 0.0  # kg after scaling to the weighed mass


def guess_material(name: str) -> tuple[str, float, bool]:
    text = name.lower()
    words = set(re.split(r"[^a-zçğıöşü]+", text))  # "M3-SCREW_12" -> m, screw
    for material, density, keys in MATERIALS:
        for key in keys:
            # short keys must be whole words ("pin" not in "spinner"); longer ones may sit
            # inside a word ("lipo" in "lipo3s", "motor" in "gearmotor")
            if (len(key) <= 3 and key in words) or (len(key) > 3 and key in text):
                return material, density, True
    return "unknown", DEFAULT_DENSITY, False


def cad_part(name: str, mesh: trimesh.Trimesh) -> CadPart | None:
    """Mass data of one part (per unit mass), or None for an empty or flat-less part."""
    if len(mesh.faces) == 0 or mesh.area <= 0:
        return None
    material, density, by_name = guess_material(name)
    if not mesh.is_watertight:  # STEP readers often give every face its own vertices
        mesh = mesh.copy()
        mesh.merge_vertices()
    closed = bool(mesh.is_watertight) and abs(mesh.volume) > 1e-12
    if closed:
        volume = abs(float(mesh.volume))
        centroid = np.asarray(mesh.center_mass, dtype=float)
        inertia = np.asarray(mesh.moment_inertia, dtype=float) / volume  # density 1 -> per kg
    else:
        from robomodel.rigid_body import shell_properties

        volume = float(mesh.area) * SHELL_THICKNESS
        centroid, inertia = shell_properties(mesh, 1.0)
    return CadPart(name, volume, centroid, inertia, closed, material, density, by_name)


def scene_parts(scene: trimesh.Scene, transform: np.ndarray) -> list[CadPart]:
    """Every geometry node of a loaded assembly, moved by `transform` (4x4, units + axes)."""
    parts = []
    for node in scene.graph.nodes_geometry:
        pose, geom_name = scene.graph[node]
        geom = scene.geometry.get(geom_name)
        if not isinstance(geom, trimesh.Trimesh):
            continue
        mesh = geom.copy()
        mesh.apply_transform(transform @ pose)
        part = cad_part(f"{node} {geom_name}", mesh)
        if part is not None:
            parts.append(part)
    return parts


@dataclass(frozen=True)
class Distribution:
    parts: list[CadPart]  # with masses
    scale: float  # weighed mass / mass from the guessed densities
    share_by_name: float  # share of the mass whose material came from a part name

    @property
    def estimated_mass(self) -> float:
        return sum(p.mass for p in self.parts) / self.scale


def distribute(parts: list[CadPart], mass: float) -> Distribution | None:
    raw = np.array([p.volume * p.density for p in parts])
    if not len(parts) or raw.sum() <= 0 or mass <= 0:
        return None
    scale = mass / raw.sum()
    with_mass = [replace(p, mass=float(m * scale)) for p, m in zip(parts, raw, strict=True)]
    named = sum(p.mass for p in with_mass if p.by_name) / mass
    return Distribution(with_mass, float(scale), float(named))


def combined(dist: Distribution) -> tuple[float, np.ndarray, np.ndarray]:
    """(mass, centre of mass, inertia about the body origin) of all parts."""
    from robomodel.rigid_body import point_inertia

    m = sum(p.mass for p in dist.parts)
    first = sum(p.mass * p.centroid for p in dist.parts)
    inertia = sum(p.mass * p.inertia_unit + point_inertia(p.mass, p.centroid) for p in dist.parts)
    return m, first / m, inertia


def translate(parts: list[CadPart], offset) -> list[CadPart]:
    offset = np.asarray(offset, dtype=float)
    return [replace(p, centroid=p.centroid + offset) for p in parts]


def warnings(dist: Distribution | None) -> list[str]:
    """A weighed mass far from what the parts would weigh in their likely materials."""
    if dist is None:
        return []
    lo, hi = SCALE_WARN
    if lo <= dist.scale <= hi:
        return []
    return [
        f"with typical materials the CAD parts would weigh {dist.estimated_mass:.3g} kg, but the "
        f"mass given is {dist.estimated_mass * dist.scale:.3g} kg: check the mass, the file units, "
        "or parts drawn solid that are hollow (or the other way round)"
    ]


def summary(dist: Distribution, top: int = 8) -> list[str]:
    """Report lines: the heaviest parts and their guessed materials."""
    rows = sorted(dist.parts, key=lambda p: -p.mass)
    lines = [
        f"{len(dist.parts)} CAD parts; {100 * dist.share_by_name:.0f} % of the mass has a "
        f"material from the part name; guessed materials give {dist.estimated_mass:.3g} kg "
        f"(scaled x{dist.scale:.2f} to the weighed mass)"
    ]
    for p in rows[:top]:
        lines.append(f"  {p.mass:8.3g} kg  {p.material:<22} {p.name[:50]}")
    if len(rows) > top:
        lines.append(f"  ... {len(rows) - top} lighter parts")
    return lines
