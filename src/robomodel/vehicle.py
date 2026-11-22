"""Vehicle description: load and validate a vehicle YAML file.

The YAML file is the single source of truth for a vehicle. All positions are in
the body frame (x forward, y right, z down; see docs/eksenler.md), in metres.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

CATEGORIES = ("underwater",)
WATER_DENSITY = {"sea": 1025.0, "fresh": 1000.0}  # kg/m^3

# Primitive shapes and the dimensions each one needs (all in metres).
PRIMITIVES = {
    "sphere": ("radius",),
    "ellipsoid": ("a", "b", "c"),  # semi-axes along x, y, z
    "cylinder": ("length", "radius"),  # axis along x
    "myring": ("a", "b", "c", "d", "n", "theta"),  # REMUS-style hull, see Prestero 2001
}
# Optional dimensions and their defaults.
OPTIONAL_DIMS = {"myring": {"a_offset": 0.0, "c_offset": 0.0}}
ORIGINS = ("center", "buoyancy")
HYDRODYNAMICS = ("auto", "slender", "frame")
AXIS_NAMES = ("x", "y", "z", "-x", "-y", "-z")


def axes_matrix(axes: tuple[str, str, str]) -> np.ndarray:
    """Rotation R with body = R @ file, from ('z', '-x', '-y')-style names."""
    rows = []
    for name in axes:
        sign = -1.0 if name.startswith("-") else 1.0
        row = np.zeros(3)
        row["xyz".index(name[-1])] = sign
        rows.append(row)
    return np.array(rows)


LENGTH_UNITS = {"m": 1.0, "mm": 1e-3, "cm": 1e-2}

# Shapes for internal parts (batteries, electronics tubes); they set the part's own inertia.
PART_SHAPES = {
    "point": (),
    "sphere": ("radius",),
    "box": ("size",),  # [lx, ly, lz] along body axes
    "cylinder": ("radius", "length", "axis"),  # axis: x, y or z
}

# Thrust limits in newtons (forward, reverse) for known thruster models.
# T200 values: Blue Robotics datasheet at 16 V, as remembered (5.25 / 4.1 kgf); check them.
THRUSTER_PRESETS = {"T200": (51.5, 40.2)}


class VehicleError(ValueError):
    """Raised when a vehicle file is invalid. Holds every problem found."""

    def __init__(self, problems: list[str]):
        self.problems = problems
        super().__init__("Invalid vehicle file:\n  - " + "\n  - ".join(problems))


@dataclass(frozen=True)
class Geometry:
    """Either a primitive shape or a mesh file (exactly one is set)."""

    primitive: str | None = None
    dims: dict[str, float] = field(default_factory=dict)
    file: Path | None = None
    unit_scale: float = 1.0  # multiply mesh coordinates by this to get metres
    # Body origin: "center" keeps primitives centred on their bounding box (mesh files keep
    # their own origin); "buoyancy" moves the origin to the centre of buoyancy.
    origin: str = "center"
    # Body axis -> file axis, e.g. ("z", "-x", "-y"): body x is the file's z axis, body y is
    # the file's -x axis... Must be a right-handed rotation.
    axes: tuple[str, str, str] = ("x", "y", "z")
    rotate: tuple[float, float, float] = (0.0, 0.0, 0.0)  # extra roll, pitch, yaw (deg)
    # Multi-part CAD assembly (open-frame ROV): turned into one closed solid by voxels.
    assembly: bool = False
    voxel: float = 0.004  # m, voxel size for assemblies
    # "slender": strip theory and Hoerner drag (torpedo AUVs); "frame": projected areas and
    # an enclosing ellipsoid (open-frame ROVs). "auto": frame for assemblies or L/D < 2.
    hydrodynamics: str = "auto"


@dataclass(frozen=True)
class Part:
    name: str
    mass: float  # kg
    position: np.ndarray  # centre of the part, body frame, m
    shape: str = "point"
    dims: dict = field(default_factory=dict)


@dataclass(frozen=True)
class Thruster:
    model: str
    position: np.ndarray  # body frame, m
    direction: np.ndarray  # unit vector, body frame, direction of positive thrust
    max_forward: float  # N, thrust at command +1
    max_reverse: float  # N (positive number), thrust magnitude at command -1
    preset: bool = False  # True if the limits come from THRUSTER_PRESETS, not the user
    dead_band: float = 0.0  # fraction of the command range with no thrust (e.g. 0.05)


@dataclass(frozen=True)
class Fin:
    """A thin lifting surface (fin, rudder, stern plane) outside the hull.

    The fin pushes water along `normal`; `position` is its centre of pressure.
    """

    name: str
    position: np.ndarray  # body frame, m
    normal: np.ndarray  # unit vector, the direction the lift acts in
    area: float  # planform area of one fin, m^2
    span: float  # root to tip, m
    control: bool = False  # True: the fin can be turned (rudder, stern plane)
    max_angle: float = np.radians(20.0)  # rad; beyond this the fin stalls (lift held flat)


@dataclass(frozen=True)
class Measured:
    """Values measured on the real vehicle; they replace the CAD estimates."""

    r_g: np.ndarray | None = None  # centre of gravity, m
    inertia_g: np.ndarray | None = None  # 3x3 inertia about the CG, kg m^2
    volume: float | None = None  # displaced volume from weighing in water, m^3


@dataclass(frozen=True)
class Calibration:
    """Scale factors fitted to test data (robomodel calibrate); 1 means the CAD value.

    Each *_sigma is the fitted relative uncertainty of the scale (1 sigma).
    """

    damping: np.ndarray = field(default_factory=lambda: np.ones(6))  # per DOF, lin + quad
    added_mass: np.ndarray = field(default_factory=lambda: np.ones(6))  # diagonal of M_A
    fin_lift: float = 1.0
    body_lift: float = 1.0
    thrust: float = 1.0  # all thrusters' force
    thrust_sigma: float = 0.0
    damping_sigma: np.ndarray = field(default_factory=lambda: np.zeros(6))
    added_mass_sigma: np.ndarray = field(default_factory=lambda: np.zeros(6))
    fin_lift_sigma: float = 0.0
    body_lift_sigma: float = 0.0
    note: str = ""

    @property
    def is_identity(self) -> bool:
        return (
            np.allclose(self.damping, 1)
            and np.allclose(self.added_mass, 1)
            and self.fin_lift == 1
            and self.body_lift == 1
            and self.thrust == 1
        )


@dataclass(frozen=True)
class Vehicle:
    name: str
    category: str
    mass: float  # total mass, kg (includes parts)
    water_density: float  # kg/m^3
    geometry: Geometry
    parts: tuple[Part, ...] = ()
    thrusters: tuple[Thruster, ...] = ()
    fins: tuple[Fin, ...] = ()
    measured: Measured = field(default_factory=Measured)
    calibration: Calibration = field(default_factory=Calibration)
    source: Path | None = None  # the YAML file it was read from, for provenance
    current: np.ndarray = field(default_factory=lambda: np.zeros(3))  # NED, m/s


def load_vehicle(path: str | Path) -> Vehicle:
    """Read a vehicle YAML file. Mesh paths are resolved relative to the file."""
    path = Path(path)
    with path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f)
    from dataclasses import replace

    return replace(parse_vehicle(data, base_dir=path.parent), source=path.resolve())


def parse_vehicle(data: object, base_dir: Path = Path(".")) -> Vehicle:
    """Validate a parsed YAML mapping and build a Vehicle.

    Collects all problems before raising, so the user can fix them in one go.
    """
    problems: list[str] = []
    if not isinstance(data, dict):
        raise VehicleError(["file must contain a mapping (key: value pairs)"])

    name = data.get("name")
    if not isinstance(name, str) or not name:
        problems.append("'name' is required and must be text")

    category = data.get("category")
    if category not in CATEGORIES:
        problems.append(f"'category' must be one of {list(CATEGORIES)}, got {category!r}")

    mass = _positive(data.get("mass"), "mass", problems)
    water_density = _water_density(data.get("water", "sea"), problems)
    geometry = _geometry(data.get("geometry"), base_dir, problems)

    parts = tuple(
        p
        for i, raw in enumerate(_list(data.get("parts"), "parts", problems))
        if (p := _part(raw, i, problems)) is not None
    )
    thrusters = tuple(
        t
        for i, raw in enumerate(_list(data.get("thrusters"), "thrusters", problems))
        if (t := _thruster(raw, i, problems)) is not None
    )

    fins = tuple(
        f
        for i, raw in enumerate(_list(data.get("fins"), "fins", problems))
        if (f := _fin(raw, i, problems)) is not None
    )
    measured = _measured(data.get("measured"), problems)
    calibration = _calibration(data.get("calibration"), problems)
    current = np.zeros(3)
    if "current" in data:
        current = _vector(data["current"], "current", problems)

    if mass is not None and parts:
        parts_mass = sum(p.mass for p in parts)
        if parts_mass > mass:
            problems.append(
                f"parts weigh {parts_mass:g} kg in total, more than the vehicle mass {mass:g} kg"
            )

    if problems:
        raise VehicleError(problems)
    return Vehicle(
        name=name,
        category=category,
        mass=mass,
        water_density=water_density,
        geometry=geometry,
        parts=parts,
        thrusters=thrusters,
        fins=fins,
        measured=measured,
        calibration=calibration,
        current=current if current is not None else np.zeros(3),
    )


def _positive(value: object, label: str, problems: list[str]) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        problems.append(f"'{label}' must be a positive number, got {value!r}")
        return None
    return float(value)


def _vector(value: object, label: str, problems: list[str]) -> np.ndarray | None:
    if (
        not isinstance(value, (list, tuple))
        or len(value) != 3
        or not all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in value)
    ):
        problems.append(f"'{label}' must be a list of 3 numbers [x, y, z], got {value!r}")
        return None
    return np.array(value, dtype=float)


def _list(value: object, label: str, problems: list[str]) -> list:
    if value is None:
        return []
    if not isinstance(value, list):
        problems.append(f"'{label}' must be a list")
        return []
    return value


def _water_density(value: object, problems: list[str]) -> float:
    if isinstance(value, str) and value in WATER_DENSITY:
        return WATER_DENSITY[value]
    if isinstance(value, (int, float)) and not isinstance(value, bool) and 900 <= value <= 1100:
        return float(value)
    problems.append(
        f"'water' must be 'sea', 'fresh' or a density in kg/m^3 (900-1100), got {value!r}"
    )
    return WATER_DENSITY["sea"]


def _geometry(raw: object, base_dir: Path, problems: list[str]) -> Geometry | None:
    if not isinstance(raw, dict):
        problems.append("'geometry' is required: give either 'primitive' or 'file'")
        return None
    has_primitive, has_file = "primitive" in raw, "file" in raw
    if has_primitive == has_file:
        problems.append("'geometry' needs exactly one of 'primitive' or 'file'")
        return None
    origin = raw.get("origin", "center")
    if origin not in ORIGINS:
        problems.append(f"'geometry.origin' must be one of {list(ORIGINS)}, got {origin!r}")
        return None
    hydro = raw.get("hydrodynamics", "auto")
    if hydro not in HYDRODYNAMICS:
        problems.append(f"'geometry.hydrodynamics' must be one of {list(HYDRODYNAMICS)}")
        return None
    axes = raw.get("axes", ["x", "y", "z"])
    if (
        not isinstance(axes, list)
        or len(axes) != 3
        or not all(a in AXIS_NAMES for a in axes)
        or abs(np.linalg.det(axes_matrix(tuple(axes))) - 1) > 1e-9
    ):
        problems.append(
            "'geometry.axes' must be 3 of x, y, z, -x, -y, -z forming a "
            f"right-handed rotation, got {axes!r}"
        )
        return None
    assembly = raw.get("assembly", False)
    if not isinstance(assembly, bool) or (assembly and not has_file):
        problems.append("'geometry.assembly' must be true/false and needs a 'file'")
        return None
    voxel_mm = raw.get("voxel", 4.0)
    if (
        isinstance(voxel_mm, bool)
        or not isinstance(voxel_mm, (int, float))
        or not (0.5 <= voxel_mm <= 50)
    ):
        problems.append("'geometry.voxel' must be in millimetres, 0.5-50")
        return None
    rotate = raw.get("rotate", [0, 0, 0])
    if (
        not isinstance(rotate, list)
        or len(rotate) != 3
        or not all(isinstance(a, (int, float)) and not isinstance(a, bool) for a in rotate)
    ):
        problems.append("'geometry.rotate' must be [roll, pitch, yaw] in degrees")
        return None
    extra = dict(
        origin=origin,
        axes=tuple(axes),
        rotate=tuple(float(a) for a in rotate),
        assembly=assembly,
        voxel=float(voxel_mm) / 1000,
        hydrodynamics=hydro,
    )

    if has_file:
        unit = raw.get("unit", "m")
        if unit not in LENGTH_UNITS:
            problems.append(f"'geometry.unit' must be one of {list(LENGTH_UNITS)}, got {unit!r}")
            return None
        file = (base_dir / str(raw["file"])).resolve()
        if not file.is_file():
            problems.append(f"geometry file not found: {file}")
            return None
        return Geometry(file=file, unit_scale=LENGTH_UNITS[unit], **extra)

    kind = raw["primitive"]
    if kind not in PRIMITIVES:
        problems.append(f"'geometry.primitive' must be one of {list(PRIMITIVES)}, got {kind!r}")
        return None
    dims = {}
    for key in PRIMITIVES[kind]:
        value = _positive(raw.get(key), f"geometry.{key}", problems)
        if value is not None:
            dims[key] = value
    if len(dims) != len(PRIMITIVES[kind]):
        return None
    for key, default in OPTIONAL_DIMS.get(kind, {}).items():
        value = raw.get(key, default)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
            problems.append(f"'geometry.{key}' must be a number >= 0, got {value!r}")
            return None
        dims[key] = float(value)
    return Geometry(primitive=kind, dims=dims, **extra)


def _part(raw: object, i: int, problems: list[str]) -> Part | None:
    if not isinstance(raw, dict):
        problems.append(f"parts[{i}] must be a mapping with name, mass, position")
        return None
    name = raw.get("name", f"part{i}")
    mass = _positive(raw.get("mass"), f"parts[{i}].mass", problems)
    position = _vector(raw.get("position"), f"parts[{i}].position", problems)
    shape = raw.get("shape", "point")
    if shape not in PART_SHAPES:
        problems.append(f"parts[{i}].shape must be one of {list(PART_SHAPES)}, got {shape!r}")
        return None
    dims: dict = {}
    for key in PART_SHAPES[shape]:
        label = f"parts[{i}].{key}"
        if key == "size":
            size = _vector(raw.get("size"), label, problems)
            if size is not None and np.all(size > 0):
                dims["size"] = size
            elif size is not None:
                problems.append(f"'{label}' values must be positive")
        elif key == "axis":
            axis = raw.get("axis", "x")
            if axis in ("x", "y", "z"):
                dims["axis"] = axis
            else:
                problems.append(f"'{label}' must be x, y or z, got {axis!r}")
        else:
            value = _positive(raw.get(key), label, problems)
            if value is not None:
                dims[key] = value
    if mass is None or position is None or len(dims) != len(PART_SHAPES[shape]):
        return None
    return Part(name=str(name), mass=mass, position=position, shape=shape, dims=dims)


def _thruster(raw: object, i: int, problems: list[str]) -> Thruster | None:
    if not isinstance(raw, dict):
        problems.append(f"thrusters[{i}] must be a mapping with model, position, direction")
        return None
    model = raw.get("model")
    if not isinstance(model, str) or not model:
        problems.append(f"thrusters[{i}].model is required")
    position = _vector(raw.get("position"), f"thrusters[{i}].position", problems)
    direction = _vector(raw.get("direction"), f"thrusters[{i}].direction", problems)
    if direction is not None:
        norm = np.linalg.norm(direction)
        if norm < 1e-9:
            problems.append(f"thrusters[{i}].direction must not be zero")
            direction = None
        else:
            direction = direction / norm
    preset = THRUSTER_PRESETS.get(model) if isinstance(model, str) else None
    limits = []
    for key, default in zip(("max_forward", "max_reverse"), preset or (None, None), strict=True):
        if key in raw or default is None:
            limits.append(_positive(raw.get(key), f"thrusters[{i}].{key}", problems))
        else:
            limits.append(default)
    uses_preset = preset is not None and "max_forward" not in raw and "max_reverse" not in raw
    dead_band = raw.get("dead_band", 0.0)
    if (
        isinstance(dead_band, bool)
        or not isinstance(dead_band, (int, float))
        or not (0 <= dead_band < 0.5)
    ):
        problems.append(f"thrusters[{i}].dead_band must be a fraction 0-0.5, got {dead_band!r}")
        return None
    if (
        not isinstance(model, str)
        or not model
        or position is None
        or direction is None
        or None in limits
    ):
        return None
    return Thruster(
        model=model,
        position=position,
        direction=direction,
        max_forward=limits[0],
        max_reverse=limits[1],
        preset=uses_preset,
        dead_band=dead_band,
    )


def _fin(raw: object, i: int, problems: list[str]) -> Fin | None:
    if not isinstance(raw, dict):
        problems.append(f"fins[{i}] must be a mapping with position, normal, area, span")
        return None
    position = _vector(raw.get("position"), f"fins[{i}].position", problems)
    normal = _vector(raw.get("normal"), f"fins[{i}].normal", problems)
    area = _positive(raw.get("area"), f"fins[{i}].area", problems)
    span = _positive(raw.get("span"), f"fins[{i}].span", problems)
    control = raw.get("control", False)
    if not isinstance(control, bool):
        problems.append(f"fins[{i}].control must be true or false")
        return None
    if normal is not None:
        norm = np.linalg.norm(normal)
        if norm < 1e-9:
            problems.append(f"fins[{i}].normal must not be zero")
            return None
        normal = normal / norm
    if any(v is None for v in (position, normal, area, span)):
        return None
    max_angle = raw.get("max_angle", 20.0)
    if (
        isinstance(max_angle, bool)
        or not isinstance(max_angle, (int, float))
        or not (0 < max_angle <= 45)
    ):
        problems.append(f"fins[{i}].max_angle must be in degrees, 0-45, got {max_angle!r}")
        return None
    return Fin(
        str(raw.get("name", f"fin{i}")),
        position,
        normal,
        area,
        span,
        control,
        float(np.radians(max_angle)),
    )


def _measured(raw: object, problems: list[str]) -> Measured:
    if raw is None:
        return Measured()
    if not isinstance(raw, dict):
        problems.append("'measured' must be a mapping (r_g, inertia_g, volume)")
        return Measured()
    r_g = _vector(raw["r_g"], "measured.r_g", problems) if "r_g" in raw else None
    inertia = None
    if "inertia_g" in raw:
        value = np.asarray(raw["inertia_g"], dtype=float) if _numeric(raw["inertia_g"]) else None
        if value is not None and value.shape == (3,):
            value = np.diag(value)
        if value is None or value.shape != (3, 3) or np.any(np.diag(value) <= 0):
            problems.append(
                "'measured.inertia_g' must be [Ixx, Iyy, Izz] or a 3x3 matrix, positive diagonal"
            )
        elif not np.allclose(value, value.T):
            problems.append("'measured.inertia_g' must be symmetric")
        else:
            inertia = value
    volume = None
    if "volume" in raw:
        volume = _positive(raw["volume"], "measured.volume", problems)
    return Measured(r_g=r_g, inertia_g=inertia, volume=volume)


def _numeric(value: object) -> bool:
    try:
        np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return False
    return True


def _calibration(raw: object, problems: list[str]) -> Calibration:
    if raw is None:
        return Calibration()
    if not isinstance(raw, dict):
        problems.append("'calibration' must be a mapping (written by robomodel calibrate)")
        return Calibration()
    values: dict = {}
    for key in ("damping", "added_mass", "damping_sigma", "added_mass_sigma"):
        if key in raw:
            arr = np.asarray(raw[key], dtype=float) if _numeric(raw[key]) else None
            positive = key.endswith("sigma") or (arr is not None and np.all(arr > 0))
            if arr is None or arr.shape != (6,) or not positive or np.any(arr < 0):
                problems.append(f"'calibration.{key}' must be 6 numbers (scales > 0)")
            else:
                values[key] = arr
    for key in (
        "fin_lift",
        "body_lift",
        "thrust",
        "fin_lift_sigma",
        "body_lift_sigma",
        "thrust_sigma",
    ):
        if key in raw:
            v = raw[key]
            ok = isinstance(v, (int, float)) and not isinstance(v, bool)
            if not ok or v < 0 or (not key.endswith("sigma") and v == 0):
                problems.append(f"'calibration.{key}' must be a positive number")
            else:
                values[key] = float(v)
    if "note" in raw:
        values["note"] = str(raw["note"])
    return Calibration(**values)
