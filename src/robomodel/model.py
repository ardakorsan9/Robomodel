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
from robomodel.frame import frame_added_mass, frame_damping
from robomodel.geometry import GeometryError, build_mesh, tilt_from_body_x
from robomodel.hydrostatics import Hydrostatics, hydrostatics
from robomodel.multihull import combine_damping, split_bodies
from robomodel.parameter import Parameter, Source
from robomodel.rigid_body import (
    MassProperties,
    mass_properties,
    rigid_body_mass_matrix,
    velocity_transform,
)
from robomodel.sections import Sections, slice_hull, voxel_sections
from robomodel.thrusters import allocation_matrix, thrust_rel_sigma
from robomodel.vehicle import Calibration, Vehicle
from robomodel.voxel import VoxelHull, voxelize

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
    r_hydro: np.ndarray = field(default_factory=lambda: np.zeros(3))  # hull centre, m
    warnings: list[str] = field(default_factory=list)
    floating: object | None = None  # surface.Floating for surface vessels
    hover: object | None = None  # aerial.Hover for drones

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
            fin_max=self.fins.max_angles,
            current_ned=np.asarray(self.vehicle.current, dtype=float),
            r_hydro=np.asarray(self.r_hydro, dtype=float),
            kind=self.vehicle.category,
            thrust_max=np.array([t.max_forward for t in self.vehicle.thrusters]),
            **self._surface_terms(),
        )

    def _surface_terms(self) -> dict:
        fl = self.floating
        if fl is None:
            return {}
        return {
            "g_matrix": fl.g_matrix,
            "eta_rest": fl.eta_rest,
            "roll_table": np.vstack([fl.gz_angle, fl.roll_moment]),
        }


def build_model(vehicle: Vehicle) -> VehicleModel:
    hull = build_mesh(vehicle.geometry)
    props = mass_properties(vehicle, hull)
    if vehicle.category == "surface":
        return _build_surface(vehicle, hull, props)
    if vehicle.category == "aerial":
        return _build_aerial(vehicle, hull, props)
    sections = slice_hull(hull)
    statics = hydrostatics(hull, props.mass, vehicle.water_density, vehicle.measured.volume)
    rho = vehicle.water_density
    hydro = _hydrodynamics(vehicle.geometry, hull, sections, rho)
    damping = hydro.damping
    fins = fin_model(vehicle.fins, rho)
    model = VehicleModel(
        vehicle=vehicle,
        hull=hull,
        sections=sections,
        mass=props,
        statics=statics,
        m_rb=rigid_body_mass_matrix(props),
        m_a=_with_fins(hydro.m_a, fins, vehicle),
        damping=damping,
        b_thr=allocation_matrix(vehicle.thrusters),
        fins=fins,
        body_lift=hydro.body_lift,
        r_hydro=hydro.centre,
        warnings=_warnings(vehicle, props, statics, hull, hydro.frame),
    )
    if not vehicle.calibration.is_identity:
        model = apply_calibration(model, vehicle.calibration)
    return model


@dataclass(frozen=True)
class _Hydro:
    m_a: Parameter  # about the body origin
    damping: Damping  # at `centre`
    body_lift: Parameter | None
    centre: np.ndarray
    frame: bool


def _with_fins(m_a: Parameter, fins: FinModel, vehicle: Vehicle) -> Parameter:
    if not vehicle.fins:
        return m_a
    # fin added mass adds to the hull's, with the hull's uncertainty
    return Parameter(
        m_a.value + fins.m_a,
        m_a.sigma + 0.3 * np.abs(fins.m_a),
        m_a.source,
        m_a.unit,
        m_a.note + "; fins as flat plates",
    )


def _hydrodynamics(g, hull, sections: Sections, rho: float) -> _Hydro:
    """Added mass, damping and body lift of a hull moving in infinite fluid."""
    frame = _uses_frame_method(g, sections, hull)
    # Hull hydrodynamics are computed about the hull's own centre, then moved to the body
    # origin. The result does not depend on where the CAD file has its origin.
    centre = np.asarray(hull.center_mass, dtype=float)
    local = replace(sections, x=sections.x - centre[0])
    from robomodel import bem

    method = g.added_mass
    exact = g.primitive in ("sphere", "ellipsoid")
    if method == "auto":
        # Slender hulls: strip theory, validated on REMUS. Box-like / odd hulls, where strip
        # theory and the bounding ellipsoid are weak: BEM if installed.
        if exact:
            method = "exact"
        elif frame:
            method = "bem" if bem.available() and _bem_affordable(hull) else "frame"
        else:
            method = "strip"
    if method == "bem" and not bem.available():
        raise GeometryError(
            'added_mass: bem needs the optional packages: pip install "robomodel[bem]"'
        )
    bodies = split_bodies(hull) if isinstance(hull, trimesh.Trimesh) and not frame else [hull]
    multi = len(bodies) > 1
    if method == "exact" and g.primitive == "sphere":
        r = g.dims["radius"]
        m_a = ellipsoid_added_mass_parameter(r, r, r, rho)
    elif method == "exact":
        m_a = ellipsoid_added_mass_parameter(g.dims["a"], g.dims["b"], g.dims["c"], rho)
    elif method == "bem":
        m_a = bem.bem_parameter(hull, rho, centre)  # about the hull centre, like the others
    elif method == "frame":
        m_a = frame_added_mass(hull, rho)
    elif multi:  # each hull on its own, added up about the body origin
        total = None
        for body in bodies:
            c = np.asarray(body.center_mass)
            s = slice_hull(body)
            part = _move_parameter(
                strip_theory_parameter(replace(s, x=s.x - c[0]), float(body.volume), rho), c
            )
            total = part if total is None else _add(total, part)
        m_a = Parameter(
            total.value,
            total.sigma,
            total.source,
            total.unit,
            total.note + f"; {len(bodies)} separate hulls added up",
        )
    else:
        m_a = strip_theory_parameter(local, float(hull.volume), rho)
    if not (multi and method == "strip"):
        m_a = _move_parameter(m_a, centre)

    if frame:
        voxels = hull if isinstance(hull, VoxelHull) else voxelize(hull, max(hull.extents) / 100)
        damping = frame_damping(voxels, rho, centre)
    elif multi:
        parts = []
        for body in bodies:
            c = np.asarray(body.center_mass)
            s = slice_hull(body)
            parts.append((estimate_damping(replace(s, x=s.x - c[0]), rho), c))
        damping = combine_damping(parts, centre)
    else:
        damping = estimate_damping(local, rho)
    lift = None if frame else _body_lift_all(bodies, sections, hull, rho, centre)
    return _Hydro(m_a, damping, lift, centre, frame)


def _add(a: Parameter, b: Parameter) -> Parameter:
    return Parameter(a.value + b.value, a.sigma + b.sigma, b.source, b.unit, b.note)


def _half(p: Parameter | None, note: str) -> Parameter | None:
    if p is None:
        return None
    value = 0.5 * np.asarray(p.value)
    return Parameter(value, 0.5 * np.asarray(p.sigma), p.source, p.unit, _join(p.note, note))


def _join(a: str, b: str) -> str:
    return f"{a}; {b}" if a else b


# Linear heave / roll / pitch damping of a floating hull (wave radiation, bilge eddies) as a
# share of critical damping. Typical monohull values (Fossen 2021, ch. 7; Journee & Massie,
# Offshore Hydromechanics, ch. 6), assumed; roll is lightly damped without bilge keels.
SURFACE_DAMPING_RATIO = {2: 0.2, 3: 0.05, 4: 0.2}
SURFACE_FROUDE_LIMIT = 0.3  # above this the wave-making resistance is not in the model


def _build_surface(vehicle: Vehicle, hull, props: MassProperties) -> VehicleModel:
    """Surface vessel: float the hull, then treat the wet part and its mirror image in the
    water surface as one body in infinite fluid and take half (see surface.double_body)."""
    from robomodel.surface import double_body, floating

    rho = vehicle.water_density
    fl = floating(hull, vehicle.mass, np.asarray(props.r_g.value), rho)
    wet = double_body(hull, fl)
    sections = slice_hull(wet)
    g = replace(
        vehicle.geometry,
        primitive=None,
        assembly=vehicle.geometry.assembly and isinstance(wet, VoxelHull),
    )
    hydro = _hydrodynamics(g, wet, sections, rho)
    note = "wet hull + mirror image in the water surface, halved"
    m_a = _half(hydro.m_a, note)
    fins = fin_model(vehicle.fins, rho)
    m_a = _with_fins(m_a, fins, vehicle)
    m_rb = rigid_body_mass_matrix(props)

    d = hydro.damping
    quad = _half(d.quadratic, note + "; no wave-making resistance")
    lin_value = 0.5 * np.asarray(d.linear.value, dtype=float)
    m_total = m_rb + np.asarray(m_a.value)
    for i, zeta in SURFACE_DAMPING_RATIO.items():
        k = fl.g_matrix[i, i]
        if k > 0:
            lin_value[i] = 2 * zeta * np.sqrt(m_total[i, i] * k)
    linear = Parameter(
        lin_value,
        lin_value * 1.0,
        Source.ASSUMED,
        d.linear.unit,
        "surge/sway/yaw: quadratic x reference speed; heave/roll/pitch: typical share of "
        f"critical damping {SURFACE_DAMPING_RATIO[2]}/{SURFACE_DAMPING_RATIO[3]}/"
        f"{SURFACE_DAMPING_RATIO[4]}, assumed",
    )
    damping = replace(d, linear=linear, quadratic=quad)

    weight = props.mass.value * 9.80665
    rel = props.mass.sigma / props.mass.value
    statics = Hydrostatics(
        volume=Parameter.relative(fl.volume, rel, Source.CAD, "m^3", "floating: volume = m / rho"),
        r_b=Parameter(
            np.asarray(fl.r_b), np.full(3, 0.005 * float(max(hull.extents))), Source.CAD, "m"
        ),
        weight=Parameter(weight, props.mass.sigma * 9.80665, props.mass.source, "N"),
        buoyancy=Parameter(weight, props.mass.sigma * 9.80665, Source.CAD, "N", "floats: B = W"),
    )
    model = VehicleModel(
        vehicle=vehicle,
        hull=hull,
        sections=sections,
        mass=props,
        statics=statics,
        m_rb=m_rb,
        m_a=m_a,
        damping=damping,
        b_thr=allocation_matrix(vehicle.thrusters),
        fins=fins,
        body_lift=_half(hydro.body_lift, note),
        r_hydro=hydro.centre,
        warnings=_surface_warnings(vehicle, props, fl, hull, sections),
        floating=fl,
    )
    if not vehicle.calibration.is_identity:
        model = apply_calibration(model, vehicle.calibration)
    return model


def _build_aerial(vehicle: Vehicle, hull, props: MassProperties) -> VehicleModel:
    """Multirotor: rigid body + rotors, frame drag in air, rotor drag; no buoyancy."""
    from robomodel.aerial import hover, rotor_drag

    rho = vehicle.water_density  # air
    m_rb = rigid_body_mass_matrix(props)
    b_thr = allocation_matrix(vehicle.thrusters)
    f_max = np.array([t.max_forward for t in vehicle.thrusters])
    weight = props.mass.value * 9.80665
    r_g = np.asarray(props.r_g.value, dtype=float)
    hov = hover(b_thr, f_max, weight, r_g)

    centre = np.asarray(hull.center_mass, dtype=float)
    voxels = hull if isinstance(hull, VoxelHull) else voxelize(hull, max(hull.extents) / 100)
    frame = frame_damping(voxels, rho, centre)
    d_rot = rotor_drag(vehicle.thrusters, hov.thrust, rho)
    # Frame drag acts at the frame centre; rotor drag is put there too (moments from its
    # height above the CG are left out).
    lin = np.asarray(frame.linear.value, dtype=float) + d_rot
    linear = Parameter(
        lin,
        lin * 1.0,
        Source.ASSUMED,
        frame.linear.unit,
        "frame: quadratic x reference speed; rotors: half the momentum ram drag at hover, assumed",
    )
    quad = Parameter(
        np.asarray(frame.quadratic.value),
        np.asarray(frame.quadratic.sigma),
        Source.EMPIRICAL,
        frame.quadratic.unit,
        f"frame drag in air: C_D = {frame.axial_cd} on projected areas",
    )
    damping = replace(frame, linear=linear, quadratic=quad)
    volume = float(hull.volume)
    statics = Hydrostatics(
        volume=Parameter.relative(volume, 0.3, Source.CAD, "m^3", "not used in air"),
        r_b=Parameter(r_g.copy(), np.zeros(3), Source.CAD, "m", "no buoyancy in air"),
        weight=Parameter(weight, props.mass.sigma * 9.80665, props.mass.source, "N"),
        buoyancy=Parameter(0.0, 0.0, Source.CAD, "N", "air buoyancy (~0.1 % of weight) ignored"),
    )
    zero = np.zeros((6, 6))
    model = VehicleModel(
        vehicle=vehicle,
        hull=hull,
        sections=slice_hull(hull) if not isinstance(hull, VoxelHull) else voxel_sections(hull),
        mass=props,
        statics=statics,
        m_rb=m_rb,
        m_a=Parameter(zero, zero, Source.CAD, "kg, kg m^2", "added mass of air ignored"),
        damping=damping,
        b_thr=b_thr,
        fins=fin_model((), rho),
        body_lift=None,
        r_hydro=centre,
        warnings=_aerial_warnings(vehicle, props, hov, hull),
        hover=hov,
    )
    if not vehicle.calibration.is_identity:
        model = apply_calibration(model, vehicle.calibration)
    return model


def _aerial_warnings(vehicle, props, hov, hull) -> list[str]:
    from robomodel import mass_parts

    out = list(getattr(hull, "metadata", {}).get("warnings", []))
    out += mass_parts.warnings(props.distribution)
    extents = np.asarray(hull.extents)
    if extents.max() > 20 or extents.max() < 0.02:
        out.append(f"the drone is {extents.max():.3g} m across: check the file units (mm / m)")
    rotors = [t for t in vehicle.thrusters if t.torque != 0 or t.diameter > 0]
    if not vehicle.thrusters:
        out.append(
            "no rotors given: the model has no control inputs and hover cannot be checked; "
            "see the requirements for the thrust the drone needs"
        )
        return out
    if any(t.assumed_limit for t in vehicle.thrusters):
        out.append(
            f"maximum rotor thrust not given: each rotor is rated at "
            f"{vehicle.thrusters[0].max_forward:.3g} N (2 x weight / number of rotors, the "
            "recommended value); the command scale follows from it, hover does not"
        )
    if not hov.possible:
        out.append(
            "the rotors cannot hold the drone level in hover: too little thrust, the centre of "
            "gravity outside the rotors' reach, or spin directions that tie yaw to roll/pitch "
            "(in an X or + layout, opposite rotors spin the same way): check max_thrust, "
            "positions, spins and CG"
        )
    elif hov.thrust_to_weight < 1.5:
        out.append(
            f"thrust-to-weight ratio is {hov.thrust_to_weight:.2f}: under 1.5 leaves little "
            "thrust for control and wind"
        )
    spins = [np.sign(t.torque) for t in rotors]
    if spins and abs(sum(spins)) > 0 and len(spins) % 2 == 0:
        out.append(
            f"{spins.count(1)} rotors spin cw and {spins.count(-1)} ccw: yaw is unbalanced, "
            "check the spin directions"
        )
    if not vehicle.parts and props.distribution is None:  # an assembly spreads it by part
        out.append(
            "no parts listed: the hull mass is spread over the CAD volume, which puts the "
            "inertia in the wrong place; list the battery and give motor_mass for the rotors"
        )
    if any(t.diameter <= 0 for t in rotors):
        out.append("some rotors have no diameter: their drag is left out")
    out.append(
        "rotor torque/thrust ratio, rotor drag and frame drag are estimates: measure thrust "
        "and torque on a thrust stand and fit with robomodel calibrate"
    )
    out.append(
        "not modelled: thrust falling with airspeed (so climb and top speeds come out high), "
        "motor response time, gyroscopic rotor effects, ground effect"
    )
    return out


def _surface_warnings(vehicle, props, fl, hull, sections) -> list[str]:
    from robomodel import mass_parts

    out = list(getattr(hull, "metadata", {}).get("warnings", []))
    out += mass_parts.warnings(props.distribution)
    extents = np.asarray(hull.extents)
    if extents.max() > 400 or extents.max() < 0.05:
        out.append(f"the vessel is {extents.max():.3g} m long: check the file units (mm / m)")
    if extents[0] < 0.9 * extents.max():
        axis = "xyz"[int(np.argmax(extents))]
        out.append(
            f"the hull is longest along body {axis}, not x (forward): rotate the CAD export so "
            "the bow points along +x"
        )
    if vehicle.measured.volume is not None:
        out.append("measured.volume is ignored for a surface vessel: it floats at V = m / rho")
    if vehicle.measured.r_g is None:
        spread = "spread over its skin" if "skin" in props.r_g.note else "uniform"
        out.append(
            "stability (GM) depends on the height of the centre of gravity: list heavy parts "
            f"or measure it (inclining test, measured.r_g); here the hull mass is {spread}"
        )
    if abs(np.degrees(fl.trim)) > 3:
        out.append(
            f"the vessel trims {np.degrees(fl.trim):+.1f} deg at rest (+ = bow up): move weight "
            "or check the CG; added mass assumes a nearly level hull"
        )
    if abs(np.degrees(fl.heel)) > 1:
        out.append(
            f"the vessel heels {np.degrees(fl.heel):+.1f} deg at rest: CG off the centreline?"
        )
    if fl.freeboard < 0.05 * extents[0]:
        out.append(f"small freeboard ({100 * fl.freeboard:.1f} cm): waves will wash over the deck")
    v_max = SURFACE_FROUDE_LIMIT * np.sqrt(9.80665 * sections.length)
    out.append(
        f"resistance is valid up to about {v_max:.2g} m/s (Froude 0.3); faster, wave-making "
        "and planing are not modelled and the resistance comes out too low"
    )
    if not vehicle.thrusters:
        out.append("no thrusters: the model has no control inputs")
    elif any(t.preset for t in vehicle.thrusters):
        out.append("thrust limits come from a preset; measure them or give max_forward/max_reverse")
    return out


def _count_bodies(hull, min_share: float = 0.05) -> int:
    """Separate closed bodies holding at least min_share of the volume (meshes only)."""
    if not isinstance(hull, trimesh.Trimesh) or len(hull.faces) > 2_000_000:
        return 1
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    nf, nv = len(hull.faces), len(hull.vertices)
    rows = np.repeat(np.arange(nf), 3)
    graph = coo_matrix((np.ones(3 * nf), (rows, hull.faces.ravel() + nf)), shape=(nf + nv,) * 2)
    _, labels = connected_components(graph, directed=False)
    tri = hull.vertices[hull.faces]
    vol = np.einsum("ij,ij->i", tri[:, 0], np.cross(tri[:, 1], tri[:, 2])) / 6
    per_body = np.abs(np.bincount(labels[:nf], weights=vol))
    return int(np.sum(per_body >= min_share * per_body.sum()))


def _move_parameter(param: Parameter, r) -> Parameter:
    """A 6x6 matrix parameter about the hull centre r, expressed about the body origin."""
    if not np.any(np.abs(r) > 1e-12):
        return param
    h = velocity_transform(r)
    value = h.T @ np.asarray(param.value) @ h
    sigma = np.abs(h).T @ np.asarray(param.sigma) @ np.abs(h)
    return Parameter(value, sigma, param.source, param.unit, param.note)


def _body_lift_all(bodies, sections, hull, rho, centre) -> Parameter | None:
    """Hoerner body lift of one hull, or the sum over separate hulls."""
    if len(bodies) == 1:
        return body_lift(sections, float(hull.bounds[1, 0]), rho, centre)
    total = None
    for body in bodies:
        lift = body_lift(slice_hull(body), float(body.bounds[1, 0]), rho, body.center_mass)
        if lift is None:
            continue
        total = (
            lift
            if total is None
            else Parameter(
                total.value + lift.value,
                total.sigma + lift.sigma,
                lift.source,
                lift.unit,
                lift.note,
            )
        )
    return total


def _bem_affordable(hull) -> bool:
    """Keep automatic BEM away from hulls whose voxel grid would be too big for memory."""
    from robomodel.voxel import VoxelHull

    return not isinstance(hull, VoxelHull) or hull.filled.size <= 30_000_000


FRAME_LD = 2.0  # below this length/diameter a hull is treated as box-like
SLENDER_RATIO = 2.0  # longest / middle size above this: the hull is clearly slender


def _uses_frame_method(geometry, sections: Sections, hull) -> bool:
    if geometry.hydrodynamics != "auto":
        return geometry.hydrodynamics == "frame"
    if geometry.assembly:
        return True
    d = 2 * np.sqrt(float(sections.area.max()) / np.pi)
    return sections.length / d < FRAME_LD and geometry.primitive not in ("sphere", "ellipsoid")


def _scaled(
    param: Parameter, scale, rel_sigma, note: str, source: Source = Source.CALIBRATED
) -> Parameter:
    """Multiply a parameter by scale factors (fitted, or a bound of the model range)."""
    value = np.asarray(param.value, dtype=float) * scale
    sigma = np.abs(value) * rel_sigma
    if np.ndim(value) == 0:
        value, sigma = float(value), float(sigma)
    return Parameter(value, sigma, source, param.unit, note)


def apply_calibration(
    model: VehicleModel, cal: Calibration, source: Source = Source.CALIBRATED
) -> VehicleModel:
    """Apply scale factors from test data (or a range bound). No geometry is recomputed."""
    if source is Source.CALIBRATED:
        note = "calibrated to test data" + (f" ({cal.note})" if cal.note else "")
    else:
        note = cal.note
    d = model.damping
    damping = replace(
        d,
        linear=_scaled(d.linear, cal.damping, cal.damping_sigma, note, source),
        quadratic=_scaled(d.quadratic, cal.damping, cal.damping_sigma, note, source),
    )
    m_a = model.m_a
    if not np.allclose(cal.added_mass, 1):
        diag = np.sqrt(cal.added_mass)  # scale rows and columns: diagonal x s, couplings x sqrt
        m_a = _scaled(
            m_a,
            np.outer(diag, diag),
            np.sqrt(np.add.outer(cal.added_mass_sigma, cal.added_mass_sigma) / 2),
            note,
            source,
        )
    fins = model.fins
    if cal.fin_lift != 1:
        fins = replace(
            fins,
            d_fin=_scaled(fins.d_fin, cal.fin_lift, cal.fin_lift_sigma, note, source),
            b_fin=fins.b_fin * cal.fin_lift,
        )
    lift = model.body_lift
    if lift is not None and cal.body_lift != 1:
        lift = _scaled(lift, cal.body_lift, cal.body_lift_sigma, note, source)
    return replace(
        model,
        damping=damping,
        m_a=m_a,
        fins=fins,
        body_lift=lift,
        b_thr=model.b_thr * cal.thrust,
        vehicle=replace(model.vehicle, calibration=cal),
    )


def _warnings(
    vehicle: Vehicle,
    props: MassProperties,
    statics: Hydrostatics,
    hull: trimesh.Trimesh,
    frame: bool = False,
) -> list[str]:
    from robomodel import mass_parts

    out = list(getattr(hull, "metadata", {}).get("warnings", []))
    out += mass_parts.warnings(props.distribution)
    extents = np.asarray(hull.extents)
    volume = float(statics.volume.value)
    density = vehicle.mass / volume if volume > 0 else np.inf
    if not 150 <= density <= 8000:
        out.append(
            f"mass / volume = {density:.3g} kg/m^3 is not plausible for a vehicle (water is "
            "1000): check the file units (mm / m) and the mass"
        )
    if extents.max() > 50 or extents.max() < 0.02:
        out.append(f"the vehicle is {extents.max():.3g} m long: check the file units (mm / m)")
    if not frame and not getattr(hull, "filled", None) is not None:
        tilt = tilt_from_body_x(hull)
        if 3 < tilt < 87:
            out.append(
                f"the hull's long axis is tilted {tilt:.0f} deg from body x: if the CAD is drawn "
                "at an angle, straighten it with geometry.rotate [roll, pitch, yaw] (deg)"
            )
    bodies = _count_bodies(hull)
    if bodies > 1:
        out.append(
            f"the hull has {bodies} separate bodies: each is computed on its own and added up; "
            "their interaction (shielding, gap flow) is not modelled"
        )
    # Slender methods need the long axis forward. Box-like ROVs are often a bit wider than
    # long, which is fine; a hull clearly slender along y or z was exported the wrong way.
    slender_elsewhere = extents.max() >= SLENDER_RATIO * np.median(extents)
    if extents[0] < 0.9 * extents.max() and (not frame or slender_elsewhere):
        axis = "xyz"[int(np.argmax(extents))]
        out.append(
            f"the hull is longest along body {axis}, not x (forward): rotate the CAD export so "
            "the nose points along +x, or added mass and damping will be wrong"
        )
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
        "floating": None
        if model.floating is None
        else {
            "eta_rest": model.floating.eta_rest.tolist(),
            "G": np.round(model.floating.g_matrix, 9).tolist(),
            "gz_angle_rad": model.floating.gz_angle.tolist(),
            "roll_restoring_Nm": model.floating.roll_moment.tolist(),
            "summary": floating_summary(model),
        },
        "hover": None
        if model.hover is None
        else {
            "thrust_N": np.asarray(model.hover.thrust).tolist(),
            "possible": bool(model.hover.possible),
            "thrust_to_weight": float(model.hover.thrust_to_weight),
            "summary": hover_summary(model),
        },
        "warnings": model.warnings,
        "checks": [
            {"name": c.name, "status": c.status, "detail": c.detail}
            for c in __import__("robomodel.checks", fromlist=["run_checks"]).run_checks(
                model.dynamics()
            )
        ],
        "provenance": __import__("robomodel.provenance", fromlist=["provenance"]).provenance(
            model.vehicle
        ),
    }


def floating_summary(model: VehicleModel) -> dict[str, str]:
    """Plain-text hydrostatics of a surface vessel, for reports."""
    from robomodel.checks import _vanishing_angle
    from robomodel.surface import natural_periods

    fl = model.floating
    periods = natural_periods(model.m_rb + np.asarray(model.m_a.value), fl.g_matrix)
    rel = fl.gz_angle - fl.heel
    gz = {a: float(np.interp(np.radians(a), rel, fl.gz)) for a in (10, 20, 30, 45, 60)}
    return {
        "origin depth z0": f"{fl.depth:+.4g} m (body origin below the water surface)",
        "draft": f"{fl.draft:.4g} m (deepest point)",
        "freeboard": f"{fl.freeboard:.4g} m (highest point above water)",
        "trim, heel": f"{np.degrees(fl.trim):+.2f} deg (bow up), {np.degrees(fl.heel):+.2f} deg",
        "waterplane": f"{fl.waterplane_area:.4g} m^2",
        "GM_T, GM_L": f"{fl.gm_t:.4g} m, {fl.gm_l:.4g} m",
        "GZ": ", ".join(f"{a} deg {v:.3g} m" for a, v in gz.items()),
        "stable up to": f"{_vanishing_angle(rel, fl.roll_moment):.0f} deg heel",
        "periods": ", ".join(f"{k} {v:.3g} s" for k, v in periods.items()) + " (undamped)",
    }


def hover_summary(model: VehicleModel) -> dict[str, str]:
    """Plain-text hover numbers of a drone, for reports."""
    hov = model.hover
    f_max = np.array([t.max_forward for t in model.vehicle.thrusters])
    if not hov.possible:
        return {"hover": "NOT possible (see checks)", "T/W": f"{hov.thrust_to_weight:.2f}"}
    return {
        "rotor thrust": ", ".join(f"{f:.3g}" for f in hov.thrust) + " N",
        "commands": ", ".join(f"{c:.3f}" for c in hov.commands(f_max)) + " (thrust ~ command^2)",
        "T/W": f"{hov.thrust_to_weight:.2f} (full thrust / weight)",
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

    if m.distribution is not None:
        from robomodel.mass_parts import summary as parts_summary

        lines += ["Mass over the CAD parts (material guessed from the part names)"]
        lines += ["  " + x for x in parts_summary(m.distribution)] + [""]
    lines.append("Mass and statics")
    row("mass", m.mass)
    row("volume", s.volume)
    row("weight W", s.weight)
    row("buoyancy B", s.buoyancy)
    row("CG r_g", m.r_g)
    row("CB r_b", s.r_b)
    row("I_g diag", m.inertia_g, diag=True)
    if model.floating is not None:
        lines += ["", "Floating (surface vessel; eta at rest = [0, 0, z0, heel, trim, 0])"]
        lines += [f"  {k:<16}{v}" for k, v in floating_summary(model).items()]
    if model.hover is not None:
        lines += ["", "Hover (level, still air)"]
        lines += [f"  {k:<16}{v}" for k, v in hover_summary(model).items()]
    heading = "Aerodynamics" if model.vehicle.category == "aerial" else "Hydrodynamics"
    lines += ["", f"{heading} (order: surge, sway, heave, roll, pitch, yaw)"]
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
    from robomodel.checks import run_checks, summary

    lines += ["", "Checks:", summary(run_checks(model.dynamics()))]
    if model.warnings:
        lines += ["", "Warnings:"] + [f"  ! {w}" for w in model.warnings]
    return "\n".join(lines)
