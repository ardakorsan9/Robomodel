"""Write a model as a Gazebo (gz-sim, ROS 2 Jazzy: Harmonic) model and test world.

Frames: robomodel uses Fossen's body frame (x forward, y right, z down, FRD) and NED; Gazebo
uses x forward, y left, z up (FLU) and ENU. A generalised vector maps with
T = diag(1, -1, -1, 1, -1, -1), so a coefficient c_ij (or c_ijk with |nu_k|) becomes
T_i T_j c_ij in Gazebo; |nu_k| does not change sign.

What goes where:
- mass, CG, inertia            -> <inertial> of base_link
- buoyancy (volume, CB)        -> a sphere collision of the same volume at the CB, used by
                                  the world's Buoyancy system (volume is all it needs)
- added mass, damping, fin and body lift (passive) -> Hydrodynamics plugin coefficients
- thrusters                    -> one Thruster plugin each, command = thrust in newtons on
                                  /model/<name>/joint/<thruster>_joint/cmd_thrust
Control fins are passive here (their lift is in the hydrodynamics); deflection commands are
not exported.
"""

from __future__ import annotations

from pathlib import Path
from xml.sax.saxutils import escape

import numpy as np

from robomodel.matlab_export import matlab_name
from robomodel.model import VehicleModel
from robomodel.provenance import provenance
from robomodel.rigid_body import velocity_transform

T6 = np.array([1, -1, -1, 1, -1, -1.0])
T3 = T6[:3]
VEL = "UVWPQR"
FORCE = "xyzkmn"
MIN_COEFFICIENT = 1e-9


def _flu(v) -> np.ndarray:
    return np.asarray(v, dtype=float) * T3


def _rpy_from_x_axis(direction: np.ndarray) -> tuple[float, float, float]:
    """Roll, pitch, yaw that turn the link x axis onto `direction` (FLU)."""
    d = direction / np.linalg.norm(direction)
    yaw = np.arctan2(d[1], d[0])
    pitch = -np.arcsin(np.clip(d[2], -1, 1))
    return 0.0, float(pitch), float(yaw)


def hydro_coefficients(model: VehicleModel) -> dict[str, float]:
    """Gazebo Hydrodynamics plugin coefficients (Fossen SNAME names, FLU frame), about the
    hull centre r_hydro: the Gazebo link frame is put there (see write_gazebo), so the hull
    drag stays diagonal."""
    dyn = model.dynamics()
    out: dict[str, float] = {}
    h_inv = np.linalg.inv(velocity_transform(dyn.r_hydro))
    m_a = h_inv.T @ dyn.m_a @ h_inv  # back from the body origin to the hull centre
    d_fin = h_inv.T @ dyn.d_fin @ h_inv
    for i in range(6):
        for j in range(6):
            c = -m_a[i, j] * T6[i] * T6[j]
            if abs(c) > MIN_COEFFICIENT:
                out[f"{FORCE[i]}Dot{VEL[j]}"] = c
    for i in range(6):
        if dyn.d_lin[i] > MIN_COEFFICIENT:
            out[f"{FORCE[i]}{VEL[i]}"] = -dyn.d_lin[i]
    quad = {}
    for i in range(6):
        quad[(i, i, i)] = quad.get((i, i, i), 0.0) - dyn.d_quad[i]
    # Fin and body lift: tau_i = -|u| d_fin[i, j] nu_j  ->  coefficient of nu_j |u|
    for i in range(6):
        for j in range(6):
            if abs(d_fin[i, j]) > MIN_COEFFICIENT:
                quad[(i, j, 0)] = quad.get((i, j, 0), 0.0) - d_fin[i, j]
    for (i, j, k), c in quad.items():
        c = c * T6[i] * T6[j]
        if abs(c) > MIN_COEFFICIENT:
            out[f"{FORCE[i]}{VEL[j]}abs{VEL[k]}"] = c
    return out


def _f(x: float) -> str:
    return f"{x:.9g}"


def write_gazebo(
    model: VehicleModel, out_dir: str | Path, disable_coriolis: bool = False
) -> list[Path]:
    out = Path(out_dir)
    name = matlab_name(model.vehicle.name)
    model_dir = out / name
    (model_dir / "meshes").mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    dyn = model.dynamics()
    v = model.vehicle

    # The Gazebo link frame sits at the hull centre (where the drag coefficients apply);
    # every position is shifted by it.
    centre = np.asarray(dyn.r_hydro, dtype=float)
    mass = float(model.mass.mass.value)
    r_g = _flu(np.asarray(model.mass.r_g.value) - centre)
    inertia = np.diag(T3) @ np.asarray(model.mass.inertia_g.value) @ np.diag(T3)
    volume = float(model.statics.volume.value)
    r_b = _flu(np.asarray(model.statics.r_b.value) - centre)
    radius = (3 * volume / (4 * np.pi)) ** (1 / 3)

    visual = ""
    if hasattr(model.hull, "export"):  # a triangle mesh (not a voxel hull)
        mesh = model.hull.copy()
        mesh.apply_translation(-centre)
        mesh.apply_transform(np.diag([1.0, -1.0, -1.0, 1.0]))
        mesh_path = model_dir / "meshes" / "hull.stl"
        mesh.export(mesh_path)
        written.append(mesh_path)
        visual = f"""      <visual name="hull_visual">
        <geometry><mesh><uri>model://{name}/meshes/hull.stl</uri></mesh></geometry>
        <material><diffuse>0.9 0.6 0.1 1</diffuse></material>
      </visual>"""
    else:
        e = model.hull.extents
        c = _flu(model.hull.bounds.mean(axis=0) - centre)
        visual = f"""      <visual name="hull_visual">
        <pose>{_f(c[0])} {_f(c[1])} {_f(c[2])} 0 0 0</pose>
        <geometry><box><size>{_f(e[0])} {_f(e[1])} {_f(e[2])}</size></box></geometry>
        <material><diffuse>0.9 0.6 0.1 0.6</diffuse></material>
      </visual>"""

    coeffs = hydro_coefficients(model)
    coeff_xml = "\n".join(f"      <{k}>{_f(c)}</{k}>" for k, c in sorted(coeffs.items()))

    thruster_xml = []
    for i, t in enumerate(v.thrusters):
        tname = f"thruster{i}"
        pos = _flu(np.asarray(t.position) - centre)
        roll, pitch, yaw = _rpy_from_x_axis(_flu(t.direction))
        scale = v.calibration.thrust
        thruster_xml.append(f"""    <link name="{tname}">
      <pose>{_f(pos[0])} {_f(pos[1])} {_f(pos[2])} {_f(roll)} {_f(pitch)} {_f(yaw)}</pose>
      <inertial><mass>0.001</mass><inertia><ixx>1e-6</ixx><iyy>1e-6</iyy><izz>1e-6</izz>
        <ixy>0</ixy><ixz>0</ixz><iyz>0</iyz></inertia></inertial>
    </link>
    <joint name="{tname}_joint" type="revolute">
      <parent>base_link</parent><child>{tname}</child>
      <axis><xyz>1 0 0</xyz><limit><lower>-1e12</lower><upper>1e12</upper></limit></axis>
    </joint>
    <plugin filename="gz-sim-thruster-system" name="gz::sim::systems::Thruster">
      <namespace>{name}</namespace>
      <joint_name>{tname}_joint</joint_name>
      <use_angvel_cmd>false</use_angvel_cmd>
      <thrust_coefficient>0.004</thrust_coefficient>
      <fluid_density>{_f(v.water_density)}</fluid_density>
      <propeller_diameter>0.08</propeller_diameter>
      <max_thrust_cmd>{_f(t.max_forward * scale)}</max_thrust_cmd>
      <min_thrust_cmd>{_f(-t.max_reverse * scale)}</min_thrust_cmd>
    </plugin>""")

    stamp = provenance(v)
    sdf = f"""<?xml version="1.0"?>
<!-- Generated by robomodel {stamp["robomodel_version"]} on {stamp["created_utc"]}
     from {escape(str(stamp.get("vehicle_file", v.name)))}.
     Gazebo frame: x forward, y left, z up (converted from Fossen FRD). -->
<sdf version="1.9">
  <model name="{name}">
    <link name="base_link">
      <inertial>
        <pose>{_f(r_g[0])} {_f(r_g[1])} {_f(r_g[2])} 0 0 0</pose>
        <mass>{_f(mass)}</mass>
        <inertia>
          <ixx>{_f(inertia[0, 0])}</ixx><iyy>{_f(inertia[1, 1])}</iyy><izz>{_f(inertia[2, 2])}</izz>
          <ixy>{_f(inertia[0, 1])}</ixy><ixz>{_f(inertia[0, 2])}</ixz><iyz>{_f(inertia[1, 2])}</iyz>
        </inertia>
      </inertial>
      <!-- Buoyancy volume: a sphere with the vehicle's displaced volume, at its centre of
           buoyancy (V = {_f(volume)} m^3). -->
      <collision name="buoyancy_volume">
        <pose>{_f(r_b[0])} {_f(r_b[1])} {_f(r_b[2])} 0 0 0</pose>
        <geometry><sphere><radius>{_f(radius)}</radius></sphere></geometry>
      </collision>
{visual}
    </link>
{chr(10).join(thruster_xml)}
    <plugin filename="gz-sim-hydrodynamics-system" name="gz::sim::systems::Hydrodynamics">
      <link_name>base_link</link_name>
      <water_density>{_f(v.water_density)}</water_density>
      <disable_coriolis>{str(disable_coriolis).lower()}</disable_coriolis>
{coeff_xml}
    </plugin>
  </model>
</sdf>
"""
    (model_dir / "model.sdf").write_text(sdf, encoding="utf-8")
    written.append(model_dir / "model.sdf")
    (model_dir / "model.config").write_text(
        f"""<?xml version="1.0"?>
<model>
  <name>{name}</name>
  <version>1.0</version>
  <sdf version="1.9">model.sdf</sdf>
  <description>Generated by robomodel from {escape(v.name)}</description>
</model>
""",
        encoding="utf-8",
    )
    written.append(model_dir / "model.config")

    current = _flu(v.current)  # NED -> ENU is (N, E, D) -> (E, N, -D); see note below
    world = f"""<?xml version="1.0"?>
<!-- Test world for {name}: water everywhere, no ground. Run with
     GZ_SIM_RESOURCE_PATH={out.resolve()} gz sim -r world.sdf -->
<sdf version="1.9">
  <world name="water">
    <physics name="1ms" type="ignored"><max_step_size>0.001</max_step_size>
      <real_time_factor>1.0</real_time_factor></physics>
    <gravity>0 0 -9.80665</gravity>
    <plugin filename="gz-sim-physics-system" name="gz::sim::systems::Physics"/>
    <plugin filename="gz-sim-user-commands-system" name="gz::sim::systems::UserCommands"/>
    <plugin filename="gz-sim-scene-broadcaster-system" name="gz::sim::systems::SceneBroadcaster"/>
    <plugin filename="gz-sim-pose-publisher-system" name="gz::sim::systems::PosePublisher"/>
    <plugin filename="gz-sim-buoyancy-system" name="gz::sim::systems::Buoyancy">
      <uniform_fluid_density>{_f(v.water_density)}</uniform_fluid_density>
    </plugin>
    <include>
      <uri>model://{name}</uri>
      <pose>0 0 -5 0 0 0</pose>
    </include>
  </world>
</sdf>
"""
    del current  # current is not exported yet: gz expects it on a topic (see README)
    (out / "world.sdf").write_text(world, encoding="utf-8")
    written.append(out / "world.sdf")
    added = np.diag(dyn.m_a)[:3].max() if dyn.m_a.size else 0.0
    notes = [
        f"# Gazebo model: {name}",
        "",
        "Run (from this folder):",
        "",
        "```bash",
        "GZ_SIM_RESOURCE_PATH=$PWD gz sim -r world.sdf",
        "```",
        "",
        "Thrust command (newtons) for thruster i:",
        "",
        "```bash",
        f"gz topic -t /model/{name}/joint/thruster0_joint/cmd_thrust "
        "-m gz.msgs.Double -p 'data: 10'",
        "```",
        "",
        "Notes:",
        "- Control fin deflections are not exported; fin lift is passive.",
        "- Water current is not exported.",
    ]
    if added > mass:
        notes.append(
            f"- Added mass ({added:.3g} kg) exceeds the vehicle mass ({mass:.3g} kg): "
            "Gazebo's explicit added-mass term can become unstable. If the model "
            "explodes, lower the step size or use --disable-coriolis."
        )
    (out / "README.md").write_text("\n".join(notes) + "\n", encoding="utf-8")
    written.append(out / "README.md")
    return written
