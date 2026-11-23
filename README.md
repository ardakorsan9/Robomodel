# robomodel

**CAD file in, ready-to-use mathematical model out.** robomodel turns a vehicle's CAD (STL or
STEP) and a few numbers (mass, thrusters, fins) into a 6-degree-of-freedom Fossen model:

```
M ν̇ + C(ν) ν + D(ν) ν + g(η) = τ,   M = M_RB + M_A
```

with MATLAB, Simulink and Gazebo files, automatic sanity checks, and an honest statement of how
far each number can be trusted. Supported: underwater vehicles (torpedo AUVs and open-frame
ROVs), surface vessels (boats, USVs), multirotor drones and wheeled / tracked ground robots.
Jointed robots (arms) on the web page and the command line; legged robots on the
command line.

Türkçe kılavuz: [docs/KILAVUZ.md](docs/KILAVUZ.md)

## What comes out of the CAD, and how good it is

Some terms follow from geometry exactly; others (mostly damping) depend on turbulence and cannot.
robomodel computes everything it can, and for the rest gives a best estimate **plus low/high
bounds** taken from vehicles we validated against measurements (REMUS 100, BlueROV2 Heavy).

| Term | From CAD | Accuracy shown in validation |
|---|---|---|
| Volume, buoyancy, centre of buoyancy | exact for a closed CAD | 1–4 % (REMUS); open frames: weigh in water |
| Mass properties (CG, inertia) | from the hull + heavy parts you list | as good as the parts list; measured values override |
| Added mass M_A | Lamb (ellipsoids, exact), strip theory, enclosing ellipsoid (frames) | 7–13 % vs theory for crossflow terms |
| Coriolis C(ν) | exact from M | – |
| Restoring forces g(η) | exact | – |
| Fin and hull lift | Helmbold / Hoerner | 0–11 % vs published formulas |
| Thruster allocation | exact from positions and directions | – |
| **Damping D(ν)** | handbook estimate only | **1.6–10× too low** in two vehicles: use the range or calibrate |

So: geometry-driven behaviour (stability, trim, how thrusters act) is reliable from CAD alone;
speeds and turning rates need either the low/high range in simulation or one pool test and
`robomodel calibrate`. Full reports: [REMUS 100](docs/dogrulama_remus100.md),
[BlueROV2 Heavy](docs/dogrulama_bluerov2.md), [calibration](docs/dogrulama_kalibrasyon.md).

## Where the mass sits

The scale gives the total mass, not where it is. robomodel spreads it as well as the CAD allows:

- **STEP assembly (recommended):** every part is kept separate and gets the density of its
  likely material from its name (`LiPo_3S` → battery, `M3-SCREW` → steel, `foam_float` → foam,
  English and Turkish keywords); the part masses are then scaled together to the weighed
  total, so only the density *ratios* matter. If the parts would weigh far more or less than
  the scale says, the report warns (wrong units, wrong mass, or parts drawn solid that are
  hollow). On the BlueROV2 R4 STEP (459 parts) 56 % of the mass got its material from a part
  name, and the guessed materials gave 12 kg against the vehicle's ~10 kg.
- **Single body (STL):** uniform density (or skin, for boats), as before.
- **Better still:** list heavy parts, or give a measured CG / inertia (`measured:`).

## Install

Python 3.12+. Tested on Linux (Ubuntu 24.04 / WSL2). All dependencies are ordinary Python
wheels that also exist for Windows and macOS, but those platforms are not tested yet.

```bash
pip install -e .
```

or with [uv](https://docs.astral.sh/uv/):

```bash
uv sync
```

## Use it in the browser

```bash
robomodel serve
```

Open http://127.0.0.1:8000, upload the CAD, check the axes in the 3D view (red = forward,
blue = down), enter the mass, click thrusters and fins onto the hull, press **Build model** and
download a zip with everything below.

## Use it from the command line

Describe the vehicle in a short YAML file (all positions in metres, body frame: x forward,
y right, z down):

```yaml
name: my_auv
category: underwater
mass: 18.0                     # kg, weighed
water: fresh                   # fresh (1000), sea (1025) or a density in kg/m^3
geometry:
  file: hull.stl               # STL or STEP
  unit: mm
  axes: [x, y, z]              # body axis <- file axis, e.g. [z, -x, -y]
  # assembly: true             # multi-part / open-frame CAD (ROV)
parts:                         # heavy internal parts: they set CG and inertia
  - {name: battery, mass: 3.0, position: [0.1, 0, 0.05], shape: box, size: [0.2, 0.08, 0.06]}
thrusters:
  - {model: T200, position: [-0.5, 0.1, 0], direction: [1, 0, 0]}
  - {model: T200, position: [-0.5, -0.1, 0], direction: [1, 0, 0], dead_band: 0.05}
fins:                          # optional; control: true = rudder / stern plane
  - {name: rudder, position: [-0.45, 0, -0.1], normal: [0, 1, 0], area: 0.006, span: 0.08,
     control: true, max_angle: 20}
measured:                      # optional, overrides CAD estimates
  volume: 0.0181               # m^3, weighing in water
  r_g: [0.0, 0.0, 0.01]        # m
current: [0.2, 0, 0]           # optional water current, NED, m/s
```

```bash
robomodel build my_auv.yaml          # report + automatic checks
robomodel matlab my_auv.yaml -o out  # MATLAB: low / nominal / high + Simulink builder
robomodel gazebo my_auv.yaml -o gz   # Gazebo (gz-sim) model and water world
robomodel wizard my_auv.yaml         # which measurement would improve the model most
robomodel testplan my_auv.yaml -o plan    # pool tests (CSV templates)
robomodel calibrate my_auv.yaml run1.csv run2.csv --validate run3.csv
```

## Surface vessels (boats, USVs)

Set `category: surface`. robomodel floats the CAD hull: it finds draft, trim and heel exactly
from the closed mesh, the restoring matrix G (waterplane area, GM_T, GM_L) and the GZ curve up
to ±90° heel. Added mass and damping come from the wet hull plus its mirror image in the water
surface, halved. The model is `M ν̇ + C(ν)ν + D(ν)ν + G(η − η_rest) = τ`, with the roll
restoring taken from the GZ curve.

```yaml
name: my_usv
category: surface
mass: 182.2
water: sea
geometry: {file: hull.stl, unit: mm}   # closed hull including the deck
measured: {r_g: [0, 0, -0.05]}         # CG height decides the stability: measure it
thrusters:
  - {model: T200, position: [-1.9, 0.08, 0.15], direction: [1, 0, 0]}
  - {model: T200, position: [-1.9, -0.08, 0.15], direction: [1, 0, 0]}
```

Validated against closed forms (box barge, Wigley hull: hydrostatics within 0.01 %, GZ exact)
and the Clarke et al. (1983) regression (sway added mass −10 %, sway force −0 %, yaw damping
−45 %): [report](docs/dogrulama_su_ustu.md). Calm-water manoeuvring only: no waves, no
wave-making resistance above Froude 0.3. Gazebo export is not available for surface vessels.

## Multirotor drones

Set `category: aerial` and list the rotors. robomodel builds the rigid body from the frame CAD
and the parts (give `motor_mass` per rotor and list the battery), the rotor allocation matrix
with each propeller's reaction torque, frame drag in air and rotor drag, and checks that the
drone can hover level, has thrust to spare and controls thrust, roll, pitch and yaw.

```yaml
name: quad450
category: aerial
mass: 1.5
air: 1.225                      # kg/m^3 (default)
geometry: {file: frame.stl, unit: mm}
parts:
  - {name: battery, mass: 0.35, position: [0, 0, 0.02]}
rotors:                         # thrust up (-z) by default; spin seen from above
  - {position: [0.159, 0.159, -0.04], spin: ccw, max_thrust: 8.0, diameter: 0.254, motor_mass: 0.06}
  - {position: [-0.159, -0.159, -0.04], spin: ccw, max_thrust: 8.0, diameter: 0.254, motor_mass: 0.06}
  - {position: [0.159, -0.159, -0.04], spin: cw, max_thrust: 8.0, diameter: 0.254, motor_mass: 0.06}
  - {position: [-0.159, 0.159, -0.04], spin: cw, max_thrust: 8.0, diameter: 0.254, motor_mass: 0.06}
wind: [3, 0, 0]                 # optional, NED m/s
```

The MATLAB example starts at the hover commands. Torque/thrust ratio (from the diameter unless
`torque_ratio` is given) and drag are estimates; thrust falling with airspeed, motor lag and
gyroscopic effects are not modelled; no Gazebo export yet. [Report](docs/dogrulama_drone.md).

## Ground robots (wheeled, skid-steer, tracked)

Set `category: ground` and list the wheels (click their hubs on the web page). robomodel gives a
planar model `[x, y, ψ, u, v, r]` with wheel-ground contact: DC-motor drive lines, rolling
resistance, side friction with slip, the friction circle; static wheel loads, tip-over angles
from the CG height, top speed, traction-limited acceleration, steepest slope and turning.

```yaml
name: rover4
category: ground
mass: 8.0
surface: concrete               # indoor, asphalt, gravel, grass, sand, or {mu: .., rolling: ..}
geometry: {file: chassis.stl, unit: mm}
parts:
  - {name: battery, mass: 1.5, position: [0, 0, 0.02]}
wheels:                         # position = wheel centre; motor data at the wheel
  - {name: fl, position: [0.18, -0.2, 0.06], radius: 0.1, mass: 0.6, driven: true, stall_torque: 2.5, no_load_rpm: 150}
  - {name: fr, position: [0.18, 0.2, 0.06], radius: 0.1, mass: 0.6, driven: true, stall_torque: 2.5, no_load_rpm: 150}
  - {name: c, position: [-0.2, 0, 0.11], radius: 0.05, caster: true}
  # steered: true (max_steer: 35) for car-like steering; track_length: 0.4 for a track
```

Validated against hand calculations (loads, tip angles, top speed, μg start, differential and
bicycle kinematics); MATLAB ode15s matches the Python model to 10⁻⁷. Grip and rolling
resistance are surface estimates with a range; load transfer, suspension and slopes are not
modelled. [Report](docs/dogrulama_kara.md).

## Robots with joints (arms, legs, wheeled bases)

A multibody robot is a tree of links connected by joints. Each link's mass properties come
from its CAD (uniform density, your weighed mass); the equations of motion
`M(q) q̈ + h(q, q̇) = τ` are computed with the standard recursive algorithms (RNEA, CRBA).

```yaml
name: arm2
category: multibody
root: base
floating: false                # true for a free-moving base (rover, drone, AUV)
links:
  - {name: base, mass: 3.0, geometry: {file: base.stl, unit: mm}}
  - {name: upper, mass: 2.0, geometry: {file: upper.stl, unit: mm, pose: {xyz: [0, 0, 0]}}}
joints:
  - {name: shoulder, type: revolute, parent: base, child: upper, xyz: [0, 0, 0.1],
     axis: [0, 1, 0], lower: -3.14, upper: 3.14, effort: 50}
```

```bash
robomodel robot arm2.yaml -o out    # report, checks, URDF + meshes (ROS, Gazebo, MATLAB)
```

**On the web page** (type *Robot arm*): choose one CAD file per link, each exported from the
assembly at its zero pose in the same coordinates (that is the default when you export parts
of an assembly). Enter each link's mass, click the joints onto the model (axis, parent, child,
limits, motor limit) and build. You get the URDF with meshes, the report, the checks and the
**torque each motor needs to hold the arm against gravity**, at the drawn pose and at the worst
pose inside the joint limits, compared with the motor limits.

MATLAB without any toolbox: `robomodel_robot_mass.m` (M(q), composite rigid body algorithm),
`robomodel_robot_bias.m` (Coriolis + gravity + friction, recursive Newton-Euler) and
`robomodel_robot_dynamics.m` (ẋ = f(x, τ) for ode45 / Simulink); verified against Python: M(q)
4·10⁻¹⁵, g(q) 3·10⁻¹⁴, a 2 s free swing 9·10⁻⁹ rad. With the Robotics System Toolbox,
`<name>_check_toolbox.m` loads the URDF with `importrobot` and compares the same numbers.

Validated: pendulum period (0.1 %), analytic 2-link mass matrix (exact), energy of a chaotic
double pendulum (10⁻⁶), free fall of a floating base, and a damped two-link swing in Gazebo
from the exported URDF (elbow within 0.3–7 mm over 5 s). Joint friction, motors, wheel and
foot contacts are not in the CAD: they need data, like water damping does.

## Outputs

| Output | Contents |
|---|---|
| `<name>_params.m`, `_low.m`, `_high.m` | All matrices; nominal and the two range bounds |
| `robomodel_dynamics.m`, `robomodel_thrust.m` | ẋ = f(x, τ), thruster commands → τ (current, fin stall, dead band included) |
| `<name>_example.m` | ode45 run of all three models, compared with the Python simulator |
| `<name>_linear*.mat` | A, B at the cruise speed, for `lqr` / `pidtune` |
| `<name>_build_simulink.m` | Builds `<name>_sim.slx` (τ in, state out); first build and compile take ~3 min |
| `gazebo/<name>/model.sdf`, `world.sdf` | Hydrodynamics + Thruster plugins, buoyancy volume, test world |
| report, `model.yaml` | Every value with uncertainty and source, checks, provenance (version, input hashes) |

Every model passes automatic checks before you use it: mass matrices symmetric and positive
definite, Coriolis does no work, damping only removes energy (also in a free-motion
simulation), self-righting, neutral buoyancy, stability at rest and at cruise speed.

## Verified

- 166 automatic tests against analytic results (sphere, ellipsoids, Lamb coefficients, energy
  conservation, terminal speed, roll period, current drift) and of every output.
- MATLAB R2025b: `ode45` against the Python simulator, largest state difference ≤ 10⁻⁸;
  the generated Simulink model ≤ 10⁻⁴ over 30 s (final speeds equal to 5 digits).
- Gazebo against the Python model: steady speed within 0.04 %.
- Odd CAD: open shells, duplicated or inverted faces, solids inside solids, tori, thin plates,
  catamarans, scans, 1 cm–20 m, tilted drawings, far-away origins are repaired or flagged
  (`tests/test_robustness.py`); the model does not depend on where the CAD origin is.
- REMUS 100 against Prestero (2001) and sea trials; BlueROV2 Heavy against von Benzon et al.
  (2022) pool identification.

## Limits

- Damping cannot come from CAD: use the low/high range or calibrate with one pool test.
- CAD volume of an open frame is ±30 %: weigh the vehicle in water (`measured.volume`).
- Large STEP assemblies are slow to load (BlueROV2: 2.5 min, 6 GB RAM); export STL if possible.
- Gazebo: control-fin deflections and the water current are not exported yet.
- Surface vessels: yaw-rate derivatives are weak (turns too fast), no waves, no Gazebo export.
- Drones: drag and torque ratio are estimates; no thrust loss with airspeed; no Gazebo export.
- Ground robots: planar, flat ground, static wheel loads; grip is an estimate.
- Jointed robots: inertia assumes uniform density per link; joint friction, motors and foot
  contacts are not modelled; legged robots and floating bases only on the command line.

## Tests

```bash
uv run pytest            # about 190 tests, ~3 min, ~1.2 GB RAM
uv run pytest -m heavy   # also the BlueROV2 test (needs its 309 MB CAD, ~4 GB RAM)
```

## Project layout

| Path | What |
|---|---|
| `src/robomodel/vehicle.py` | Vehicle file schema and validation |
| `src/robomodel/geometry.py`, `voxel.py`, `sections.py` | CAD loading, voxel hulls, cross-sections |
| `src/robomodel/rigid_body.py`, `hydrostatics.py` | M_RB, CG, inertia, buoyancy, g(η) |
| `src/robomodel/added_mass.py`, `damping.py`, `frame.py`, `fins.py`, `body_lift.py` | Hydrodynamics |
| `src/robomodel/thrusters.py`, `dynamics.py` | Thrusters, equations of motion, simulator, linearisation |
| `src/robomodel/checks.py`, `model_range.py`, `provenance.py` | Checks, low/high bounds, version stamps |
| `src/robomodel/matlab_export.py`, `gazebo_export.py` | Outputs |
| `src/robomodel/calibration.py`, `testplan.py`, `wizard.py` | Pool tests and calibration |
| `src/robomodel/web.py`, `static/index.html` | Web app |
| `src/robomodel/bem.py` | Boundary element added mass (optional `robomodel[bem]`) |
| `src/robomodel/surface.py` | Floating position, GM, GZ curve, restoring matrix, mirror-image hull |
| `src/robomodel/aerial.py` | Drone hover allocation, thrust-to-weight, rotor drag |
| `src/robomodel/ground.py`, `ground_matlab.py` | Ground robots: wheel contact, loads, tip-over, MATLAB |
| `src/robomodel/multihull.py` | Separate hulls computed one by one and added up |
| `src/robomodel/multibody/` | Links, joints, RNEA/CRBA dynamics, robot files, URDF |
| `validation/` | REMUS 100, BlueROV2 and calibration studies (write the reports in `docs/`) |
| `docs/eksenler.md`, `docs/plan.md` | Axis conventions, project plan |

## License

Apache License 2.0, see [LICENSE](LICENSE). Copyright 2026 Arda Onur Kirik.
