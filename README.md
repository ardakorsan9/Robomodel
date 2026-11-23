# robomodel

**CAD file in, ready-to-use mathematical model out.** robomodel turns a vehicle's CAD (STL or
STEP) and a few numbers (mass, thrusters, fins) into a 6-degree-of-freedom Fossen model:

```
M ν̇ + C(ν) ν + D(ν) ν + g(η) = τ,   M = M_RB + M_A
```

with MATLAB, Simulink and Gazebo files, automatic sanity checks, and an honest statement of how
far each number can be trusted. First target: underwater vehicles (torpedo AUVs and open-frame
ROVs). Other robot types (rovers, drones, arms) share the same core and are planned.

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
- Only underwater vehicles so far.

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
| `src/robomodel/multihull.py` | Separate hulls computed one by one and added up |
| `src/robomodel/multibody/` | Links, joints, RNEA/CRBA dynamics, robot files, URDF |
| `validation/` | REMUS 100, BlueROV2 and calibration studies (write the reports in `docs/`) |
| `docs/eksenler.md`, `docs/plan.md` | Axis conventions, project plan |
