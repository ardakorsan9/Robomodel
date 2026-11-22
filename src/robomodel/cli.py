"""Command line interface: robomodel check | build | matlab | gazebo | testplan | calibrate |
wizard."""

from __future__ import annotations

import argparse
import sys

import yaml

from robomodel.geometry import GeometryError
from robomodel.model import build_model, report, to_dict
from robomodel.vehicle import VehicleError, load_vehicle


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="robomodel", description="Build a mathematical model of a robot from its description."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    check = commands.add_parser("check", help="validate a vehicle YAML file and print a summary")
    check.add_argument("vehicle", help="path to the vehicle YAML file")
    build = commands.add_parser("build", help="compute the model and print a report")
    build.add_argument("vehicle", help="path to the vehicle YAML file")
    build.add_argument("-o", "--output", help="also save all parameters to this YAML file")
    matlab = commands.add_parser("matlab", help="write MATLAB files for the model")
    matlab.add_argument("vehicle", help="path to the vehicle YAML file")
    matlab.add_argument("-o", "--output", required=True, help="output folder")
    matlab.add_argument(
        "--speed",
        type=float,
        default=0.0,
        help="surge speed (m/s) of the linear model's operating point",
    )
    testplan = commands.add_parser("testplan", help="write a pool test plan (CSV templates)")
    testplan.add_argument("vehicle", help="path to the vehicle YAML file")
    testplan.add_argument("-o", "--output", required=True, help="output folder")
    cal = commands.add_parser("calibrate", help="fit uncertain terms to logged test runs")
    cal.add_argument("vehicle", help="path to the vehicle YAML file")
    cal.add_argument("logs", nargs="+", help="CSV logs used for fitting")
    cal.add_argument("--validate", nargs="*", default=[], help="CSV logs only for checking")
    cal.add_argument(
        "--groups",
        default="damping,fin_lift,body_lift",
        help="comma list: damping, added_mass, fin_lift, body_lift, thrust",
    )
    cal.add_argument(
        "-o", "--output", help="calibrated vehicle YAML (default: <vehicle>_calibrated.yaml)"
    )
    gazebo = commands.add_parser("gazebo", help="write a Gazebo (gz-sim) model and test world")
    gazebo.add_argument("vehicle", help="path to the vehicle YAML file")
    gazebo.add_argument("-o", "--output", required=True, help="output folder")
    gazebo.add_argument(
        "--disable-coriolis",
        action="store_true",
        help="turn off added-mass Coriolis in Gazebo (if it goes unstable)",
    )
    serve = commands.add_parser("serve", help="start the web app (upload CAD, get the model)")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    wizard = commands.add_parser("wizard", help="which measurement improves the model most")
    wizard.add_argument("vehicle", help="path to the vehicle YAML file")
    args = parser.parse_args(argv)

    if args.command == "serve":
        from robomodel.web import serve as run_server

        print(f"robomodel web app: http://{args.host}:{args.port}")
        run_server(args.host, args.port)
        return 0
    if args.command == "gazebo":
        from robomodel.gazebo_export import write_gazebo

        model = _load_model(args.vehicle)
        if model is None:
            return 1
        for f in write_gazebo(model, args.output, args.disable_coriolis):
            print(f"wrote {f}")
        return 0
    if args.command == "wizard":
        from robomodel.wizard import report as wizard_report

        model = _load_model(args.vehicle)
        if model is None:
            return 1
        print(wizard_report(model))
        return 0
    if args.command == "testplan":
        return _testplan(args.vehicle, args.output)
    if args.command == "calibrate":
        return _calibrate(args)
    if args.command == "check":
        return _check(args.vehicle)
    if args.command == "build":
        return _build(args.vehicle, args.output)
    if args.command == "matlab":
        return _matlab(args.vehicle, args.output, args.speed)
    return 1  # unreachable: argparse rejects unknown commands


def _build(path: str, output: str | None) -> int:
    try:
        model = build_model(load_vehicle(path))
    except FileNotFoundError:
        print(f"error: file not found: {path}", file=sys.stderr)
        return 1
    except (VehicleError, GeometryError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    print(report(model))
    if output:
        with open(output, "w", encoding="utf-8") as f:
            yaml.safe_dump(to_dict(model), f, sort_keys=False, allow_unicode=True)
        print(f"\nsaved: {output}")
    return 0


def _matlab(path: str, output: str, speed: float) -> int:
    from robomodel.matlab_export import write_matlab

    try:
        model = build_model(load_vehicle(path))
    except FileNotFoundError:
        print(f"error: file not found: {path}", file=sys.stderr)
        return 1
    except (VehicleError, GeometryError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    for f in write_matlab(model, output, speed):
        print(f"wrote {f}")
    return 0


def _load_model(path: str):
    try:
        return build_model(load_vehicle(path))
    except FileNotFoundError:
        print(f"error: file not found: {path}", file=sys.stderr)
    except (VehicleError, GeometryError) as e:
        print(f"error: {e}", file=sys.stderr)
    return None


def _testplan(path: str, output: str) -> int:
    from robomodel.testplan import write_plan

    model = _load_model(path)
    if model is None:
        return 1
    for f in write_plan(model, output):
        print(f"wrote {f}")
    return 0


def _calibrate(args) -> int:
    from pathlib import Path

    from robomodel.calibration import calibrate, calibration_to_yaml, load_log, summary

    model = _load_model(args.vehicle)
    if model is None:
        return 1
    try:
        logs = [load_log(p) for p in args.logs]
        checks = [load_log(p) for p in args.validate]
        groups = tuple(g.strip() for g in args.groups.split(",") if g.strip())

        def progress(n, cost):
            if n % 10 == 0:
                print(f"  ... {n} simulations, cost {cost:.4g}", file=sys.stderr, flush=True)

        print(f"calibrating on {len(logs)} log(s); this can take a few minutes", file=sys.stderr)
        result = calibrate(model, logs, groups=groups, validation=checks, progress=progress)
    except (OSError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    print(summary(result))
    source = Path(args.vehicle)
    out = Path(args.output) if args.output else source.with_name(source.stem + "_calibrated.yaml")
    with source.open(encoding="utf-8") as f:
        data = yaml.safe_load(f)
    geometry = data.get("geometry", {})
    if "file" in geometry and not Path(geometry["file"]).is_absolute():
        # keep the mesh path valid if the output goes to another folder
        geometry["file"] = str((source.parent / geometry["file"]).resolve())
    data["calibration"] = calibration_to_yaml(result.calibration)
    with out.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)
    print(f"\nsaved: {out}  (use it with build / matlab like any vehicle file)")
    return 0


def _check(path: str) -> int:
    try:
        vehicle = load_vehicle(path)
    except FileNotFoundError:
        print(f"error: file not found: {path}", file=sys.stderr)
        return 1
    except VehicleError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    g = vehicle.geometry
    shape = f"{g.primitive} {g.dims}" if g.primitive else f"mesh {g.file} (x{g.unit_scale})"
    print(f"{vehicle.name}: OK")
    print(f"  category      {vehicle.category}")
    print(f"  mass          {vehicle.mass:g} kg")
    print(f"  water density {vehicle.water_density:g} kg/m^3")
    print(f"  geometry      {shape}")
    print(f"  parts         {len(vehicle.parts)}")
    print(f"  thrusters     {len(vehicle.thrusters)}")
    controls = sum(f.control for f in vehicle.fins)
    print(f"  fins          {len(vehicle.fins)} ({controls} control)")
    m = vehicle.measured
    measured = [n for n, x in (("r_g", m.r_g), ("inertia_g", m.inertia_g)) if x is not None]
    print(f"  measured      {', '.join(measured) or 'none'}")
    cal = vehicle.calibration
    print(f"  calibration   {'none' if cal.is_identity else cal.note or 'yes'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
