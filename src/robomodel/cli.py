"""Command line interface: robomodel check | build | matlab."""

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
    args = parser.parse_args(argv)

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
    return 0


if __name__ == "__main__":
    sys.exit(main())
