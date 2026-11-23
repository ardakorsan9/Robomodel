"""Web app: upload a CAD file, describe the vehicle in the browser, download the model.

    uv run robomodel serve            # then open http://127.0.0.1:8000

The browser shows the CAD in 3D (three.js) in the body frame, so the user can check which
way is forward and up and click thrusters and fins onto the hull. The server builds the
model with the same code as the command line and returns a zip with the report, the vehicle
file, MATLAB (low/nominal/high + Simulink builder) and Gazebo files.

Limits for a public deployment: MAX_UPLOAD_MB, allowed file types, MAX_JOBS builds at once,
each in its own process with BUILD_TIMEOUT_S. Uploads and results are deleted after
RETENTION_HOURS (and on request), and the whole work folder at exit.
"""

from __future__ import annotations

import atexit
import json
import multiprocessing
import shutil
import tempfile
import threading
import time
import uuid
import zipfile
from pathlib import Path

import numpy as np
import trimesh
import yaml
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, Response

from robomodel import __version__
from robomodel.checks import run_checks
from robomodel.geometry import GeometryError
from robomodel.model import build_model, report, to_dict
from robomodel.vehicle import VehicleError, parse_vehicle

MAX_UPLOAD_MB = 400
MAX_JOBS = 2  # models built at the same time
BUILD_TIMEOUT_S = 600
RETENTION_HOURS = 2  # uploads and results are deleted after this
ALLOWED = {".stl", ".step", ".stp", ".obj", ".ply"}
PREVIEW_FACES = 300_000  # larger meshes are decimated by sampling faces for the browser
STATIC = Path(__file__).parent / "static"

WORK = Path(tempfile.mkdtemp(prefix="robomodel_web_"))
atexit.register(shutil.rmtree, WORK, ignore_errors=True)

app = FastAPI(title="robomodel", version=__version__)
JOBS = threading.BoundedSemaphore(MAX_JOBS)


@app.middleware("http")
async def security_headers(request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "no-referrer"
    return response


def _upload_dir(upload_id: str) -> Path:
    if not upload_id or not all(c in "0123456789abcdef" for c in upload_id):
        raise HTTPException(400, "bad upload id")
    path = WORK / upload_id
    if not path.is_dir():
        raise HTTPException(404, "upload not found (the server may have restarted)")
    return path


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return (STATIC / "index.html").read_text(encoding="utf-8")


@app.get("/api/version")
def version() -> dict:
    return {"robomodel": __version__}


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)) -> dict:
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED:
        raise HTTPException(400, f"file type {suffix or '?'} not supported; use STL or STEP")
    _cleanup_old()
    upload_id = uuid.uuid4().hex
    folder = WORK / upload_id
    folder.mkdir()
    target = folder / f"cad{suffix}"
    size = 0
    with target.open("wb") as out:
        while chunk := await file.read(1 << 20):
            size += len(chunk)
            if size > MAX_UPLOAD_MB << 20:
                shutil.rmtree(folder, ignore_errors=True)
                raise HTTPException(413, f"file larger than {MAX_UPLOAD_MB} MB")
            out.write(chunk)
    (folder / "name.txt").write_text(Path(file.filename).name, encoding="utf-8")
    return {"id": upload_id, "filename": Path(file.filename).name, "size_mb": size / 2**20}


@app.get("/api/preview/{upload_id}")
def preview(upload_id: str) -> Response:
    """The CAD as binary STL in file units (STEP is tessellated by the server)."""
    folder = _upload_dir(upload_id)
    cache = folder / "preview.stl"
    if not cache.exists():
        source = next(p for p in folder.iterdir() if p.stem == "cad")
        try:
            mesh = trimesh.load(source, force="mesh")
        except Exception as e:  # trimesh raises many types
            raise HTTPException(400, f"could not read the CAD file: {e}") from e
        if len(mesh.faces) > PREVIEW_FACES:
            keep = np.random.default_rng(0).choice(len(mesh.faces), PREVIEW_FACES, replace=False)
            mesh = trimesh.Trimesh(mesh.vertices, mesh.faces[np.sort(keep)], process=False)
        info = {
            "faces": int(len(mesh.faces)),
            "watertight": bool(mesh.is_watertight),
            "bounds": np.asarray(mesh.bounds).tolist(),
            "extents": np.asarray(mesh.extents).tolist(),  # in file units
        }
        mesh.export(cache)
        (folder / "preview.json").write_text(json.dumps(info), encoding="utf-8")
    return FileResponse(cache, media_type="model/stl")


@app.get("/api/preview/{upload_id}/info")
def preview_info(upload_id: str) -> dict:
    folder = _upload_dir(upload_id)
    if not (folder / "preview.json").exists():
        preview(upload_id)
    return json.loads((folder / "preview.json").read_text(encoding="utf-8"))


def _run_build(folder: str, spec: dict, result_id: str) -> None:
    """The heavy work, in its own process: model, report, MATLAB, Gazebo, zip.

    Writes out/result.json, or out/error.json with a message for the user.
    """
    folder = Path(folder)
    out = folder / result_id
    out.mkdir(parents=True, exist_ok=True)
    try:
        source = next(p for p in folder.iterdir() if p.stem == "cad")
        geometry = dict(spec.get("geometry") or {})
        geometry.pop("primitive", None)
        geometry["file"] = str(source)
        spec = dict(spec, geometry=geometry)
        try:
            model = build_model(parse_vehicle(spec, base_dir=folder))
        except (VehicleError, GeometryError) as e:
            (out / "error.json").write_text(json.dumps({"status": 422, "detail": str(e)}))
            return

        from robomodel.gazebo_export import write_gazebo
        from robomodel.matlab_export import write_matlab

        name = (folder / "name.txt").read_text(encoding="utf-8")
        shown = dict(spec, geometry=dict(geometry, file=name))
        (out / "vehicle.yaml").write_text(yaml.safe_dump(shown, sort_keys=False), encoding="utf-8")
        text = report(model)
        (out / "report.txt").write_text(text + "\n", encoding="utf-8")
        data = to_dict(model)
        (out / "model.yaml").write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
        write_matlab(model, out / "matlab", speed=float(spec.get("speed", 1.0)))
        if model.floating is None:
            write_gazebo(model, out / "gazebo")
        else:  # the gz-sim Hydrodynamics plugin assumes a fully submerged body
            (out / "gazebo").mkdir(exist_ok=True)
            (out / "gazebo" / "README.txt").write_text(
                "Gazebo export is not available for surface vessels yet: gz-sim has no "
                "plugin for a floating hull's waterline forces. Use the MATLAB files.\n",
                encoding="utf-8",
            )
        with zipfile.ZipFile(folder / f"{result_id}.zip", "w", zipfile.ZIP_DEFLATED) as z:
            for p in sorted(out.rglob("*")):
                if p.is_file() and p.name != "result.json":
                    z.write(p, Path(model.vehicle.name) / p.relative_to(out))
        checks = [
            {"name": c.name, "status": c.status, "detail": c.detail}
            for c in run_checks(model.dynamics())
        ]
        result = {
            "result": result_id,
            "report": text,
            "checks": checks,
            "warnings": model.warnings,
            "summary": {
                "mass": data["mass"]["value"],
                "volume": data["volume"]["value"],
                "buoyancy": data["buoyancy"]["value"],
                "weight": data["weight"]["value"],
                "r_g": data["r_g"]["value"],
                "r_b": data["r_b"]["value"],
                "added_mass_diag": np.diag(np.asarray(data["M_A"]["value"])).tolist(),
                "damping_quadratic": data["D_quadratic"]["value"],
                "bounds": np.asarray(model.hull.bounds).tolist(),
                "category": model.vehicle.category,
                "floating": data["floating"]["summary"] if data["floating"] else None,
            },
            "model": data,  # every value with sigma, source and unit
            "behaviour": _behaviour(model),
        }
        (out / "result.json").write_text(json.dumps(result, default=float), encoding="utf-8")
    except Exception as e:  # anything unexpected: tell the user, keep the server alive
        (out / "error.json").write_text(
            json.dumps({"status": 500, "detail": f"model building failed: {type(e).__name__}: {e}"})
        )


def _behaviour(model) -> dict:
    """Plain numbers a user cares about, for the nominal model and the low/high bounds:
    top speed, turn rate, roll and pitch periods, net buoyancy. Keys are stable names."""
    from robomodel.model_range import model_range
    from robomodel.wizard import _metrics

    keys = {
        "son hız": "top_speed",
        "dönüş hızı (5° dümen)": "turn_rate_fins",
        "dönüş hızı (%50 itici)": "turn_rate_thrusters",
        "dikey hız (%50 itici)": "vertical_speed",
        "yuvarlanma periyodu": "roll_period",
        "yunuslama periyodu": "pitch_period",
        "net kaldırma": "net_buoyancy",
        "dalga periyodu (dikey)": "heave_period",
    }
    out: dict = {}
    try:
        for label, variant in model_range(model).items():
            for name, value in _metrics(variant, variant.dynamics()).items():
                out.setdefault(keys.get(name, name), {})[label] = float(value)
    except Exception as e:  # a summary must never break the build
        out["error"] = str(e)
    return out


@app.post("/api/build/{upload_id}")
def build(upload_id: str, spec: dict) -> dict:
    """spec: a vehicle description like the YAML file, without geometry.file.

    Runs in a separate process with a time limit, at most MAX_JOBS at once, so one huge or
    broken CAD cannot freeze the site for everyone.
    """
    folder = _upload_dir(upload_id)
    if not JOBS.acquire(blocking=False):
        raise HTTPException(503, "the server is busy building other models; try again shortly")
    try:
        result_id = uuid.uuid4().hex
        job = multiprocessing.get_context("spawn").Process(
            target=_run_build, args=(str(folder), dict(spec), result_id), daemon=True
        )
        job.start()
        job.join(BUILD_TIMEOUT_S)
        if job.is_alive():
            job.terminate()
            job.join(5)
            raise HTTPException(
                504,
                f"building took longer than {BUILD_TIMEOUT_S} s; try a lighter CAD "
                "(fewer triangles) or a larger voxel size",
            )
    finally:
        JOBS.release()
    out = folder / result_id
    if (out / "error.json").exists():
        err = json.loads((out / "error.json").read_text())
        raise HTTPException(err["status"], err["detail"])
    if not (out / "result.json").exists():
        raise HTTPException(500, "model building stopped unexpectedly (out of memory?)")
    return json.loads((out / "result.json").read_text(encoding="utf-8"))


@app.delete("/api/upload/{upload_id}")
def delete_upload(upload_id: str) -> dict:
    """Delete the CAD and every result made from it, now."""
    shutil.rmtree(_upload_dir(upload_id), ignore_errors=True)
    return {"deleted": upload_id}


@app.get("/api/health")
def health() -> dict:
    return {"ok": True, "version": __version__, "uploads": sum(1 for _ in WORK.iterdir())}


@app.post("/api/example")
def example() -> dict:
    """A ready example: the REMUS 100 hull (Myring profile) as STL in mm, plus its data."""
    from robomodel.geometry import build_mesh
    from robomodel.vehicle import Geometry

    _cleanup_old()
    upload_id = uuid.uuid4().hex
    folder = WORK / upload_id
    folder.mkdir()
    dims = dict(
        a=0.191, b=0.654, c=0.541, d=0.191, n=2.0, theta=0.436, a_offset=0.0165, c_offset=0.0368
    )
    mesh = build_mesh(Geometry(primitive="myring", dims=dims))
    # Positions in the body frame of this file (origin at the hull's mid-length), metres.
    cb_x = float(mesh.center_mass[0])
    tail_x = float(mesh.bounds[0, 0])
    fin_x = round(cb_x - 0.638, 4)  # Prestero Table 2.2: fins 0.638 m behind the CB
    fin_r = 0.0904  # hull radius at the fin + half span
    fins = [
        {"name": "rudder_top", "position": [fin_x, 0, -fin_r], "normal": "+y"},
        {"name": "rudder_bottom", "position": [fin_x, 0, fin_r], "normal": "+y"},
        {"name": "plane_port", "position": [fin_x, -fin_r, 0], "normal": "+z"},
        {"name": "plane_stbd", "position": [fin_x, fin_r, 0], "normal": "+z"},
    ]
    mesh.apply_scale(1000.0)
    mesh.export(folder / "cad.stl")
    (folder / "name.txt").write_text("remus100_hull.stl", encoding="utf-8")
    spec = {
        "name": "remus100_example",
        "mass": 30.49,
        "water": "sea",
        "unit": "mm",
        "forward": "+x",
        "up": "-z",
        # Prestero 2001: CG 1.96 cm below the CB; four 66.5 cm^2 tail fins, span 8.57 cm.
        "cg": [round(cb_x, 4), 0.0, 0.0196],
        # Weighed in water: B = 306 N (Table 2.3); the CAD counts the flooded nose as solid.
        "volume": round(306 / (1025 * 9.80665), 5),
        "fins": [dict(f, area_cm2=66.5, span_cm=8.57, control=True) for f in fins],
        # The REMUS propeller, as a thruster: ~15 N gives its 2.5 m/s top speed (estimate).
        "thrusters": [
            {
                "name": "propeller",
                "position": [round(tail_x + 0.01, 4), 0, 0],
                "direction": "+x",
                "model": "REMUS prop",
                "max_forward": 15,
                "max_reverse": 8,
            }
        ],
    }
    return {"id": upload_id, "filename": "remus100_hull.stl", "spec": spec}


def _cleanup_old() -> None:
    """Privacy: forget uploads and results older than RETENTION_HOURS."""
    limit = time.time() - RETENTION_HOURS * 3600
    for folder in WORK.iterdir():
        try:
            if folder.stat().st_mtime < limit:
                shutil.rmtree(folder, ignore_errors=True)
        except FileNotFoundError:
            pass


@app.get("/api/download/{upload_id}/{result_id}")
def download(upload_id: str, result_id: str) -> FileResponse:
    folder = _upload_dir(upload_id)
    if not all(c in "0123456789abcdef" for c in result_id):
        raise HTTPException(400, "bad result id")
    path = folder / f"{result_id}.zip"
    if not path.exists():
        raise HTTPException(404, "result not found")
    return FileResponse(path, media_type="application/zip", filename="robomodel_model.zip")


def serve(host: str = "127.0.0.1", port: int = 8000) -> None:
    import uvicorn

    uvicorn.run(app, host=host, port=port, log_level="info")
