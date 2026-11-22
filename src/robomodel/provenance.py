"""Provenance: which robomodel version made a model, when, and from which inputs.

Every output carries this so that a model can be reproduced or traced back later.
"""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from pathlib import Path

from robomodel import __version__
from robomodel.vehicle import Vehicle


def file_sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def provenance(vehicle: Vehicle) -> dict:
    out = {
        "robomodel_version": __version__,
        "created_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "vehicle": vehicle.name,
    }
    if vehicle.source is not None and Path(vehicle.source).exists():
        out["vehicle_file"] = Path(vehicle.source).name
        out["vehicle_file_sha256"] = file_sha256(vehicle.source)
    if vehicle.geometry.file is not None and Path(vehicle.geometry.file).exists():
        out["geometry_file"] = Path(vehicle.geometry.file).name
        out["geometry_file_sha256"] = file_sha256(vehicle.geometry.file)
    return out
