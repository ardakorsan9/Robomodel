"""Low / nominal / high models for terms CAD cannot pin down.

Some terms cannot be computed exactly from geometry (above all damping). Instead of one
number we give three models that bracket the real vehicle, so a controller can be checked
in simulation without the vehicle: design on "nominal", then check it on "low" (least
damping and inertia, most thrust: lively) and "high" (most damping and inertia, least
thrust: sluggish).

The bounds come from the two vehicles we validated against measurements:
- REMUS 100 (Prestero 2001): crossflow damping from CAD was ~10x too low; surge drag within
  2 %; fin lift table vs formula differ 2.2x.
- BlueROV2 Heavy (von Benzon 2022): frame drag from CAD 1.6-4.6x too low.
So damping may be several times higher than the CAD value, rarely much lower.
"""

from __future__ import annotations

import numpy as np

from robomodel.model import VehicleModel, apply_calibration
from robomodel.parameter import Source
from robomodel.vehicle import Calibration

# Scale factors on the CAD values: (low, high)
DAMPING_SURGE = (0.7, 3.0)
DAMPING_OTHER = (0.7, 10.0)  # sway, heave, roll, pitch, yaw
ADDED_MASS = (0.6, 1.5)
FIN_LIFT = (0.45, 1.2)
BODY_LIFT = (0.5, 1.5)
THRUST = (0.85, 1.15)


def bounds_calibration(which: str) -> Calibration:
    i = {"low": 0, "high": 1}[which]
    damping = np.array([DAMPING_SURGE[i]] + [DAMPING_OTHER[i]] * 5)
    # "low" = least resistance: little damping and inertia, strong thrust (lively).
    thrust = THRUST[1 - i]
    return Calibration(
        damping=damping,
        added_mass=np.full(6, ADDED_MASS[i]),
        fin_lift=FIN_LIFT[i],
        body_lift=BODY_LIFT[i],
        thrust=thrust,
        note=f"model range bound '{which}' (validated spread of CAD estimates)",
    )


def model_range(model: VehicleModel) -> dict[str, VehicleModel]:
    """{"low": ..., "nominal": model, "high": ...}. A calibrated model has no range: test
    data already fixed those terms."""
    if not model.vehicle.calibration.is_identity:
        return {"nominal": model}
    return {
        "low": apply_calibration(model, bounds_calibration("low"), Source.ASSUMED),
        "nominal": model,
        "high": apply_calibration(model, bounds_calibration("high"), Source.ASSUMED),
    }
