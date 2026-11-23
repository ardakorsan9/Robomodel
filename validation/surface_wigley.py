"""Surface vessels: robomodel against analytic hydrostatics and Clarke et al. (1983).

Run: uv run python validation/surface_wigley.py   (report: docs/dogrulama_su_ustu.md)

Hydrostatics: box barge and Wigley hull have closed-form values.
Manoeuvring: Clarke, Gedling & Hine (1983), "The application of manoeuvring criteria in hull
design using linear theory", Trans. RINA 125 - regressions of many ship models, as quoted in
Fossen, Handbook of Marine Craft Hydrodynamics and Motion Control (2nd ed., 2021). The
formulas below were typed from that source; check them there before relying on them.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from robomodel.dynamics import coriolis_matrix, damping_force
from robomodel.model import build_model
from robomodel.vehicle import load_vehicle

ROOT = Path(__file__).resolve().parents[1]


def clarke(L, B, T, cb):
    """-X' for each derivative (prime system: 1/2 rho L^k U^j)."""
    k = np.pi * (T / L) ** 2
    return {
        "Y_vdot": k * (1 + 0.16 * cb * B / T - 5.1 * (B / L) ** 2),
        "N_rdot": k * (1 / 12 + 0.017 * cb * B / T - 0.33 * B / L),
        "Y_v": k * (1 + 0.40 * cb * B / T),
        "Y_r": k * (-1 / 2 + 2.2 * B / L - 0.080 * B / T),
        "N_v": k * (1 / 2 + 2.4 * T / L),
        "N_r": k * (1 / 4 + 0.039 * B / T - 0.56 * B / L),
    }


def main() -> None:
    rho = 1025.0
    barge = build_model(load_vehicle(ROOT / "examples/barge.yaml")).floating
    print("Box barge 10 x 4 x 2 m, T = 1 m, KG = 1 m")
    for name, got, exp in (
        ("volume m^3", barge.volume, 40.0),
        ("waterplane m^2", barge.waterplane_area, 40.0),
        ("GM_T m", barge.gm_t, 0.5 + 16 / 12 - 1),
        ("GM_L m", barge.gm_l, 0.5 + 100 / 12 - 1),
    ):
        print(f"  {name:<16}{got:10.5g}  analytic {exp:10.5g}  {100 * (got / exp - 1):+.3f} %")
    phi = np.radians([10, 20, 25])
    wall = np.sin(phi) * (0.8333333 + 0.5 * 1.3333333 * np.tan(phi) ** 2)
    gz = np.interp(phi, barge.gz_angle, barge.gz)
    for a, g, w in zip((10, 20, 25), gz, wall, strict=True):
        print(f"  GZ {a} deg       {g:10.5g}  wall-sided {w:10.5g}  {100 * (g / w - 1):+.2f} %")

    model = build_model(load_vehicle(ROOT / "examples/wigley.yaml"))
    fl = model.floating
    L, B, T = 4.0, 0.4, 0.25
    keel = model.hull.bounds[1, 2]
    kg = keel - model.mass.r_g.value[2]
    print("\nWigley hull L = 4 m, B = 0.4 m, T = 0.25 m")
    for name, got, exp in (
        ("C_B", fl.volume / (L * B * T), 4 / 9),
        ("C_WP", fl.waterplane_area / (L * B), 2 / 3),
        ("KB m", keel - fl.r_b[2], 5 * T / 8),
        ("GM_T m", fl.gm_t, 5 * T / 8 + 3 * B**2 / (35 * T) - kg),
    ):
        print(f"  {name:<16}{got:10.5g}  analytic {exp:10.5g}  {100 * (got / exp - 1):+.3f} %")

    ma = np.asarray(model.m_a.value)
    dyn = model.dynamics()
    u = 1.0
    jac = np.zeros((6, 6))
    for j in range(6):
        d = np.zeros(6)
        d[j] = 1e-6
        base = np.array([u, 0, 0, 0, 0, 0.0])

        # all hydrodynamic forces: damping, lift and the added-mass Coriolis (Munk) terms,
        # as in Clarke's derivatives (which include the added-mass part)
        def hydro(nu):
            return damping_force(dyn, nu) + coriolis_matrix(dyn.m_a, nu) @ nu

        jac[:, j] = (hydro(base + d) - hydro(base - d)) / 2e-6
    q = 0.5 * rho
    ours = {
        "Y_vdot": ma[1, 1] / (q * L**3),
        "N_rdot": ma[5, 5] / (q * L**5),
        "Y_v": jac[1, 1] / (q * L**2 * u),
        "Y_r": jac[1, 5] / (q * L**3 * u),
        "N_v": jac[5, 1] / (q * L**3 * u),
        "N_r": jac[5, 5] / (q * L**4 * u),
    }
    # every entry is -X' (Clarke's form); ours: jac is d(hydrodynamic terms)/d nu = -dX/d nu
    print("\n  non-dimensional -Y_vdot', -N_rdot', -Y_v', -Y_r', -N_v', -N_r' at u = 1 m/s")
    for key, exp in clarke(L, B, T, 4 / 9).items():
        got = ours[key]
        print(f"  {key:<8}{got:10.4g}  Clarke {exp:10.4g}  {100 * (got / exp - 1):+.0f} %")


if __name__ == "__main__":
    main()
