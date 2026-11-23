"""Write a model as MATLAB files.

<name>_params.m       parameter struct (all matrices, Fossen frame)
robomodel_dynamics.m  xdot = f(t, x, tau, p): the same equations as dynamics.py
robomodel_thrust.m    thruster commands [-1, 1] -> generalised force tau
<name>_example.m      ode45 run, comparison with the Python reference, LQR-ready A, B
<name>_reference.csv  the same run from the Python simulator
<name>_linear.mat     A (12x12), B (12x6, input tau), Bu (12xn, input thrust in N)
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
from scipy.io import savemat

from robomodel.dynamics import linearize, simulate
from robomodel.model import VehicleModel
from robomodel.thrusters import thrust

EXAMPLE_TIME = 30.0  # s
# A hovering drone is unstable without a controller: round-off grows like exp(t / 1.3 s), so
# MATLAB and Python can only be compared over a short run.
AERIAL_EXAMPLE_TIME = 5.0  # s
EXAMPLE_DT = 0.05  # s
EXAMPLE_COMMAND = 0.5  # every thruster at half command
NO_THRUSTER_SURGE_FORCE = 10.0  # N, used when the vehicle has no thrusters


def matlab_name(name: str) -> str:
    """A valid MATLAB identifier from a vehicle name."""
    ident = re.sub(r"\W", "_", name)
    if not ident or not ident[0].isalpha():
        ident = "v_" + ident
    return ident[:50]


def _mat(a) -> str:
    """MATLAB literal with full double precision."""
    a = np.atleast_2d(np.asarray(a, dtype=float))
    rows = ["  ".join(f"{x:.17g}" for x in row) for row in a]
    return "[" + "; ...\n     ".join(rows) + "]"


def example_tau(model: VehicleModel) -> tuple[np.ndarray, np.ndarray | None]:
    thrusters = model.vehicle.thrusters
    if model.hover is not None and model.hover.possible:  # drone: hover in place
        commands = model.hover.commands(np.array([t.max_forward for t in thrusters]))
        return model.b_thr @ thrust(thrusters, commands), commands
    if thrusters:
        commands = np.full(len(thrusters), EXAMPLE_COMMAND)
        return model.b_thr @ thrust(thrusters, commands), commands
    return np.array([NO_THRUSTER_SURGE_FORCE, 0, 0, 0, 0, 0.0]), None


def write_matlab(
    model: VehicleModel, out_dir: str | Path, speed: float = 0.0, with_range: bool = True
) -> list[Path]:
    """Write all MATLAB files; return their paths.

    speed: surge speed (m/s) of the operating point used for the linear model.
    with_range: also write the low/high bound models (model_range.py), as
    <name>_params_low.m / _high.m and <name>_linear_low.mat / _high.mat.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    name = matlab_name(model.vehicle.name)
    dyn = model.dynamics()
    written = []

    def write(filename: str, text: str) -> None:
        path = out / filename
        path.write_text(text, encoding="utf-8")
        written.append(path)

    from robomodel.model_range import model_range
    from robomodel.provenance import provenance

    stamp = provenance(model.vehicle)
    stamp_text = ", ".join(f"{k}={v}" for k, v in stamp.items() if not k.endswith("sha256"))
    variants = model_range(model) if with_range else {"nominal": model}
    for label, variant in variants.items():
        suffix = "" if label == "nominal" else f"_{label}"
        write(
            f"{name}_params{suffix}.m",
            _params_text(variant, f"{name}_params{suffix}", label, stamp_text),
        )
        savemat_path = out / f"{name}_linear{suffix}.mat"
        _save_linear(variant, speed, savemat_path)
        written.append(savemat_path)
    write(f"{name}_build_simulink.m", _simulink_text(name))
    write(f"{name}_provenance.txt", "\n".join(f"{k}: {v}" for k, v in stamp.items()) + "\n")

    write(
        "robomodel_dynamics.m",
        """function xdot = robomodel_dynamics(~, x, tau, p)
%ROBOMODEL_DYNAMICS  Fossen 6-DOF equations of motion (same as robomodel dynamics.py).
%   x = [eta; nu], eta = [x y z phi theta psi]' (NED), nu = [u v w p q r]' (body).
eta = x(1:6);
nu = x(7:12);
phi = eta(4); theta = eta(5); psi = eta(6);

M = p.M_RB + p.M_A;
% Water current: hydrodynamic terms use the velocity relative to the water.
v_c = Rzyx(phi, theta, psi)' * p.current_ned(:);
nu_r = nu - [v_c; 0; 0; 0];
nu_c_dot = [-cross(nu(4:6), v_c); 0; 0; 0];
C = coriolis(p.M_RB, nu) * nu + coriolis(p.M_A, nu_r) * nu_r;
H = [eye(3), -S(p.r_hydro); zeros(3), eye(3)];   % velocity at the hull centre = H * nu
nu_h = H * nu_r;
D = H' * (p.D_lin .* nu_h + p.D_quad .* abs(nu_h) .* nu_h) + abs(nu_r(1)) * p.D_fin * nu_r;
delta = max(min(p.delta, p.fin_max), -p.fin_max);   % fin stall: hold at max angle
tau_fin = nu_r(1) * abs(nu_r(1)) * p.B_fin * delta;   % control fins
if p.surface
    d_eta = eta - p.eta_rest(:);
    g = p.G * d_eta;
    if ~isempty(p.roll_table)   % large heel: righting moment from the GZ curve
        g(4) = g(4) + interp1(p.roll_table(1, :), p.roll_table(2, :), ...
               min(max(phi, p.roll_table(1, 1)), p.roll_table(1, end))) ...
               - p.G(4, 4) * d_eta(4);
    end
else
    g = restoring(phi, theta, p);
end
nu_dot = M \\ (tau(:) + tau_fin - C - D - g + p.M_A * nu_c_dot);
eta_dot = [Rzyx(phi, theta, psi) * nu(1:3); Tzyx(phi, theta) * nu(4:6)];
xdot = [eta_dot; nu_dot];
end

function C = coriolis(M, nu)
a = M(1:3, 1:3) * nu(1:3) + M(1:3, 4:6) * nu(4:6);
b = M(4:6, 1:3) * nu(1:3) + M(4:6, 4:6) * nu(4:6);
C = [zeros(3), -S(a); -S(a), -S(b)];
end

function g = restoring(phi, theta, p)
W = p.W; B = p.B;
xg = p.r_g(1); yg = p.r_g(2); zg = p.r_g(3);
xb = p.r_b(1); yb = p.r_b(2); zb = p.r_b(3);
g = [ (W - B) * sin(theta);
     -(W - B) * cos(theta) * sin(phi);
     -(W - B) * cos(theta) * cos(phi);
     -(yg*W - yb*B) * cos(theta) * cos(phi) + (zg*W - zb*B) * cos(theta) * sin(phi);
      (zg*W - zb*B) * sin(theta) + (xg*W - xb*B) * cos(theta) * cos(phi);
     -(xg*W - xb*B) * cos(theta) * sin(phi) - (yg*W - yb*B) * sin(theta)];
end

function R = Rzyx(phi, theta, psi)
cph = cos(phi); sph = sin(phi); cth = cos(theta); sth = sin(theta);
cps = cos(psi); sps = sin(psi);
R = [cps*cth, -sps*cph + cps*sth*sph,  sps*sph + cps*cph*sth;
     sps*cth,  cps*cph + sph*sth*sps, -cps*sph + sth*sps*cph;
     -sth,     cth*sph,                cth*cph];
end

function T = Tzyx(phi, theta)
T = [1, sin(phi)*tan(theta), cos(phi)*tan(theta);
     0, cos(phi),            -sin(phi);
     0, sin(phi)/cos(theta), cos(phi)/cos(theta)];
end

function M = S(v)
M = [0, -v(3), v(2); v(3), 0, -v(1); -v(2), v(1), 0];
end
""",
    )

    write(
        "robomodel_thrust.m",
        """function tau = robomodel_thrust(commands, p)
%ROBOMODEL_THRUST  Thruster commands in [-1, 1] -> generalised force tau (6x1).
%   Thrust grows with the square of the command (same as robomodel thrusters.py).
c = max(min(commands(:)', 1), -1);
c = sign(c) .* max((abs(c) - p.dead_band) ./ (1 - p.dead_band), 0);   % dead band
f = zeros(size(c));
pos = c >= 0;
f(pos) = p.f_max_forward(pos) .* c(pos).^2;
f(~pos) = -p.f_max_reverse(~pos) .* c(~pos).^2;
tau = p.B_thr * f(:);
end
""",
    )

    tau, commands = example_tau(model)
    t_end = AERIAL_EXAMPLE_TIME if dyn.kind == "aerial" else EXAMPLE_TIME
    traj = simulate(dyn, t_end, tau=tau, dt_out=EXAMPLE_DT)
    csv = out / f"{name}_reference.csv"
    header = "t,x,y,z,phi,theta,psi,u,v,w,p,q,r"
    np.savetxt(
        csv,
        np.column_stack([traj.t, traj.eta, traj.nu]),
        delimiter=",",
        header=header,
        comments="",
        fmt="%.12g",
    )
    written.append(csv)

    input_line = (
        f"tau = robomodel_thrust({_mat([commands])}, p);   % "
        + ("hover" if model.hover is not None else f"every thruster at {EXAMPLE_COMMAND:g}")
        if commands is not None
        else f"tau = [{NO_THRUSTER_SURGE_FORCE:g}; 0; 0; 0; 0; 0];   % no thrusters: push forward"
    )
    write(
        f"{name}_example.m",
        f"""%{name.upper()}_EXAMPLE  Simulate '{model.vehicle.name}' and check it against robomodel.
p = {name}_params();
{input_line}
x0 = {_mat(dyn.rest_state.reshape(-1, 1))};   % at rest (floating position for a boat)
opts = odeset('RelTol', 1e-8, 'AbsTol', 1e-10);
times = 0:{EXAMPLE_DT:g}:{t_end:g};
[t, x] = ode45(@(t, x) robomodel_dynamics(t, x, tau, p), times, x0, opts);

% Same run from the Python simulator: the two must agree.
ref = readmatrix('{name}_reference.csv', 'NumHeaderLines', 1);
err = max(abs(x - ref(:, 2:13)), [], 'all');
fprintf('robomodel check: max difference MATLAB vs Python = %.3g\\n', err);
fprintf('final surge speed u = %.4f m/s, position x = %.3f m\\n', x(end, 7), x(end, 1));

figure('Name', p.name);
subplot(2, 1, 1); plot(t, x(:, 1:3)); grid on;
legend('x (north)', 'y (east)', 'z (down)'); ylabel('position [m]');
subplot(2, 1, 2); plot(t, x(:, 7:9)); grid on;
legend('u', 'v', 'w'); ylabel('velocity [m/s]'); xlabel('time [s]');

% Model range: the same run with the low and high bound models (if written).
for v = {{'low', 'high'}}
    f = ['{name}_params_' v{{1}}];
    if exist(f, 'file')
        pv = feval(f);
        [~, xv] = ode45(@(t, x) robomodel_dynamics(t, x, tau, pv), times, x0, opts);
        fprintf('%-7s model: final surge speed u = %.4f m/s\\n', v{{1}}, xv(end, 7));
        subplot(2, 1, 2); hold on; plot(t, xv(:, 7), '--');
    end
end

% Linear model for controller design (operating point: surge speed {speed:g} m/s).
lin = load('{name}_linear.mat');
fprintf('linear model: %d states, slowest pole %.3g 1/s\\n', size(lin.A, 1), ...
        max(real(eig(lin.A))));
% Example: sys = ss(lin.A, lin.B, eye(12), 0);  K = lqr(lin.A, lin.B, Q, R);
""",
    )
    return written


def _params_text(model: VehicleModel, fn: str, label: str, stamp: str) -> str:
    dyn = model.dynamics()
    thr = model.vehicle.thrusters
    fin_names = ", ".join(model.fins.control_names) or "none"
    note = {
        "nominal": "Best estimate from CAD.",
        "low": "Range bound LOW: least damping, added mass and lift; most thrust.",
        "high": "Range bound HIGH: most damping, added mass and lift; least thrust.",
    }[label]
    return f"""function p = {fn}()
%{fn.upper()}  Model of '{model.vehicle.name}' generated by robomodel.
%   {note}
%   Fossen body frame: x forward, y right, z down. Matrices about the body origin CO.
%   Equations: M*nu_dot + C(nu)*nu + D(nu)*nu + g(eta) = tau, M = M_RB + M_A.
%   Uncertainties and sources of every value: see the robomodel report / YAML output.
%   {stamp}

p.name = '{model.vehicle.name}';
p.variant = '{label}';
p.M_RB = {_mat(dyn.m_rb)};
p.M_A = {_mat(dyn.m_a)};
p.D_lin = {_mat(dyn.d_lin.reshape(-1, 1))};   % positive, X_u = -D_lin(1)
p.D_quad = {_mat(dyn.d_quad.reshape(-1, 1))};  % positive, X_u|u| = -D_quad(1)
p.W = {dyn.weight:.17g};   % weight, N
p.B = {dyn.buoyancy:.17g};   % buoyancy, N
p.r_g = {_mat(dyn.r_g.reshape(-1, 1))};   % centre of gravity, m
p.r_b = {_mat(dyn.r_b.reshape(-1, 1))};   % centre of buoyancy, m
p.B_thr = {_mat(dyn.b_thr) if thr else "zeros(6, 0)"};   % tau = B_thr * f, f in N
p.f_max_forward = {_mat([[t.max_forward for t in thr]]) if thr else "zeros(1, 0)"};
p.f_max_reverse = {_mat([[t.max_reverse for t in thr]]) if thr else "zeros(1, 0)"};
p.D_fin = {_mat(dyn.d_fin)};   % fin lift damping, multiplied by |u|
p.B_fin = {_mat(dyn.b_fin) if dyn.b_fin.shape[1] else "zeros(6, 0)"};   % x u|u| x delta
p.fin_max = {_mat(dyn.fin_max.reshape(-1, 1)) if dyn.fin_max.size else "zeros(0, 1)"};   % rad
p.delta = zeros({dyn.b_fin.shape[1]}, 1);   % control fin deflections (rad): {fin_names}
p.dead_band = {_mat([[t.dead_band for t in thr]]) if thr else "zeros(1, 0)"};   % command fraction
p.current_ned = {_mat(dyn.current_ned.reshape(-1, 1))};   % constant water current, m/s (NED)
p.r_hydro = {_mat(dyn.r_hydro.reshape(-1, 1))};   % hull centre: D_lin, D_quad act here, m
{_surface_params(dyn)}end
"""


def _surface_params(dyn) -> str:
    if dyn.g_matrix is None:
        # fields exist in both cases so Simulink code generation sees the same struct
        return (
            "p.surface = false;   % fully submerged: restoring from W, B, r_g, r_b\n"
            "p.eta_rest = zeros(6, 1);\np.G = zeros(6);\np.roll_table = zeros(2, 0);\n"
        )
    return f"""p.surface = true;   % floating vessel: g = G * (eta - eta_rest), roll from roll_table
p.eta_rest = {_mat(dyn.eta_rest.reshape(-1, 1))};   % floating at rest [x y z phi theta psi]
p.G = {_mat(dyn.g_matrix)};   % hydrostatic stiffness about the rest position
p.roll_table = {_mat(dyn.roll_table)};   % heel (rad); roll restoring (N m)
"""


def _save_linear(model: VehicleModel, speed: float, path: Path) -> None:
    dyn = model.dynamics()
    state0 = dyn.rest_state.copy()
    state0[6] = speed
    tau0 = np.zeros(6)
    tau0[0] = dyn.d_lin[0] * speed + dyn.d_quad[0] * speed**2  # force to hold the speed
    a, b = linearize(dyn, state0, tau0)
    savemat(path, {"A": a, "B": b, "Bu": b @ dyn.b_thr, "x0": state0, "tau0": tau0, "speed": speed})


def _simulink_text(name: str) -> str:
    """A script that builds <name>_sim.slx: tau (6x1) in, state x = [eta; nu] (12x1) out."""
    mdl = f"{name}_sim"
    return f"""%{name.upper()}_BUILD_SIMULINK  Build the Simulink model {mdl}.slx for '{name}'.
%   Input  tau (6x1): generalised force [X Y Z K M N] in the body frame (N, N m).
%   Output x (12x1): [x y z phi theta psi u v w p q r] (NED position, body velocities).
%   The dynamics block calls robomodel_dynamics with {name}_params (edit the params
%   function or p.delta / p.current_ned there). Run this script once, then use {mdl}.slx.
mdl = '{mdl}';
if bdIsLoaded(mdl), close_system(mdl, 0); end
if exist([mdl '.slx'], 'file'), delete([mdl '.slx']); end
new_system(mdl);
add_block('simulink/Sources/In1', [mdl '/tau'], 'Position', [30 60 60 74]);
set_param([mdl '/tau'], 'PortDimensions', '6');
add_block('simulink/User-Defined Functions/MATLAB Function', [mdl '/dynamics'], ...
    'Position', [120 40 230 100]);
chart = find(sfroot, '-isa', 'Stateflow.EMChart', 'Path', [mdl '/dynamics']);
chart.Script = sprintf(['function xdot = dynamics(x, tau)\\n%%#codegen\\n' ...
    'p = {name}_params();\\nxdot = robomodel_dynamics(0, x, tau, p);\\nend\\n']);
p0 = {name}_params();
x0 = [p0.eta_rest(:); zeros(6, 1)];   % at rest (floating position for a boat)
add_block('simulink/Continuous/Integrator', [mdl '/state'], ...
    'InitialCondition', mat2str(x0, 17), 'Position', [280 55 310 85]);
add_block('simulink/Sinks/Out1', [mdl '/x'], 'Position', [370 63 400 77]);
add_line(mdl, 'tau/1', 'dynamics/2');
add_line(mdl, 'state/1', 'dynamics/1', 'autorouting', 'on');
add_line(mdl, 'dynamics/1', 'state/1');
add_line(mdl, 'state/1', 'x/1');
set_param(mdl, 'Solver', 'ode45', 'RelTol', '1e-8', 'AbsTol', '1e-10', 'StopTime', '30');
save_system(mdl);
fprintf('built %s.slx\\n', mdl);
"""
