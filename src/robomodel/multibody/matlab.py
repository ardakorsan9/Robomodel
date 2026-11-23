"""MATLAB files for a jointed robot.

No toolbox needed:
  <name>_params.m              tree, joint transforms, spatial inertias (robomodel's own data)
  robomodel_robot_mass.m       M(q) by the composite rigid body algorithm (Featherstone)
  robomodel_robot_bias.m       h(q, qd) = C qd + g + joint friction, recursive Newton-Euler
  robomodel_robot_dynamics.m   xdot = f(t, x, tau, p), x = [q; qd]
  <name>_example.m             checks M, g at reference poses and a 2 s swing against Python
With the Robotics System Toolbox:
  <name>_check_toolbox.m       importrobot(URDF), massMatrix / gravityTorque against the same
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from robomodel.matlab_export import matlab_name
from robomodel.multibody.robot import Robot

POSES = 6


def write_robot_matlab(robot: Robot, out_dir: str | Path, urdf_path: str) -> list[Path]:
    """urdf_path: the URDF relative to out_dir, with / (e.g. "../urdf/arm.urdf")."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    name = matlab_name(getattr(robot, "name", "robot"))
    rng = np.random.default_rng(1)
    n = robot.n
    rows = []
    zero = np.zeros(n)
    for k in range(POSES):
        q = zero if k == 0 else rng.uniform(-1.5, 1.5, n)
        m = robot.mass_matrix(q)
        g = robot.bias(q, zero)
        rows.append(np.concatenate([q, m.ravel(order="F"), g]))
    header = (
        [f"q_{d}" for d in robot.dof_names]
        + [f"M_{i + 1}_{j + 1}" for j in range(n) for i in range(n)]
        + [f"g_{d}" for d in robot.dof_names]
    )
    csv = out / f"{name}_reference.csv"
    np.savetxt(
        csv, np.array(rows), delimiter=",", header=",".join(header), comments="", fmt="%.15g"
    )
    names = ", ".join(f"'{d}'" for d in robot.dof_names)
    script = out / f"{name}_check_toolbox.m"
    script.write_text(
        f"""%{name.upper()}_CHECK_TOOLBOX Load the URDF into MATLAB and check it.
%   Needs the Robotics System Toolbox (importrobot). URDF: {urdf_path}
%   Equations: M(q) qdd + C(q, qd) qd + g(q) = tau; massMatrix, velocityProduct,
%   gravityTorque and forwardDynamics give every term for your controller.
here = fileparts(mfilename('fullpath'));
robot = importrobot(fullfile(here, {", ".join(repr(x) for x in urdf_path.split("/"))}));
robot.DataFormat = 'column';
robot.Gravity = [0 0 -9.80665];
ref = readmatrix(fullfile(here, '{name}_reference.csv'), 'NumHeaderLines', 1);
ours = {{{names}}};   % robomodel's joint order
n = numel(ours);
% MATLAB's joint order: bodies with a moving joint, in the order of robot.Bodies
theirs = {{}};
for b = 1:robot.NumBodies
    if ~strcmp(robot.Bodies{{b}}.Joint.Type, 'fixed')
        theirs{{end + 1}} = robot.Bodies{{b}}.Joint.Name; %#ok<AGROW>
    end
end
idx = cellfun(@(j) find(strcmp(ours, j)), theirs);   % theirs(k) = ours(idx(k))
errM = 0; errg = 0;
for k = 1:size(ref, 1)
    q = ref(k, 1:n)';
    Mr = reshape(ref(k, n + 1:n + n * n), n, n);
    gr = ref(k, n + n * n + 1:end)';
    M = massMatrix(robot, q(idx));
    g = gravityTorque(robot, q(idx));
    errM = max(errM, max(abs(M - Mr(idx, idx)), [], 'all'));
    errg = max(errg, max(abs(g - gr(idx))));
end
fprintf('robomodel check: max difference M(q) = %.3g, g(q) = %.3g\\n', errM, errg);
""",
        encoding="utf-8",
    )
    written = [csv, script] + _toolbox_free(robot, out, name)
    return written


def _m(a) -> str:
    a = np.atleast_2d(np.asarray(a, dtype=float))
    return "[" + "; ".join(" ".join(f"{x:.17g}" for x in row) for row in a) + "]"


def _toolbox_free(robot: Robot, out: Path, name: str) -> list[Path]:
    n = robot.n
    lines = [
        f"function p = {name}_params()",
        f"%{name.upper()}_PARAMS  Jointed robot from robomodel.",
        "%   Body i moves with joint i; parent(i) = 0 is the fixed world (or base).",
        "%   Xtree: parent frame -> joint frame at q = 0 (Plucker, motion vectors).",
        f"p.n = {n};",
        f"p.names = {{{', '.join(repr(d) for d in robot.dof_names)}}};",
        f"p.parent = {_m(np.array(robot.parent) + 1)};",
        f"p.revolute = logical({_m([1.0 if k == 'R' else 0.0 for k in robot.kind])});",
        f"p.axis = {_m(np.array(robot.axis).T)};   % joint axes (3 x n)",
        f"p.gravity = {_m(np.asarray(robot.gravity).reshape(-1, 1))};",
    ]
    damp = [j.damping if j is not None else 0.0 for j in robot.dof_joint]
    fric = [j.friction if j is not None else 0.0 for j in robot.dof_joint]
    lines += [
        f"p.damping = {_m(np.reshape(damp, (-1, 1)))};",
        f"p.friction = {_m(np.reshape(fric, (-1, 1)))};",
    ]
    lines += [f"p.Xtree = zeros(6, 6, {n});", f"p.I = zeros(6, 6, {n});"]
    for i in range(n):
        lines.append(f"p.Xtree(:, :, {i + 1}) = {_m(robot.x_tree[i])};")
        lines.append(f"p.I(:, :, {i + 1}) = {_m(robot.inertia[i])};")
    lines.append("end")
    files = {f"{name}_params.m": "\n".join(lines) + "\n"}
    files["robomodel_robot_mass.m"] = _MASS + _HELPERS
    files["robomodel_robot_bias.m"] = _BIAS + _HELPERS
    files["robomodel_robot_dynamics.m"] = _DYN

    # 2 s free swing from a tilted pose, for the example's check
    q0 = np.full(n, 0.3)
    t, q, qd = robot.simulate(2.0, q0=q0, dt_out=0.02)
    ref = out / f"{name}_swing.csv"
    np.savetxt(
        ref,
        np.column_stack([t, q, qd]),
        delimiter=",",
        comments="",
        fmt="%.15g",
        header=",".join(
            ["t"] + [f"q_{d}" for d in robot.dof_names] + [f"qd_{d}" for d in robot.dof_names]
        ),
    )
    files[
        f"{name}_example.m"
    ] = f"""%{name.upper()}_EXAMPLE  M(q), g(q) and a free swing of the robot, no toolbox needed.
%   Equations: M(q) qdd + h(q, qd) = tau, h = Coriolis + gravity + joint friction.
p = {name}_params();
n = p.n;
ref = readmatrix('{name}_reference.csv', 'NumHeaderLines', 1);
errM = 0; errg = 0;
for k = 1:size(ref, 1)
    q = ref(k, 1:n)';
    Mr = reshape(ref(k, n + 1:n + n * n), n, n);
    gr = ref(k, n + n * n + 1:end)';
    errM = max(errM, max(abs(robomodel_robot_mass(q, p) - Mr), [], 'all'));
    errg = max(errg, max(abs(robomodel_robot_bias(q, zeros(n, 1), p) - gr)));
end
fprintf('robomodel check: max difference M(q) = %.3g, g(q) = %.3g\\n', errM, errg);
sw = readmatrix('{name}_swing.csv', 'NumHeaderLines', 1);
opts = odeset('RelTol', 1e-10, 'AbsTol', 1e-12);
x0 = [sw(1, 2:n + 1)'; zeros(n, 1)];
[t, x] = ode45(@(t, x) robomodel_robot_dynamics(t, x, zeros(n, 1), p), sw(:, 1), x0, opts);
err = max(abs(x - sw(:, 2:end)), [], 'all');
fprintf('free swing, 2 s: max difference MATLAB vs Python = %.3g rad\\n', err);
figure('Name', 'swing'); plot(t, x(:, 1:n)); grid on; legend(p.names);
xlabel('time [s]'); ylabel('q [rad]');
"""
    written = []
    for fname, text in files.items():
        (out / fname).write_text(text, encoding="utf-8")
        written.append(out / fname)
    return written + [ref]


_HELPERS = """
function [X, S] = joint_x(i, qi, p)
a = p.axis(:, i);
S = zeros(6, 1);
if p.revolute(i)
    S(1:3) = a;
    E = rot_axis(a, qi)';
    X = [E, zeros(3); zeros(3), E];
else
    S(4:6) = a;
    X = [eye(3), zeros(3); -skew(a * qi), eye(3)];
end
end

function R = rot_axis(a, ang)
K = skew(a / norm(a));
R = eye(3) + sin(ang) * K + (1 - cos(ang)) * (K * K);
end

function M = skew(v)
M = [0, -v(3), v(2); v(3), 0, -v(1); -v(2), v(1), 0];
end

function C = crm(v)
C = [skew(v(1:3)), zeros(3); skew(v(4:6)), skew(v(1:3))];
end
"""

_MASS = """function M = robomodel_robot_mass(q, p)
%ROBOMODEL_ROBOT_MASS  Joint-space mass matrix M(q), composite rigid body algorithm.
n = p.n;
Xup = zeros(6, 6, n); S = zeros(6, n);
for i = 1:n
    [XJ, S(:, i)] = joint_x(i, q(i), p);
    Xup(:, :, i) = XJ * p.Xtree(:, :, i);
end
Ic = p.I;
for i = n:-1:1
    if p.parent(i) > 0
        j = p.parent(i);
        Ic(:, :, j) = Ic(:, :, j) + Xup(:, :, i)' * Ic(:, :, i) * Xup(:, :, i);
    end
end
M = zeros(n);
for i = 1:n
    F = Ic(:, :, i) * S(:, i);
    M(i, i) = S(:, i)' * F;
    j = i;
    while p.parent(j) > 0
        F = Xup(:, :, j)' * F;
        j = p.parent(j);
        M(i, j) = S(:, j)' * F;
        M(j, i) = M(i, j);
    end
end
end
"""

_BIAS = """function h = robomodel_robot_bias(q, qd, p)
%ROBOMODEL_ROBOT_BIAS  h(q, qd): Coriolis, centrifugal, gravity and joint friction (RNEA).
n = p.n;
a_g = [zeros(3, 1); -p.gravity(:)];   % the base accelerates up instead of gravity pulling
v = zeros(6, n); a = zeros(6, n); f = zeros(6, n); Xup = zeros(6, 6, n); S = zeros(6, n);
for i = 1:n
    [XJ, S(:, i)] = joint_x(i, q(i), p);
    Xup(:, :, i) = XJ * p.Xtree(:, :, i);
    vJ = S(:, i) * qd(i);
    if p.parent(i) == 0
        v(:, i) = vJ;
        a(:, i) = Xup(:, :, i) * a_g;
    else
        v(:, i) = Xup(:, :, i) * v(:, p.parent(i)) + vJ;
        a(:, i) = Xup(:, :, i) * a(:, p.parent(i)) + crm(v(:, i)) * vJ;
    end
    f(:, i) = p.I(:, :, i) * a(:, i) - crm(v(:, i))' * p.I(:, :, i) * v(:, i);
end
h = zeros(n, 1);
for i = n:-1:1
    h(i) = S(:, i)' * f(:, i);
    if p.parent(i) > 0
        f(:, p.parent(i)) = f(:, p.parent(i)) + Xup(:, :, i)' * f(:, i);
    end
end
h = h + p.damping .* qd(:) + p.friction .* tanh(qd(:) / 1e-3);
end
"""

_DYN = """function xdot = robomodel_robot_dynamics(~, x, tau, p)
%ROBOMODEL_ROBOT_DYNAMICS  x = [q; qd]: M(q) qdd + h(q, qd) = tau.
n = p.n;
q = x(1:n); qd = x(n + 1:end);
qdd = robomodel_robot_mass(q, p) \\ (tau(:) - robomodel_robot_bias(q, qd, p));
xdot = [qd; qdd];
end
"""
