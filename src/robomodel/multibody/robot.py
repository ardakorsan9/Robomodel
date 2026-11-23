"""Robots made of links and joints (a kinematic tree), and their equations of motion

    M(q) q_ddot + h(q, q_dot) = tau + J^T f_ext,     h = C(q, q_dot) q_dot + g(q) + friction

computed with the standard recursive algorithms (Featherstone 2008):
- RNEA (recursive Newton-Euler): inverse dynamics tau = ID(q, q_dot, q_ddot)
- CRBA (composite rigid body): the joint-space mass matrix M(q)
- forward dynamics: q_ddot = M^-1 (tau - h)

Frames follow URDF: every link has a frame; a joint sits at `origin` (xyz, rpy) in its
parent link's frame, and the child link's frame is the joint frame after the joint moves.
A floating base (a free-moving vehicle) is six one-degree-of-freedom joints in a row:
x, y, z, roll, pitch, yaw (the same z-y-x Euler angles as the vehicle models).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.integrate import solve_ivp

from robomodel.multibody.spatial import (
    crf,
    crm,
    rot_axis,
    rpy_matrix,
    spatial_inertia,
    xform,
)

JOINT_TYPES = ("revolute", "continuous", "prismatic", "fixed")
GRAVITY = 9.80665


@dataclass(frozen=True)
class Link:
    name: str
    mass: float
    com: np.ndarray  # centre of mass in the link frame, m
    inertia: np.ndarray  # 3x3 about the centre of mass, link axes, kg m^2
    mesh: object | None = None  # optional trimesh for visuals / URDF


@dataclass(frozen=True)
class Joint:
    name: str
    type: str
    parent: str
    child: str
    xyz: np.ndarray = field(default_factory=lambda: np.zeros(3))
    rpy: np.ndarray = field(default_factory=lambda: np.zeros(3))
    axis: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 1.0]))
    lower: float = -np.inf
    upper: float = np.inf
    effort: float = np.inf
    damping: float = 0.0  # viscous joint friction, N m s/rad or N s/m (needs data)
    friction: float = 0.0  # Coulomb joint friction, N m or N (needs data)


class Robot:
    """A kinematic tree with a fixed or floating root."""

    def __init__(
        self,
        links: list[Link],
        joints: list[Joint],
        root: str,
        floating: bool = False,
        gravity=(0.0, 0.0, -GRAVITY),
    ):
        self.links = {link.name: link for link in links}
        if len(self.links) != len(links):
            raise ValueError("link names must be unique")
        if root not in self.links:
            raise ValueError(f"root link {root!r} not found")
        self.root = root
        self.floating = floating
        self.gravity = np.asarray(gravity, dtype=float)  # in the world (root) frame
        children: dict[str, list[Joint]] = {}
        seen_child = set()
        for j in joints:
            if j.type not in JOINT_TYPES:
                raise ValueError(f"joint {j.name}: type must be one of {JOINT_TYPES}")
            if j.parent not in self.links or j.child not in self.links:
                raise ValueError(f"joint {j.name}: unknown parent or child link")
            if j.child in seen_child or j.child == root:
                raise ValueError(f"joint {j.name}: link {j.child} has two parents (not a tree)")
            seen_child.add(j.child)
            children.setdefault(j.parent, []).append(j)
        if seen_child | {root} != set(self.links):
            missing = set(self.links) - seen_child - {root}
            raise ValueError(f"links not connected to the root: {sorted(missing)}")
        self.joints = joints
        self._build(children)

    # ---- tree arrays (one "body" per moving degree of freedom) -----------------------

    def _build(self, children) -> None:
        self.parent: list[int] = []  # parent body index, -1 = world
        self.x_tree: list[np.ndarray] = []  # parent frame -> joint frame (q = 0)
        self.axis: list[np.ndarray] = []
        self.kind: list[str] = []  # "R" revolute, "P" prismatic
        self.inertia: list[np.ndarray] = []  # spatial inertia of the body (link frame)
        self.dof_names: list[str] = []
        self.dof_joint: list[Joint | None] = []
        self.body_link: list[str | None] = []

        def add_dof(parent, x_tree, kind, axis, name, joint=None):
            self.parent.append(parent)
            self.x_tree.append(x_tree)
            self.kind.append(kind)
            self.axis.append(np.asarray(axis, dtype=float) / np.linalg.norm(axis))
            self.inertia.append(np.zeros((6, 6)))
            self.dof_names.append(name)
            self.dof_joint.append(joint)
            self.body_link.append(None)
            return len(self.parent) - 1

        root_body = -1
        if self.floating:  # x, y, z, roll, pitch, yaw in a row (z-y-x Euler)
            eye = np.eye(6)
            root_body = add_dof(-1, eye, "P", (1, 0, 0), "base_x")
            root_body = add_dof(root_body, eye, "P", (0, 1, 0), "base_y")
            root_body = add_dof(root_body, eye, "P", (0, 0, 1), "base_z")
            root_body = add_dof(root_body, eye, "R", (0, 0, 1), "base_yaw")
            root_body = add_dof(root_body, eye, "R", (0, 1, 0), "base_pitch")
            root_body = add_dof(root_body, eye, "R", (1, 0, 0), "base_roll")
        # link name -> (body index carrying it, transform from that body's frame to link)
        self._link_body: dict[str, tuple[int, np.ndarray]] = {self.root: (root_body, np.eye(6))}
        if root_body >= 0:
            self.body_link[root_body] = self.root
            self.inertia[root_body] = self._link_inertia(self.root)
        else:
            self._fixed_root_inertia = self._link_inertia(self.root)

        stack = [self.root]
        while stack:
            name = stack.pop()
            body, x_link = self._link_body[name]
            for j in children.get(name, []):
                x_joint = xform(rpy_matrix(*j.rpy), j.xyz) @ x_link
                if j.type == "fixed":
                    self._link_body[j.child] = (body, x_joint)
                    if body >= 0:
                        x_inv_t = x_joint.T  # inertia moves with X^T I X
                        self.inertia[body] = (
                            self.inertia[body] + x_inv_t @ self._link_inertia(j.child) @ x_joint
                        )
                else:
                    kind = "P" if j.type == "prismatic" else "R"
                    new = add_dof(body, x_joint, kind, j.axis, j.name, j)
                    self.body_link[new] = j.child
                    self.inertia[new] = self._link_inertia(j.child)
                    self._link_body[j.child] = (new, np.eye(6))
                stack.append(j.child)
        self.n = len(self.parent)

    def _link_inertia(self, name: str) -> np.ndarray:
        link = self.links[name]
        return spatial_inertia(link.mass, link.com, link.inertia)

    # ---- kinematics ---------------------------------------------------------------------

    def _joint_x(self, i: int, qi: float) -> tuple[np.ndarray, np.ndarray]:
        """Joint transform X_J(q) and motion subspace S (in the joint/child frame)."""
        a = self.axis[i]
        s = np.zeros(6)
        if self.kind[i] == "R":
            s[:3] = a
            # child frame rotated by +q about a: E = R^T
            return xform(rot_axis(a, qi), np.zeros(3)), s
        s[3:] = a
        return xform(np.eye(3), a * qi), s

    def _up(self, q):
        """X from parent body frame to each body frame, and motion subspaces."""
        xs, ss = [], []
        for i in range(self.n):
            xj, s = self._joint_x(i, q[i])
            xs.append(xj @ self.x_tree[i])
            ss.append(s)
        return xs, ss

    def link_transform(self, q, name: str) -> np.ndarray:
        """4x4 pose of a link frame in the world (root) frame."""
        xs, _ = self._up(q)
        body, x_link = self._link_body[name]
        x = x_link
        while body >= 0:
            x = x @ xs[body]
            body = self.parent[body]
        e = x[:3, :3]  # world -> link rotation
        r = -np.linalg.solve(e, x[3:, :3]) if np.any(x[3:, :3]) else np.zeros((3, 3))
        p = np.array([r[2, 1], r[0, 2], r[1, 0]])  # from -E S(p) = x[3:, :3]
        out = np.eye(4)
        out[:3, :3] = e.T
        out[:3, 3] = p
        return out

    # ---- dynamics -----------------------------------------------------------------------

    def _friction(self, qd) -> np.ndarray:
        out = np.zeros(self.n)
        for i, j in enumerate(self.dof_joint):
            if j is not None:
                out[i] = j.damping * qd[i] + j.friction * np.tanh(qd[i] / 1e-3)
        return out

    def inverse_dynamics(self, q, qd, qdd, f_ext=None, with_gravity=True) -> np.ndarray:
        """RNEA. f_ext: {body index: spatial force in that body's frame}."""
        q, qd, qdd = (np.asarray(a, dtype=float) for a in (q, qd, qdd))
        xs, ss = self._up(q)
        a_g = np.zeros(6)
        if with_gravity:
            a_g[3:] = -self.gravity  # fictitious upward acceleration of the base
        v = [None] * self.n
        a = [None] * self.n
        f = [None] * self.n
        for i in range(self.n):
            vj = ss[i] * qd[i]
            if self.parent[i] < 0:
                v[i] = vj
                a[i] = xs[i] @ a_g + ss[i] * qdd[i]
            else:
                p = self.parent[i]
                v[i] = xs[i] @ v[p] + vj
                a[i] = xs[i] @ a[p] + ss[i] * qdd[i] + crm(v[i]) @ vj
            f[i] = self.inertia[i] @ a[i] + crf(v[i]) @ self.inertia[i] @ v[i]
            if f_ext and i in f_ext:
                f[i] = f[i] - f_ext[i]
        tau = np.zeros(self.n)
        for i in reversed(range(self.n)):
            tau[i] = ss[i] @ f[i]
            if self.parent[i] >= 0:
                f[self.parent[i]] = f[self.parent[i]] + xs[i].T @ f[i]
        return tau + self._friction(qd)

    def mass_matrix(self, q) -> np.ndarray:
        """CRBA: joint-space mass matrix M(q), symmetric positive definite."""
        xs, ss = self._up(np.asarray(q, dtype=float))
        ic = [m.copy() for m in self.inertia]
        for i in reversed(range(self.n)):
            if self.parent[i] >= 0:
                ic[self.parent[i]] = ic[self.parent[i]] + xs[i].T @ ic[i] @ xs[i]
        m = np.zeros((self.n, self.n))
        for i in range(self.n):
            fi = ic[i] @ ss[i]
            m[i, i] = ss[i] @ fi
            j = i
            while self.parent[j] >= 0:
                fi = xs[j].T @ fi
                j = self.parent[j]
                m[i, j] = m[j, i] = ss[j] @ fi
        return m

    def bias(self, q, qd) -> np.ndarray:
        """h(q, q_dot): Coriolis, centrifugal, gravity and joint friction."""
        return self.inverse_dynamics(q, qd, np.zeros(self.n))

    def forward_dynamics(self, q, qd, tau, f_ext=None) -> np.ndarray:
        h = self.inverse_dynamics(q, qd, np.zeros(self.n), f_ext)
        return np.linalg.solve(self.mass_matrix(q), np.asarray(tau, dtype=float) - h)

    def energy(self, q, qd) -> tuple[float, float]:
        """(kinetic, potential) energy; potential is zero at the world origin height."""
        qd = np.asarray(qd, dtype=float)
        kinetic = 0.5 * qd @ self.mass_matrix(q) @ qd
        potential = 0.0
        for name, link in self.links.items():
            pose = self.link_transform(q, name)
            com_world = pose[:3, :3] @ link.com + pose[:3, 3]
            potential -= link.mass * self.gravity @ com_world
        return float(kinetic), float(potential)

    def simulate(
        self,
        t_end: float,
        q0=None,
        qd0=None,
        tau=None,
        dt_out: float = 0.01,
        rtol: float = 1e-9,
        atol: float = 1e-11,
    ):
        """Integrate the equations of motion. tau: constant vector or tau(t, q, qd)."""
        q0 = np.zeros(self.n) if q0 is None else np.asarray(q0, dtype=float)
        qd0 = np.zeros(self.n) if qd0 is None else np.asarray(qd0, dtype=float)
        if tau is None:
            tau = np.zeros(self.n)
        tau_fn = tau if callable(tau) else (lambda _t, _q, _qd, c=np.asarray(tau): c)  # noqa: B008

        def rhs(t, x):
            q, qd = x[: self.n], x[self.n :]
            return np.concatenate([qd, self.forward_dynamics(q, qd, tau_fn(t, q, qd))])

        times = np.arange(0.0, t_end + dt_out / 2, dt_out)
        sol = solve_ivp(
            rhs,
            (0.0, t_end),
            np.concatenate([q0, qd0]),
            t_eval=times,
            rtol=rtol,
            atol=atol,
            method="DOP853",
        )
        if not sol.success:
            raise RuntimeError(f"simulation failed: {sol.message}")
        return sol.t, sol.y[: self.n].T, sol.y[self.n :].T
