"""Independent computational model (Working-Model-equivalent multibody simulation).

Working Model 2D builds a model from rigid bodies, pin joints, a keyed slot
joint and a velocity motor, then integrates the constrained equations of motion
in time.  This module does the same thing from scratch, WITHOUT using the
closed-form loop equations of kinematics.py:

  * Cartesian coordinates q = (x, y, phi) of the mass centre of each moving
    body 2..6  ->  15 coordinates
  * 15 constraint equations Phi(q, t) = 0: six pin joints (O, A, B, E, C, D),
    the keyed slot (y6 = -H, phi6 = 0) and the motor driver phi2 = w2 t
  * Equations of motion  M q'' + Phi_q^T lambda = Q  (index-1 DAE with
    Baumgarte stabilisation) integrated with an explicit Runge-Kutta solver
  * Joint reaction forces, the slot reaction and the motor torque are obtained
    from the Lagrange multipliers lambda, exactly as a WM "force meter" would
    report them.

The results are compared with the analytical (closed-form + matrix) solution in
run_all.py.
"""
import numpy as np
from scipy.integrate import solve_ivp
from params import P as DEFAULT_P


def A_(phi):
    c, s = np.cos(phi), np.sin(phi)
    return np.array([[c, -s], [s, c]])


def B_(phi):  # dA/dphi
    c, s = np.cos(phi), np.sin(phi)
    return np.array([[-s, -c], [c, -s]])


class SixBarModel:
    """Bodies: 0->link2, 1->link3, 2->link4, 3->link5, 4->slider6."""

    def __init__(self, p=DEFAULT_P, gravity=0.0, theta2_0=0.0):
        self.p = p
        self.g = gravity
        self.th20 = theta2_0
        self.w2 = p.omega2
        l2, l3, l4, l5, b = p.l2, p.l3, p.l4, p.l5, p.beta
        # body-fixed joint coordinates (origin at the body's mass centre)
        self.s2O = np.array([-l2 / 2, 0.0]); self.s2A = np.array([l2 / 2, 0.0])
        G3l = np.array([p.g3_local.real, p.g3_local.imag])   # G3 in A-frame (x along AB)
        Cl = p.AC * np.array([np.cos(b), -np.sin(b)])
        self.s3A = -G3l
        self.s3B = np.array([l3, 0.0]) - G3l
        self.s3C = Cl - G3l
        self.s4E = np.array([-l4 / 2, 0.0]); self.s4B = np.array([l4 / 2, 0.0])
        self.s5C = np.array([-l5 / 2, 0.0]); self.s5D = np.array([l5 / 2, 0.0])
        self.E = p.l1 * np.array([np.cos(p.theta1), np.sin(p.theta1)])
        m = [p.m2, p.m3, p.m4, p.m5, p.m6]
        I = [p.I2, p.I3, p.I4, p.I5, 1e-3]  # slider rotation is locked; any inertia works
        self.M = np.diag(np.repeat(m, 3).astype(float))
        for k in range(5):
            self.M[3 * k + 2, 3 * k + 2] = I[k]
        self.m = m
        # list of pin joints: (name, body_i, s_i, body_j or None, s_j or ground point)
        self.pins = [
            ("O", 0, self.s2O, None, np.zeros(2)),
            ("A", 0, self.s2A, 1, self.s3A),
            ("B", 1, self.s3B, 2, self.s4B),
            ("E", 2, self.s4E, None, self.E),
            ("C", 1, self.s3C, 3, self.s5C),
            ("D", 3, self.s5D, 4, np.zeros(2)),
        ]

    # ---------------- constraint functions ----------------
    def Phi(self, q, t):
        out = []
        r = lambda k: q[3 * k:3 * k + 2]
        ph = lambda k: q[3 * k + 2]
        for _, i, si, j, sj in self.pins:
            v = r(i) + A_(ph(i)) @ si
            v = v - (r(j) + A_(ph(j)) @ sj if j is not None else sj)
            out.extend(v)
        out.append(q[3 * 4 + 1] + self.p.H)          # slider on y = -H
        out.append(q[3 * 4 + 2])                     # slider does not rotate
        out.append(q[2] - (self.th20 + self.w2 * t))  # motor driver
        return np.array(out)

    def Phi_q(self, q):
        Jm = np.zeros((15, 15))
        row = 0
        for _, i, si, j, sj in self.pins:
            Jm[row:row + 2, 3 * i:3 * i + 2] = np.eye(2)
            Jm[row:row + 2, 3 * i + 2] = B_(q[3 * i + 2]) @ si
            if j is not None:
                Jm[row:row + 2, 3 * j:3 * j + 2] = -np.eye(2)
                Jm[row:row + 2, 3 * j + 2] = -(B_(q[3 * j + 2]) @ sj)
            row += 2
        Jm[12, 13] = 1.0
        Jm[13, 14] = 1.0
        Jm[14, 2] = 1.0
        return Jm

    def Phi_t(self):
        v = np.zeros(15)
        v[14] = -self.w2
        return v

    def gamma(self, q, qd):
        """Right-hand side of Phi_q q'' = gamma."""
        g = np.zeros(15)
        row = 0
        for _, i, si, j, sj in self.pins:
            v = A_(q[3 * i + 2]) @ si * qd[3 * i + 2] ** 2
            if j is not None:
                v = v - A_(q[3 * j + 2]) @ sj * qd[3 * j + 2] ** 2
            g[row:row + 2] = v
            row += 2
        return g  # slider and driver rows: 0 (constant motor speed)

    # ---------------- assembly (like WM's initial "assemble") ----------------
    def assemble(self, q_guess, t=0.0, tol=1e-13, itmax=50):
        q = q_guess.copy()
        for _ in range(itmax):
            f = self.Phi(q, t)
            if np.linalg.norm(f) < tol:
                break
            q = q - np.linalg.solve(self.Phi_q(q), f)
        qd = np.linalg.solve(self.Phi_q(q), -self.Phi_t())
        return q, qd

    # ---------------- dynamics ----------------
    def Q_applied(self):
        Q = np.zeros(15)
        for k in range(5):
            Q[3 * k + 1] = -self.m[k] * self.g
        return Q

    def solve_acc_lambda(self, q, qd, t, alpha_b=20.0, beta_b=20.0):
        Jm = self.Phi_q(q)
        rhs_c = self.gamma(q, qd) - 2 * alpha_b * (Jm @ qd + self.Phi_t()) - beta_b ** 2 * self.Phi(q, t)
        Aug = np.block([[self.M, Jm.T], [Jm, np.zeros((15, 15))]])
        sol = np.linalg.solve(Aug, np.concatenate([self.Q_applied(), rhs_c]))
        return sol[:15], sol[15:]

    def rhs(self, t, z):
        q, qd = z[:15], z[15:]
        qdd, _ = self.solve_acc_lambda(q, qd, t)
        return np.concatenate([qd, qdd])

    def reactions(self, lam):
        """Convert Lagrange multipliers into course-notation forces/torque."""
        L = {}
        for k, (name, *_rest) in enumerate(self.pins):
            L[name] = lam[2 * k] + 1j * lam[2 * k + 1]
        return dict(
            F12=-L["O"], F32=-L["A"], F43=-L["B"], F14=-L["E"], F53=-L["C"], F65=-L["D"],
            F16y=-lam[12], M16=-lam[13], T12=-lam[14],
        )

    def run(self, n_rev=1.0, n_out=721, q_guess=None):
        if q_guess is None:
            raise ValueError("provide an approximate initial configuration")
        q0, qd0 = self.assemble(q_guess)
        T = n_rev * 2 * np.pi / self.w2
        t_eval = np.linspace(0.0, T, n_out)
        sol = solve_ivp(self.rhs, (0.0, T), np.concatenate([q0, qd0]), method="DOP853",
                        t_eval=t_eval, rtol=1e-11, atol=1e-12)
        results = []
        for k, t in enumerate(sol.t):
            q, qd = sol.y[:15, k], sol.y[15:, k]
            qdd, lam = self.solve_acc_lambda(q, qd, t)
            results.append(dict(t=t, q=q, qd=qd, qdd=qdd, lam=lam, F=self.reactions(lam),
                                residual=np.linalg.norm(self.Phi(q, t))))
        return results

    # ------------- helpers to read joint kinematics from body states -------------
    def point(self, q, body, s):
        return q[3 * body:3 * body + 2] + A_(q[3 * body + 2]) @ s

    def point_vel(self, q, qd, body, s):
        return qd[3 * body:3 * body + 2] + B_(q[3 * body + 2]) @ s * qd[3 * body + 2]

    def point_acc(self, q, qd, qdd, body, s):
        return (qdd[3 * body:3 * body + 2] + B_(q[3 * body + 2]) @ s * qdd[3 * body + 2]
                - A_(q[3 * body + 2]) @ s * qd[3 * body + 2] ** 2)
