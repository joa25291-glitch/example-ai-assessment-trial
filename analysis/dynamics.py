"""Dynamic force analysis by the matrix method (Week 9 course convention).

Notation (as in the lecture "Help for: 6 Bar Linkage Mechanism"):
    F_ij  = force exerted BY link i ON link j      (so F_ji = -F_ij)
    R_ij  = position vector from the mass centre G_j of link j to the point
            where F_ij acts on link j
    T12   = driving torque applied by the ground (motor) on the crank, link 2
Gravity is neglected (dynamic forces only), as in the course examples.

Unknowns (14):
    F12x F12y F32x F32y F43x F43y F53x F53y F14x F14y F65x F65y F16y T12
Equations (14): 3 per link for links 2-5 (sum Fx, sum Fy, sum M about G),
2 for the slider 6 (sum Fx, sum Fy).  Slider friction is included as
F16x = mu_eff * F16y  (mu_eff = 0 for a frictionless slot).
"""
import numpy as np
from params import P as DEFAULT_P
import kinematics as K

UNKNOWNS = ["F12x", "F12y", "F32x", "F32y", "F43x", "F43y", "F53x", "F53y",
            "F14x", "F14y", "F65x", "F65y", "F16y", "T12"]


def r_vectors(pos):
    """Course R-vectors (from each link's G to each joint) as complex numbers."""
    return dict(
        R12=pos["O"] - pos["G2"], R32=pos["A"] - pos["G2"],
        R23=pos["A"] - pos["G3"], R43=pos["B"] - pos["G3"], R53=pos["C"] - pos["G3"],
        R34=pos["B"] - pos["G4"], R14=pos["E"] - pos["G4"],
        R35=pos["C"] - pos["G5"], R65=pos["D"] - pos["G5"],
    )


def build_matrix(pos, acc, p=DEFAULT_P, mu_eff=0.0):
    R = r_vectors(pos)
    x = {k: v.real for k, v in R.items()}
    y = {k: v.imag for k, v in R.items()}
    A = np.zeros((14, 14))
    c = np.zeros(14)
    col = {name: i for i, name in enumerate(UNKNOWNS)}

    def put(row, name, val):
        A[row, col[name]] += val

    # ---- link 2 ----
    put(0, "F12x", 1); put(0, "F32x", 1); c[0] = p.m2 * acc["aG2"].real
    put(1, "F12y", 1); put(1, "F32y", 1); c[1] = p.m2 * acc["aG2"].imag
    # T12 + (R12 x F12) + (R32 x F32) = I2 alpha2
    put(2, "F12x", -y["R12"]); put(2, "F12y", x["R12"])
    put(2, "F32x", -y["R32"]); put(2, "F32y", x["R32"])
    put(2, "T12", 1);          c[2] = p.I2 * acc["alpha2"]
    # ---- link 3 :  F23 = -F32 ----
    put(3, "F32x", -1); put(3, "F43x", 1); put(3, "F53x", 1); c[3] = p.m3 * acc["aG3"].real
    put(4, "F32y", -1); put(4, "F43y", 1); put(4, "F53y", 1); c[4] = p.m3 * acc["aG3"].imag
    # -(R23 x F32) + (R43 x F43) + (R53 x F53) = I3 alpha3
    put(5, "F32x", y["R23"]);  put(5, "F32y", -x["R23"])
    put(5, "F43x", -y["R43"]); put(5, "F43y", x["R43"])
    put(5, "F53x", -y["R53"]); put(5, "F53y", x["R53"]); c[5] = p.I3 * acc["alpha3"]
    # ---- link 4 :  F34 = -F43 ----
    put(6, "F43x", -1); put(6, "F14x", 1); c[6] = p.m4 * acc["aG4"].real
    put(7, "F43y", -1); put(7, "F14y", 1); c[7] = p.m4 * acc["aG4"].imag
    # -(R34 x F43) + (R14 x F14) = I4 alpha4
    put(8, "F43x", y["R34"]);  put(8, "F43y", -x["R34"])
    put(8, "F14x", -y["R14"]); put(8, "F14y", x["R14"]); c[8] = p.I4 * acc["alpha4"]
    # ---- link 5 :  F35 = -F53 ----
    put(9, "F53x", -1);  put(9, "F65x", 1);  c[9] = p.m5 * acc["aG5"].real
    put(10, "F53y", -1); put(10, "F65y", 1); c[10] = p.m5 * acc["aG5"].imag
    # -(R35 x F53) + (R65 x F65) = I5 alpha5
    put(11, "F53x", y["R35"]);  put(11, "F53y", -x["R35"])
    put(11, "F65x", -y["R65"]); put(11, "F65y", x["R65"]); c[11] = p.I5 * acc["alpha5"]
    # ---- link 6 (slider) :  F56 = -F65 ,  F16x = mu_eff * F16y ----
    put(12, "F65x", -1); put(12, "F16y", mu_eff); c[12] = p.m6 * acc["aG6"].real
    put(13, "F65y", -1); put(13, "F16y", 1);      c[13] = p.m6 * acc["aG6"].imag
    return A, c


def solve_forces(pos, vel, acc, p=DEFAULT_P, mu=0.0):
    """Solve the 14x14 system.  With mu > 0 the friction force on the slider is
    F16x = -mu*|F16y|*sign(V_D) (Coulomb friction opposes sliding).  Because the
    matrix needs F16x = mu_eff*F16y, the sign of F16y is found iteratively."""
    mu_eff = 0.0
    for _ in range(5):
        A, c = build_matrix(pos, acc, p, mu_eff)
        sol = np.linalg.solve(A, c)
        if mu == 0.0:
            break
        F16y = sol[UNKNOWNS.index("F16y")]
        new_mu_eff = -mu * np.sign(vel["VD"].real) * np.sign(F16y)
        if np.isclose(new_mu_eff, mu_eff):
            break
        mu_eff = new_mu_eff
    out = dict(zip(UNKNOWNS, sol))
    out["F16x"] = mu_eff * out["F16y"]
    out["mu_eff"] = mu_eff
    out["matrix"] = A
    out["rhs"] = c
    return out


def shaking(pos, F):
    """Shaking force and moment transmitted to the frame (Week 10 template).
    Fs = F21 + F41 + F61 ;  Ms(about O) = T21 + R_E x F41 + R_D x F61."""
    F21 = -(F["F12x"] + 1j * F["F12y"])
    F41 = -(F["F14x"] + 1j * F["F14y"])
    F61 = -(F["F16x"] + 1j * F["F16y"])
    T21 = -F["T12"]
    Fs = F21 + F41 + F61
    cross = lambda r, f: r.real * f.imag - r.imag * f.real
    Ms = T21 + cross(pos["E"], F41) + cross(pos["D"], F61)
    return dict(F21=F21, F41=F41, F61=F61, T21=T21, Fs=Fs, Ms=Ms)


def inertia_checks(pos, vel, acc, p=DEFAULT_P):
    """Independent checks: power balance for T12 and inertia-force sums for the
    shaking force/moment (valid for the frictionless case)."""
    ms = {2: p.m2, 3: p.m3, 4: p.m4, 5: p.m5, 6: p.m6}
    Is = {2: p.I2, 3: p.I3, 4: p.I4, 5: p.I5, 6: 0.0}
    al = {2: acc["alpha2"], 3: acc["alpha3"], 4: acc["alpha4"], 5: acc["alpha5"], 6: 0.0}
    om = {2: vel["omega2"], 3: vel["omega3"], 4: vel["omega4"], 5: vel["omega5"], 6: 0.0}
    dot = lambda u, v: u.real * v.real + u.imag * v.imag
    cross = lambda r, f: r.real * f.imag - r.imag * f.real
    power = sum(ms[i] * dot(acc[f"aG{i}"], vel[f"VG{i}"]) + Is[i] * al[i] * om[i] for i in range(2, 7))
    T12_power = power / vel["omega2"]
    Fs_inertia = -sum(ms[i] * acc[f"aG{i}"] for i in range(2, 7))
    Ms_inertia = -sum(cross(pos[f"G{i}"], ms[i] * acc[f"aG{i}"]) + Is[i] * al[i] for i in range(2, 7))
    return dict(T12_power=T12_power, Fs_inertia=Fs_inertia, Ms_inertia=Ms_inertia)


if __name__ == "__main__":
    th = np.radians(DEFAULT_P.theta2_deg)
    pos, vel, acc = K.solve(th)
    for mu in (0.0, 0.3):
        F = solve_forces(pos, vel, acc, mu=mu)
        print(f"--- mu = {mu} (mu_eff = {F['mu_eff']:+.2f}) ---")
        for k in UNKNOWNS + ["F16x"]:
            print(f"{k:5s} = {F[k]:12.3f}")
        S = shaking(pos, F)
        print("Fs =", S["Fs"], abs(S["Fs"]), np.degrees(np.angle(S["Fs"])), " Ms =", S["Ms"])
    chk = inertia_checks(pos, vel, acc)
    print("power-balance T12 =", chk["T12_power"], " Fs(inertia) =", chk["Fs_inertia"], " Ms(inertia) =", chk["Ms_inertia"])
