"""Independent recheck of the worked solution.

Recomputes the results from first principles with DIFFERENT methods from the
main scripts (nothing is imported from them) and compares with
output/results.json:

  positions      circle-circle and circle-line intersections (no half-angle formula)
  velocities     5-point central finite differences of the position solver in time
  accelerations  the same, second derivative
  ICs            Kennedy lines intersected by a generic two-line solver
  forces         Newton-Euler assembled with a different unknown set and sign
                 convention (each pin force taken "on the next link"), then
                 mapped to the course notation F_ij
  torque         also from the power (virtual-work) balance
  shaking        from the solved ground reactions and from -sum(m a_G)
  balancing      Berkof-Lowen residual must equal -(m6 + m5/2) a_D

Run from the repository root:  python3 analysis/verify_independent.py
                (mock test:     MIET_PRESET=mock2 python3 analysis/verify_independent.py)
"""
import json
import math
import os
import sys

import numpy as np
from scipy.optimize import brentq

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PRESET = os.environ.get("MIET_PRESET", "example1")
R = json.load(open(os.path.join(ROOT, "output" if PRESET == "example1" else "output_" + PRESET, "results.json")))
print("checking preset:", R["preset"]["name"], "-", R["preset"]["label"])
assert R["preset"]["name"] == PRESET

# ---------------------------------------------------------------- inputs (handout + selection rules)
pr = R["params"]
OA, AB, BE = pr["OA"], pr["AB"], pr["BE"]
beta, th1 = math.radians(pr["beta_deg"]), math.radians(pr["theta1_deg"])
n2 = pr["n2_rpm"]
OE = CD = 4 * OA                     # data-selection rules, re-applied here
AC = 0.6 * AB
H = OA / 4
w2 = 2 * math.pi * n2 / 60
m = dict(m2=1.0, m3=2.5, m4=1.5, m5=1.8, m6=0.9)          # handout item 6
I = dict(I2=0.002, I3=0.008, I4=0.005, I5=0.006)
assert pr["masses"] == m and pr["inertias"] == I, "report used different mass properties from the handout"
E = OE * complex(math.cos(th1), math.sin(th1))

FAILS = []


def check(name, mine, theirs, tol, rel=False):
    mine, theirs = complex(mine), complex(theirs)
    err = abs(mine - theirs) / (abs(theirs) if rel and abs(theirs) > 0 else 1.0)
    ok = err <= tol
    if not ok:
        FAILS.append(name)
    def fmt(z):
        return f"{z.real:.6g}" if abs(z.imag) < 1e-12 else f"({z.real:.6g}, {z.imag:.6g})"
    print(f"  {'ok ' if ok else 'BAD'}  {name:34s} independent={fmt(mine):>26s}  report={fmt(theirs):>26s}  err={err:.1e}")
    return ok


cross = lambda r, f: r.real * f.imag - r.imag * f.real
dot = lambda u, v: u.real * v.real + u.imag * v.imag
cpx = lambda xy: complex(xy[0], xy[1])


# ---------------------------------------------------------------- independent position solver
def pos(th2):
    A = OA * complex(math.cos(th2), math.sin(th2))
    d = abs(E - A)
    a = (AB ** 2 - BE ** 2 + d ** 2) / (2 * d)
    h = math.sqrt(AB ** 2 - a ** 2)
    u = (E - A) / d
    B = A + a * u + 1j * h * u           # intersection on the left of A->E (open branch, B above E)
    t3 = math.atan2((B - A).imag, (B - A).real)
    t4 = math.atan2((B - E).imag, (B - E).real)
    C = A + AC * complex(math.cos(t3 - beta), math.sin(t3 - beta))
    dy = -H - C.imag
    D = complex(C.real + math.sqrt(CD ** 2 - dy ** 2), -H)
    t5 = math.atan2((D - C).imag, (D - C).real)
    G3 = A + (AB / 2) * complex(math.cos(t3 - beta / 2), math.sin(t3 - beta / 2))
    return dict(A=A, B=B, C=C, D=D, E=E, O=0j, G2=A / 2, G3=G3, G4=(E + B) / 2, G5=(C + D) / 2, G6=D,
                t3=t3, t4=t4, t5=t5)


def kin(th2_deg, dth=1e-3):
    """Positions, velocities, accelerations by 5-point finite differences in time."""
    th = math.radians(th2_deg)
    h = dth / w2
    P = [pos(th + k * dth) for k in (-2, -1, 0, 1, 2)]
    out = dict(P[2])
    for key in ("A", "B", "C", "D", "G2", "G3", "G4", "G5", "G6"):
        f = [p[key] for p in P]
        out["V" + key] = (f[0] - 8 * f[1] + 8 * f[3] - f[4]) / (12 * h)
        out["a" + key] = (-f[0] + 16 * f[1] - 30 * f[2] + 16 * f[3] - f[4]) / (12 * h * h)
    for key in ("t3", "t4", "t5"):
        f = np.unwrap([p[key] for p in P])
        out["w" + key[1]] = (f[0] - 8 * f[1] + 8 * f[3] - f[4]) / (12 * h)
        out["al" + key[1]] = (-f[0] + 16 * f[1] - 30 * f[2] + 16 * f[3] - f[4]) / (12 * h * h)
    return out


# ---------------------------------------------------------------- independent force solver
NAMES = ["ROx", "ROy", "RAx", "RAy", "RBx", "RBy", "REx", "REy", "RCx", "RCy", "RDx", "RDy", "N", "T"]


def forces(k, mu_eff=0.0):
    col = {n_: i for i, n_ in enumerate(NAMES)}
    M = np.zeros((14, 14)); rhs = np.zeros(14); row = [0]

    def eq(terms, value):
        for name, coef in terms:
            M[row[0], col[name]] += coef
        rhs[row[0]] = value
        row[0] += 1

    def force_terms(Rname, sign):           # sign*R as x / y rows
        return [(Rname + "x", sign)], [(Rname + "y", sign)]

    def moment_terms(r, Rname, sign):       # r x (sign*R) = sign*(rx*Ry - ry*Rx)
        return [(Rname + "y", sign * r.real), (Rname + "x", -sign * r.imag)]

    # link 2: +R_O (ground), -R_A (reaction of link 3), +T
    eq([("ROx", 1), ("RAx", -1)], m["m2"] * k["aG2"].real)
    eq([("ROy", 1), ("RAy", -1)], m["m2"] * k["aG2"].imag)
    eq(moment_terms(k["O"] - k["G2"], "RO", 1) + moment_terms(k["A"] - k["G2"], "RA", -1) + [("T", 1)], 0.0)
    # link 3: +R_A, -R_B, -R_C
    eq([("RAx", 1), ("RBx", -1), ("RCx", -1)], m["m3"] * k["aG3"].real)
    eq([("RAy", 1), ("RBy", -1), ("RCy", -1)], m["m3"] * k["aG3"].imag)
    eq(moment_terms(k["A"] - k["G3"], "RA", 1) + moment_terms(k["B"] - k["G3"], "RB", -1)
       + moment_terms(k["C"] - k["G3"], "RC", -1), I["I3"] * k["al3"])
    # link 4: +R_B, +R_E (ground)
    eq([("RBx", 1), ("REx", 1)], m["m4"] * k["aG4"].real)
    eq([("RBy", 1), ("REy", 1)], m["m4"] * k["aG4"].imag)
    eq(moment_terms(k["B"] - k["G4"], "RB", 1) + moment_terms(k["E"] - k["G4"], "RE", 1), I["I4"] * k["al4"])
    # link 5: +R_C, -R_D
    eq([("RCx", 1), ("RDx", -1)], m["m5"] * k["aG5"].real)
    eq([("RCy", 1), ("RDy", -1)], m["m5"] * k["aG5"].imag)
    eq(moment_terms(k["C"] - k["G5"], "RC", 1) + moment_terms(k["D"] - k["G5"], "RD", -1), I["I5"] * k["al5"])
    # link 6: +R_D, ground normal N (vertical) + friction mu_eff*N (horizontal)
    eq([("RDx", 1), ("N", mu_eff)], m["m6"] * k["aG6"].real)
    eq([("RDy", 1), ("N", 1)], m["m6"] * k["aG6"].imag)
    x = dict(zip(NAMES, np.linalg.solve(M, rhs)))
    c = lambda n_: complex(x[n_ + "x"], x[n_ + "y"])
    # map to the course notation F_ij = force of link i on link j
    return dict(F12=c("RO"), F32=-c("RA"), F43=-c("RB"), F14=c("RE"), F53=-c("RC"), F65=-c("RD"),
                F16=complex(mu_eff * x["N"], x["N"]), T12=x["T"])


def solve_mu(k, mu):
    mu_eff = 0.0
    for _ in range(6):
        F = forces(k, mu_eff)
        new = -mu * np.sign(k["VD"].real) * np.sign(F["F16"].imag)
        if abs(new - mu_eff) < 1e-12:
            break
        mu_eff = new
    return forces(k, mu_eff), mu_eff


def inertia_sums(k):
    ms = [m["m2"], m["m3"], m["m4"], m["m5"], m["m6"]]
    Is = [I["I2"], I["I3"], I["I4"], I["I5"], 0.0]
    G = ["G2", "G3", "G4", "G5", "G6"]
    al = [0.0, k["al3"], k["al4"], k["al5"], 0.0]
    om = [w2, k["w3"], k["w4"], k["w5"], 0.0]
    Fs = -sum(mi * k["a" + g] for mi, g in zip(ms, G))
    Ms = -sum(cross(k[g], mi * k["a" + g]) + Ii * a for mi, g, Ii, a in zip(ms, G, Is, al))
    P = sum(mi * dot(k["a" + g], k["V" + g]) + Ii * a * o for mi, g, Ii, a, o in zip(ms, G, Is, al, om))
    return Fs, Ms, P / w2


def line_x(p1, d1, p2, d2):
    Mx = np.array([[d1.real, -d2.real], [d1.imag, -d2.imag]])
    t = np.linalg.solve(Mx, np.array([(p2 - p1).real, (p2 - p1).imag]))
    return p1 + t[0] * d1


# ================================================================= checks
a60 = R["analytical60"]
k = kin(60.0)
print("\n[1] Positions at theta2 = 60 deg (circle intersections)")
check("theta3 (deg)", math.degrees(k["t3"]), a60["theta3"], 1e-9)
check("theta4 (deg)", math.degrees(k["t4"]), a60["theta4"], 1e-9)
check("theta5 (deg)", math.degrees(k["t5"]), a60["theta5"], 1e-9)
check("S (m)", k["D"].real, a60["S"], 1e-12)
for key in ("A", "B", "C", "D", "G2", "G3", "G4", "G5"):
    check(f"position {key} (m)", k[key], cpx(a60["pos_" + key]), 1e-12)
check("|B-A| = AB", abs(k["B"] - k["A"]), AB, 1e-12)
check("|C-B| from triangle", abs(k["C"] - k["B"]), math.sqrt(AB**2 + AC**2 - 2*AB*AC*math.cos(beta)), 1e-12)

print("\n[2] Velocities (finite differences, relative tolerance 1e-7)")
for key, rk in (("w3", "omega3"), ("w4", "omega4"), ("w5", "omega5")):
    check(rk + " (rad/s)", k[key], a60[rk], 1e-7, rel=True)
for key in ("A", "B", "C", "D", "G2", "G3", "G4", "G5"):
    check(f"V{key} (m/s)", k["V" + key], cpx(a60["V" + key]), 1e-7, rel=True)

print("\n[3] Accelerations (finite differences, relative tolerance 1e-6)")
for key, rk in (("al3", "alpha3"), ("al4", "alpha4"), ("al5", "alpha5")):
    check(rk + " (rad/s2)", k[key], a60[rk], 1e-6, rel=True)
for key in ("A", "B", "C", "D", "G2", "G3", "G4", "G5"):
    check(f"a{key} (m/s2)", k["a" + key], cpx(a60["a" + key]), 1e-6, rel=True)

print("\n[4] Instantaneous centres (generic line intersections)")
V = 1j                                    # direction of every line through I16 (vertical)
I13 = line_x(0j, k["A"], E, k["B"] - E)
I24 = line_x(0j, E, k["A"], k["B"] - k["A"])
I15 = line_x(I13, k["C"] - I13, k["D"], V)
I36 = line_x(k["C"], k["D"] - k["C"], I13, V)
I25 = line_x(k["A"], k["C"] - k["A"], 0j, I15)
I45 = line_x(k["B"], k["C"] - k["B"], E, I15 - E)
I26 = line_x(0j, V, k["A"], I36 - k["A"])
I46 = line_x(E, V, k["B"], I36 - k["B"])
ics = R["sec4"]["ics"]
for name, z in (("13", I13), ("24", I24), ("15", I15), ("36", I36), ("25", I25), ("45", I45), ("26", I26), ("46", I46)):
    check("I" + name + " (m)", z, cpx(ics[name]), 1e-9)
w3_ic = (1j * w2 * k["A"] * np.conj(1j * (k["A"] - I13))).real / abs(k["A"] - I13) ** 2
check("omega3 from I13 (rad/s)", w3_ic, a60["omega3"], 1e-9)
check("|V_D| = w2*|O I26| (m/s)", w2 * abs(I26), abs(cpx(a60["VD"])), 1e-9)

print("\n[5] Joint forces and driving torque (independent Newton-Euler)")
F = forces(k)
s6 = R["sec6"]["0.0"]["F"]
for key in ("F12", "F32", "F43", "F14", "F53", "F65"):
    check(key + " (N)", F[key], complex(s6[key + "x"], s6[key + "y"]), 1e-4)
check("F16y (N)", F["F16"].imag, s6["F16y"], 1e-4)
check("T12 (N m)", F["T12"], s6["T12"], 1e-5)
Fs_in, Ms_in, T_pow = inertia_sums(k)
check("T12 by power balance (N m)", T_pow, s6["T12"], 1e-5)
F3, mu_eff = solve_mu(k, 0.3)
s63 = R["sec6"]["0.3"]["F"]
check("mu=0.3: effective mu'", mu_eff, s63["mu_eff"], 1e-12)
check("mu=0.3: F16x (N)", F3["F16"].real, s63["F16x"], 1e-4)
check("mu=0.3: F16y (N)", F3["F16"].imag, s63["F16y"], 1e-4)
check("mu=0.3: T12 (N m)", F3["T12"], s63["T12"], 1e-5)
check("mu=0.3: extra torque = |F16x V_D|/w2", F3["T12"] - F["T12"], abs(F3["F16"].real * k["VD"].real) / w2, 1e-6)

print("\n[6] Shaking force and moment at 60 deg")
F21, F41, F61 = -F["F12"], -F["F14"], -F["F16"]
Fs = F21 + F41 + F61
Ms = -F["T12"] + cross(E, F41) + cross(k["D"], F61)
s8 = R["sec8"]["at60"]
check("Fs from ground reactions (N)", Fs, complex(s8["Fsx"], s8["Fsy"]), 1e-4)
check("Fs = -sum(m a_G) (N)", Fs_in, complex(s8["Fsx"], s8["Fsy"]), 1e-4)
check("Ms about O (N m)", Ms, s8["Ms"], 1e-5)
check("Ms = -sum(r x m a + I alpha) (N m)", Ms_in, s8["Ms"], 1e-5)

print("\n[7] Sixteen positions (Table 4 / Table 15 / simulation Table 20)")
worst = dict(t3=0, t4=0, t5=0, S=0, w=0, al=0, T=0)
for row, ana, sim in zip(R["sec1"]["positions16"], R["analytical16"], R["sec7"]["rows16"]):
    kk = kin(row["theta2"])
    worst["t3"] = max(worst["t3"], abs(math.degrees(kk["t3"]) - row["theta3"]), abs(math.degrees(kk["t3"]) - sim["theta3"]))
    worst["t4"] = max(worst["t4"], abs(math.degrees(kk["t4"]) - row["theta4"]), abs(math.degrees(kk["t4"]) - sim["theta4"]))
    worst["t5"] = max(worst["t5"], abs(math.degrees(kk["t5"]) - row["theta5"]), abs(math.degrees(kk["t5"]) - sim["theta5"]))
    worst["S"] = max(worst["S"], abs(kk["D"].real - row["S"]), abs(kk["D"].real * 1000 - sim["S_mm"]) / 1000)
    worst["w"] = max(worst["w"], *(abs(kk["w" + i] - ana["w" + i]) / max(1, abs(ana["w" + i])) for i in "345"))
    worst["al"] = max(worst["al"], *(abs(kk["al" + i] - ana["al" + i]) / max(1, abs(ana["al" + i])) for i in "345"))
    worst["T"] = max(worst["T"], abs(forces(kk)["T12"] - sim["T12"]) / max(1, abs(sim["T12"])))
for key, tol in (("t3", 1e-8), ("t4", 1e-8), ("t5", 1e-8), ("S", 1e-10), ("w", 1e-6), ("al", 1e-5), ("T", 1e-5)):
    check("16-position worst error: " + key, worst[key], 0.0, tol)

print("\n[8] Limits of motion and ranges")
dS = lambda t: (pos(t + 1e-6)["D"].real - pos(t - 1e-6)["D"].real)
dT4 = lambda t: (pos(t + 1e-6)["t4"] - pos(t - 1e-6)["t4"])
grid = np.radians(np.linspace(0, 360, 721))
def roots(f):
    v = [f(t) for t in grid]
    return sorted(math.degrees(brentq(f, grid[i], grid[i + 1], xtol=1e-13)) % 360
                  for i in range(len(grid) - 1) if np.sign(v[i]) != np.sign(v[i + 1]))
lim = R["sec1"]["limits"]
for mine, theirs, name in zip(roots(dS), sorted(e["theta2"] for e in lim["slider6"]), ("slider limit 1", "slider limit 2")):
    check(name + " theta2 (deg)", mine, theirs, 1e-6)
for mine, theirs, name in zip(roots(dT4), sorted(e["theta2"] for e in lim["rocker4"]), ("rocker limit 1", "rocker limit 2")):
    check(name + " theta2 (deg)", mine, theirs, 1e-6)
# toggle positions by geometry: |OB| = AB + OA (extended) and AB - OA (folded)
for rOB, name in ((AB + OA, "extended toggle theta4"), (AB - OA, "folded toggle theta4")):
    d = abs(E); a = (rOB ** 2 - BE ** 2 + d ** 2) / (2 * d); h = math.sqrt(rOB ** 2 - a ** 2)
    Bt = a * E / d + 1j * h * E / d
    t4 = math.degrees(math.atan2((Bt - E).imag, (Bt - E).real))
    theirs = [e["theta4"] for e in lim["rocker4"]]
    check(name + " (deg)", t4, min(theirs, key=lambda x: abs(x - t4)), 1e-6)
Ss = [pos(t)["D"].real for t in np.radians(np.arange(0, 360, 0.001))]
Sl = sorted(e["S"] for e in lim["slider6"])
check("stroke (m), report uses exact limits", max(Ss) - min(Ss), Sl[1] - Sl[0], 1e-9)
# transmission angle of the four-bar (angle between AB and EB), fine grid, compared at the printed precision
gam = []
for t in np.radians(np.arange(0, 360, 0.01)):
    p_ = pos(t); g = abs((p_["t3"] - p_["t4"] + math.pi) % (2 * math.pi) - math.pi); gam.append(min(g, math.pi - g))
check("min transmission angle (deg, 1 dp)", round(math.degrees(min(gam)), 1), round(R["transmission"]["fourbar_min"], 1), 1e-9)
for key, idx in (("theta3", "t3"), ("theta4", "t4"), ("theta5", "t5")):
    vals = [math.degrees(pos(t)[idx]) for t in np.radians(np.arange(0, 360, 0.01))]
    check(f"{key} range (deg, 2 dp)", complex(round(min(vals), 2), round(max(vals), 2)),
          complex(round(R["ranges"][key][0], 2), round(R["ranges"][key][1], 2)), 1e-9)

print("\n[9] Over one revolution: torque, pin forces, shaking force, balancing")
ths = np.arange(0.0, 360.0, 0.5)
T_all, Fs_all, Ms_all, aD_all, pinmax = [], [], [], [], dict(F12=0, F32=0, F43=0, F14=0, F53=0, F65=0)
ks = [kin(t) for t in ths]
for kk in ks:
    Fk = forces(kk); fs, ms, _ = inertia_sums(kk)
    T_all.append(Fk["T12"]); Fs_all.append(fs); Ms_all.append(ms); aD_all.append(kk["aD"].real)
    for key in pinmax:
        pinmax[key] = max(pinmax[key], abs(Fk[key]))
T_all, Fs_all, Ms_all, aD_all = map(np.array, (T_all, Fs_all, Ms_all, aD_all))
sm = R["sec7"]["summary"]
check("T12 max (N m)", T_all.max(), sm["T12_max"], 1e-3, rel=True)
check("T12 min (N m)", T_all.min(), sm["T12_min"], 1e-3, rel=True)
check("T12 mean (N m)", T_all.mean(), 0.0, 1e-6)
for key in pinmax:
    check(key + " max over revolution (N)", pinmax[key], sm["pin_max"][key], 2e-3, rel=True)
un = R["sec8"]["stats"]["unbalanced"]
check("unbalanced Fs RMS (N)", np.sqrt(np.mean(np.abs(Fs_all) ** 2)), un["Fs_rms"], 1e-6, rel=True)
check("unbalanced Fs max (N)", np.abs(Fs_all).max(), un["Fs_max"], 1e-6, rel=True)
check("unbalanced Ms max (N m)", np.abs(Ms_all).max(), un["Ms_max"], 1e-6, rel=True)
# Berkof-Lowen counterweights from the report applied to the independent kinematics
des = R["sec8"]["designs"]["berkof_lowen"]
P2 = des["m2c"] * cpx(des["u2"]); P4 = des["m4c"] * cpx(des["u4"])
resid = 0.0
for kk, fs in zip(ks, Fs_all):
    th2 = math.atan2(kk["A"].imag, kk["A"].real)
    a2 = -w2 ** 2 * P2 * complex(math.cos(th2), math.sin(th2))               # crank counterweight accel * mass
    a4 = (1j * kk["al4"] - kk["w4"] ** 2) * P4 * complex(math.cos(kk["t4"]), math.sin(kk["t4"]))
    fs_bal = fs - a2 - a4
    resid = max(resid, abs(fs_bal - (-(m["m6"] + m["m5"] / 2) * kk["aD"].real)))
check("Berkof-Lowen residual = -(m6+m5/2) a_D (N)", resid, 0.0, 1e-3)

print("\n[10] Graphical constructions are geometrically exact before rounding")
KV, KA = R["scales"]["KV"], R["scales"]["KA"]           # drawing scales used for this parameter set
vm = R["sec2"]["measured"]
exact_v = {"V_A": k["VA"], "V_B": k["VB"], "V_C": k["VC"], "V_D": k["VD"], "V_B/A": k["VB"] - k["VA"],
           "V_C/A": k["VC"] - k["VA"], "V_C/B": k["VC"] - k["VB"], "V_D/C": k["VD"] - k["VC"],
           "V_G2": k["VG2"], "V_G3": k["VG3"], "V_G4": k["VG4"], "V_G5": k["VG5"]}
for name, z in exact_v.items():
    check(f"{name} drawn length (mm)", vm[name]["length_exact"], KV * abs(z), 1e-6)
am = R["sec3"]["measured"]
for name, key in (("a_A", "aA"), ("a_B", "aB"), ("a_C", "aC"), ("a_D", "aD"), ("a_G3", "aG3")):
    check(f"{name} graphical vs exact (rel)", am[name]["value"], abs(k[key]), 4e-3, rel=True)

print("\n" + ("ALL CHECKS PASSED" if not FAILS else f"{len(FAILS)} CHECK(S) FAILED: " + ", ".join(FAILS)))
sys.exit(1 if FAILS else 0)
