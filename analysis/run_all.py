"""Run every section and collect the numbers used by the report.

    python3 analysis/run_all.py                       # example 1   -> output/
    MIET_PRESET=mock2 python3 analysis/run_all.py     # mock test   -> output_mock2/

Outputs: <out>/figures/*.svg|png, <out>/tables/*.csv, <out>/results.json
"""
import dataclasses
import json
import os
import numpy as np
from params import P
import layout as L
import kinematics as K
import ic as IC
import sec1_position as S1
import sec2_velocity as S2
import sec3_acceleration as S3
import sec4_ic as S4
import sec6_fbd as S6
import sec7_simulation as S7
import sec8_balancing as S8

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), P.out_dir)


def cnum(z):
    return [float(np.real(z)), float(np.imag(z))]


def analytical_at(theta2_deg):
    pos, vel, acc = K.solve(np.radians(theta2_deg))
    d = dict(theta2=theta2_deg)
    for k in ["theta3", "theta4", "theta5"]:
        d[k] = float(np.degrees(pos[k]))
    d["S"] = float(pos["S"])
    for k in ["O", "A", "B", "C", "D", "E", "G2", "G3", "G4", "G5", "G6"]:
        d["pos_" + k] = cnum(pos[k])
    for k in ["omega2", "omega3", "omega4", "omega5"]:
        d[k] = float(vel[k])
    for k in ["VA", "VB", "VC", "VD", "VBA", "VCA", "VCB", "VDC", "VG2", "VG3", "VG4", "VG5", "VG6"]:
        d[k] = cnum(vel[k])
    for k in ["alpha2", "alpha3", "alpha4", "alpha5"]:
        d[k] = float(acc[k])
    for k in ["aA", "aB", "aC", "aD", "aBA", "aCA", "aCB", "aDC", "aG2", "aG3", "aG4", "aG5", "aG6"]:
        d[k] = cnum(acc[k])
    d["consts"] = {k: float(v) for k, v in acc["consts"].items()}
    # course template intermediate values (Part 2)
    l1, l2, l3, l4 = P.l1, P.l2, P.l3, P.l4
    th2 = np.radians(theta2_deg)
    Re = l1 * np.cos(P.theta1) - l2 * np.cos(th2)
    Im = l1 * np.sin(P.theta1) - l2 * np.sin(th2)
    Cc = (Re ** 2 + Im ** 2 + l4 ** 2 - l3 ** 2) / (2 * l4)
    disc = Re ** 2 + Im ** 2 - Cc ** 2
    d["template"] = dict(Re=float(Re), Im=float(Im), C=float(Cc), disc=float(disc), sqrt=float(np.sqrt(disc)),
                         t_minus=float((-Im - np.sqrt(disc)) / (Cc - Re)), t_plus=float((-Im + np.sqrt(disc)) / (Cc - Re)),
                         th4_minus=float(np.degrees(2 * np.arctan((-Im - np.sqrt(disc)) / (Cc - Re)))),
                         th4_plus=float(np.degrees(2 * np.arctan((-Im + np.sqrt(disc)) / (Cc - Re)))))
    return d


def analytical_table16():
    rows = []
    for i in range(16):
        t = (P.theta2_deg + 22.5 * i) % 360
        pos, vel, acc = K.solve(np.radians(t))
        rows.append(dict(i=i + 1, theta2=t, theta3=float(np.degrees(pos["theta3"])), theta4=float(np.degrees(pos["theta4"])),
                         theta5=float(np.degrees(pos["theta5"])), S=float(pos["S"]),
                         w3=float(vel["omega3"]), w4=float(vel["omega4"]), w5=float(vel["omega5"]), VD=float(vel["VD"].real),
                         VB=float(abs(vel["VB"])), VC=float(abs(vel["VC"])),
                         al3=float(acc["alpha3"]), al4=float(acc["alpha4"]), al5=float(acc["alpha5"]),
                         aD=float(acc["aD"].real), aB=float(abs(acc["aB"])), aC=float(abs(acc["aC"]))))
    return rows


def wm_build(theta2_deg):
    pos, _, _ = K.solve(np.radians(theta2_deg))
    tri_centroid = (pos["A"] + pos["B"] + pos["C"]) / 3
    return dict(theta2=theta2_deg, points={k: cnum(pos[k]) for k in ["O", "A", "B", "C", "D", "E", "G2", "G3", "G4", "G5"]},
                angles={k: float(np.degrees(pos[k])) for k in ["theta3", "theta4", "theta5"]},
                tri_centroid=cnum(tri_centroid), G3_minus_centroid=cnum(pos["G3"] - tri_centroid))


def main():
    os.makedirs(os.path.join(OUT, "figures"), exist_ok=True)
    os.makedirs(os.path.join(OUT, "tables"), exist_ok=True)
    R = {}
    R["preset"] = dict(name=P.name, label=P.label, note=P.note)
    R["scales"] = dict(N=L.N, N_ic=L.N_IC, N_ov=L.N_OV, KV=L.KV, KA=L.KA, KVI=L.KVI, text=L.strings())
    R["params"] = dict(OA=P.OA, beta_deg=P.beta_deg, theta1_deg=P.theta1_deg, n2_rpm=P.n2_rpm, AB=P.AB, BE=P.BE,
                       OE=P.OE, CD=P.CD, AC=P.AC, H=P.H, omega2=P.omega2,
                       masses=dict(m2=P.m2, m3=P.m3, m4=P.m4, m5=P.m5, m6=P.m6),
                       inertias=dict(I2=P.I2, I3=P.I3, I4=P.I4, I5=P.I5),
                       checks=[[c[0], float(c[1]), float(c[2]), bool(c[3])] for c in P.selection_checks()])
    s1 = S1.run(OUT)
    R["sec1"] = dict(limits=s1["limits"], positions16=s1["positions16"])
    s2 = S2.run(OUT)
    R["sec2"] = s2
    s3 = S3.run(OUT, s2)
    R["sec3"] = s3
    R["sec4"] = S4.run(OUT)
    R["analytical60"] = analytical_at(P.theta2_deg)
    R["analytical16"] = analytical_table16()
    R["sec6"] = S6.run(OUT)
    s7 = S7.run(OUT)
    R["sec7"] = dict(compare=s7["compare"], summary=s7["summary"], rows16=s7["rows16"])
    R["sec8"] = S8.run(OUT)
    R["wm"] = dict(at60=wm_build(60.0), at0=wm_build(0.0),
                   frame16=60.0 / P.n2_rpm / 16.0, frame6=60.0 / P.n2_rpm / 6.0, period=60.0 / P.n2_rpm)
    # transmission angle range of the four-bar
    th = np.radians(np.arange(0, 360, 0.5))
    pos = K.positions(th)
    g = np.abs(((pos["theta3"] - pos["theta4"]) + np.pi) % (2 * np.pi) - np.pi)
    gam = np.degrees(np.minimum(g, np.pi - g))
    R["transmission"] = dict(fourbar_min=float(gam.min()), fourbar_max=float(gam.max()),
                             slider_min=float(90 - np.degrees(np.max(np.abs(pos["theta5"])))))
    R["ranges"] = dict(theta3=[float(np.degrees(pos["theta3"]).min()), float(np.degrees(pos["theta3"]).max())],
                       theta4=[float(np.degrees(pos["theta4"]).min()), float(np.degrees(pos["theta4"]).max())],
                       theta5=[float(np.degrees(pos["theta5"]).min()), float(np.degrees(pos["theta5"]).max())],
                       S=[float(pos["S"].min()), float(pos["S"].max())])
    # sensitivity of the dynamic results at 60 deg to the G3 interpretation (bisector vs centroid)
    import dynamics as Dy
    sens = {}
    for mode in ("bisector", "centroid"):
        pm = dataclasses.replace(P, G3_mode=mode)
        ps, vs, ac = K.solve(np.radians(60.0), pm)
        Fm = Dy.solve_forces(ps, vs, ac, pm)
        sens[mode] = dict(aG3=abs(ac["aG3"]), T12=Fm["T12"], F12=abs(Fm["F12x"] + 1j * Fm["F12y"]),
                          F32=abs(Fm["F32x"] + 1j * Fm["F32y"]), F43=abs(Fm["F43x"] + 1j * Fm["F43y"]),
                          F14=abs(Fm["F14x"] + 1j * Fm["F14y"]), F53=abs(Fm["F53x"] + 1j * Fm["F53y"]))
    R["g3_sensitivity"] = {k: float(100 * (sens["centroid"][k] - sens["bisector"][k]) / sens["bisector"][k])
                           for k in sens["bisector"]}
    R["g3_sensitivity"]["centroid_values"] = {k: float(v) for k, v in sens["centroid"].items()}
    with open(os.path.join(OUT, "results.json"), "w") as f:
        json.dump(R, f, indent=1, default=lambda o: float(o) if isinstance(o, (np.floating,)) else str(o))
    print("wrote", os.path.join(OUT, "results.json"))


if __name__ == "__main__":
    main()
