"""Section 7 - computational model over one crank revolution (WM-equivalent).

Runs the independent multibody simulation (simulation.py), extracts the same
outputs Working Model would give (paths, velocities, accelerations, pin forces,
slider side load, motor torque) and compares them with the analytical solution.
"""
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from params import P
import kinematics as K
import dynamics as Dy
from simulation import SixBarModel
from drawing import INK, GREY, BLUE, RED, GREEN, ORANGE, PURPLE, TEAL

C5 = [BLUE, ORANGE, GREEN, RED, PURPLE, TEAL]


def run_model(gravity=0.0, n_out=721):
    model = SixBarModel(P, gravity=gravity, theta2_0=0.0)
    p0 = K.positions(0.0)
    rnd = lambda z, s=0.01: np.round(z / s) * s
    guess = np.array([rnd(p0["G2"].real), rnd(p0["G2"].imag), 0.0,
                      rnd(p0["G3"].real), rnd(p0["G3"].imag), np.radians(5 * round(np.degrees(p0["theta3"]) / 5)),
                      rnd(p0["G4"].real), rnd(p0["G4"].imag), np.radians(5 * round(np.degrees(p0["theta4"]) / 5)),
                      rnd(p0["G5"].real), rnd(p0["G5"].imag), np.radians(5 * round(np.degrees(p0["theta5"]) / 5)),
                      rnd(p0["D"].real), -0.05, 0.0])
    res = model.run(n_rev=1.0, n_out=n_out, q_guess=guess)
    return model, res


def extract(model, res):
    out = {k: [] for k in ["theta2", "theta3", "theta4", "theta5", "S", "w3", "w4", "w5", "VD", "al3", "al4", "al5",
                           "aD", "T12", "F16y", "F12", "F32", "F43", "F14", "F53", "F65", "residual"]}
    pts = {k: [] for k in ["A", "B", "C", "D", "G2", "G3", "G4", "G5"]}
    vel = {k: [] for k in ["A", "B", "C", "G3", "G5"]}
    acc = {k: [] for k in ["A", "B", "C", "G3", "G5", "G2", "G4"]}
    for r in res:
        q, qd, qdd, F = r["q"], r["qd"], r["qdd"], r["F"]
        th2 = (model.th20 + model.w2 * r["t"])
        out["theta2"].append(np.degrees(th2))
        out["theta3"].append(np.degrees(q[5])); out["theta4"].append(np.degrees(q[8])); out["theta5"].append(np.degrees(q[11]))
        out["S"].append(q[12]); out["w3"].append(qd[5]); out["w4"].append(qd[8]); out["w5"].append(qd[11])
        out["VD"].append(qd[12]); out["al3"].append(qdd[5]); out["al4"].append(qdd[8]); out["al5"].append(qdd[11])
        out["aD"].append(qdd[12]); out["T12"].append(F["T12"]); out["F16y"].append(F["F16y"])
        for k in ["F12", "F32", "F43", "F14", "F53", "F65"]:
            out[k].append(F[k])
        out["residual"].append(r["residual"])
        pts["A"].append(complex(*model.point(q, 0, model.s2A))); pts["B"].append(complex(*model.point(q, 1, model.s3B)))
        pts["C"].append(complex(*model.point(q, 1, model.s3C))); pts["D"].append(complex(*q[12:14]))
        for k, b in (("G2", 0), ("G3", 1), ("G4", 2), ("G5", 3)):
            pts[k].append(complex(*q[3 * b:3 * b + 2]))
        vel["A"].append(complex(*model.point_vel(q, qd, 0, model.s2A)))
        vel["B"].append(complex(*model.point_vel(q, qd, 1, model.s3B)))
        vel["C"].append(complex(*model.point_vel(q, qd, 1, model.s3C)))
        vel["G3"].append(complex(*qd[3:5])); vel["G5"].append(complex(*qd[9:11]))
        acc["A"].append(complex(*model.point_acc(q, qd, qdd, 0, model.s2A)))
        acc["B"].append(complex(*model.point_acc(q, qd, qdd, 1, model.s3B)))
        acc["C"].append(complex(*model.point_acc(q, qd, qdd, 1, model.s3C)))
        for k, b in (("G2", 0), ("G3", 1), ("G4", 2), ("G5", 3)):
            acc[k].append(complex(*qdd[3 * b:3 * b + 2]))
    out = {k: np.array(v) for k, v in out.items()}
    pts = {k: np.array(v) for k, v in pts.items()}
    vel = {k: np.array(v) for k, v in vel.items()}
    acc = {k: np.array(v) for k, v in acc.items()}
    return out, pts, vel, acc


def analytic_sweep(theta2_deg, mu=0.0):
    rows = []
    for t in theta2_deg:
        pos, vel, acc = K.solve(np.radians(t))
        F = Dy.solve_forces(pos, vel, acc, mu=mu)
        rows.append((pos, vel, acc, F))
    return rows


def _fmt_axes(ax, ylabel):
    ax.set_xlim(0, 360); ax.set_xticks(range(0, 361, 60)); ax.grid(alpha=0.3)
    ax.axvline(60, color=GREY, lw=0.8, ls="--")
    ax.set_ylabel(ylabel, fontsize=8); ax.tick_params(labelsize=7)


def figures(outdir, out, pts, vel, acc, out_g, an_mu03):
    th = out["theta2"]
    marks = np.array([(60 + 22.5 * k) % 360 for k in range(16)])
    an16 = analytic_sweep(marks)
    # --- paths (traces) ---
    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    for k, col in zip(["A", "B", "C", "D"], C5):
        ax.plot(pts[k].real * 1000, pts[k].imag * 1000, color=col, lw=1.4, label="joint " + k)
    for k, col in zip(["G2", "G3", "G4", "G5"], [BLUE, PURPLE, TEAL, "#b03a2e"]):
        ax.plot(pts[k].real * 1000, pts[k].imag * 1000, color=col, lw=0.8, ls="--", label="mass centre " + k)
    p60 = K.positions(np.radians(60))
    for a, b in [("O", "A"), ("A", "B"), ("A", "C"), ("B", "C"), ("E", "B"), ("C", "D")]:
        ax.plot([p60[a].real * 1000, p60[b].real * 1000], [p60[a].imag * 1000, p60[b].imag * 1000], color=INK, lw=1.2)
    ax.set_aspect("equal"); ax.grid(alpha=0.3); ax.set_xlabel("x (mm)", fontsize=8); ax.set_ylabel("y (mm)", fontsize=8)
    ax.legend(fontsize=6.3, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.17), frameon=False)
    ax.tick_params(labelsize=7)
    fig.tight_layout(); fig.savefig(f"{outdir}/figures/fig10_sim_paths.svg"); fig.savefig(f"{outdir}/figures/fig10_sim_paths.png", dpi=250)
    plt.close(fig)
    # --- angular kinematics ---
    fig, axs = plt.subplots(3, 1, figsize=(7.0, 6.4), sharex=True)
    # link 2 (input): theta2 wraps 0-360, omega2 constant, alpha2 = 0
    axs[0].plot(th, th % 360, color=GREY, lw=1.0, label="link 2 (input)")
    axs[1].plot(th, np.full_like(th, P.omega2), color=GREY, lw=1.0, label="link 2")
    axs[2].plot(th, np.zeros_like(th), color=GREY, lw=1.0, label="link 2")
    for key, lab, col, n in [("3", "link 3", BLUE, 3), ("4", "link 4", ORANGE, 4), ("5", "link 5", GREEN, 5)]:
        axs[0].plot(th, out["theta" + key], color=col, label=lab)
        axs[1].plot(th, out["w" + key], color=col, label=lab)
        axs[2].plot(th, out["al" + key], color=col, label=lab)
        axs[0].plot(marks, [np.degrees(r[0]["theta" + key]) for r in an16], "o", ms=3, mfc="white", mec=col)
        axs[1].plot(marks, [r[1]["omega" + key] for r in an16], "o", ms=3, mfc="white", mec=col)
        axs[2].plot(marks, [r[2]["alpha" + key] for r in an16], "o", ms=3, mfc="white", mec=col)
    _fmt_axes(axs[0], "angle θ (deg)"); _fmt_axes(axs[1], "ω (rad/s)"); _fmt_axes(axs[2], "α (rad/s²)")
    axs[2].set_xlabel("crank angle θ$_2$ (deg)", fontsize=8)
    axs[0].legend(fontsize=7, ncol=4, loc="upper left")
    axs[0].set_ylim(-40, 380)
    axs[0].set_title("lines: computational model;  circles: analytical solution at the 16 positions", fontsize=8)
    fig.tight_layout(); fig.savefig(f"{outdir}/figures/fig11_sim_angular.svg"); fig.savefig(f"{outdir}/figures/fig11_sim_angular.png", dpi=250)
    plt.close(fig)
    # --- slider and joint linear kinematics ---
    fig, axs = plt.subplots(3, 1, figsize=(7.0, 6.4), sharex=True)
    axs[0].plot(th, out["S"] * 1000, color=RED); axs[0].plot(marks, [r[0]["S"] * 1000 for r in an16], "o", ms=3, mfc="white", mec=RED)
    _fmt_axes(axs[0], "slider S (mm)")
    for k, col in zip(["A", "B", "C"], C5):
        axs[1].plot(th, np.abs(vel[k]), color=col, label="|V$_%s$|" % k)
        axs[2].plot(th, np.abs(acc[k]), color=col, label="|a$_%s$|" % k)
    axs[1].plot(th, out["VD"], color=RED, label="V$_D$ (signed)")
    axs[2].plot(th, out["aD"], color=RED, label="a$_D$ (signed)")
    axs[1].plot(marks, [r[1]["VD"].real for r in an16], "o", ms=3, mfc="white", mec=RED)
    axs[2].plot(marks, [r[2]["aD"].real for r in an16], "o", ms=3, mfc="white", mec=RED)
    _fmt_axes(axs[1], "velocity (m/s)"); _fmt_axes(axs[2], "acceleration (m/s²)")
    axs[1].legend(fontsize=7, ncol=4); axs[2].legend(fontsize=7, ncol=4)
    axs[2].set_xlabel("crank angle θ$_2$ (deg)", fontsize=8)
    fig.tight_layout(); fig.savefig(f"{outdir}/figures/fig12_sim_linear.svg"); fig.savefig(f"{outdir}/figures/fig12_sim_linear.png", dpi=250)
    plt.close(fig)
    # --- pin forces ---
    fig, ax = plt.subplots(figsize=(7.0, 3.4))
    for k, lab, col in [("F12", "pin O  |F$_{12}$|", BLUE), ("F32", "pin A  |F$_{32}$|", ORANGE), ("F43", "pin B  |F$_{43}$|", GREEN),
                        ("F14", "pin E  |F$_{14}$|", PURPLE), ("F53", "pin C  |F$_{53}$|", TEAL), ("F65", "pin D  |F$_{65}$|", RED)]:
        ax.plot(th, np.abs(out[k]), color=col, label=lab)
    for k, col in [("F12", BLUE), ("F32", ORANGE), ("F43", GREEN), ("F14", PURPLE), ("F53", TEAL), ("F65", RED)]:
        ax.plot(marks, [abs(r[3][k + "x"] + 1j * r[3][k + "y"]) for r in an16], "o", ms=3, mfc="white", mec=col)
    _fmt_axes(ax, "pin force magnitude (N)"); ax.set_xlabel("crank angle θ$_2$ (deg)", fontsize=8)
    ax.legend(fontsize=6.5, ncol=3)
    fig.tight_layout(); fig.savefig(f"{outdir}/figures/fig13_sim_pin_forces.svg"); fig.savefig(f"{outdir}/figures/fig13_sim_pin_forces.png", dpi=250)
    plt.close(fig)
    # --- side load and driving torque ---
    fig, axs = plt.subplots(2, 1, figsize=(7.0, 4.8), sharex=True)
    axs[0].plot(th, out["F16y"], color=RED, label="F$_{16y}$ (μ = 0, no gravity)")
    axs[0].plot(th, out_g["F16y"], color=RED, lw=0.8, ls=":", label="with gravity (WM default)")
    axs[0].plot(th, [r[3]["F16y"] for r in an_mu03], color=ORANGE, lw=0.9, ls="--", label="μ = 0.3 (matrix method)")
    axs[0].plot(marks, [r[3]["F16y"] for r in an16], "o", ms=3, mfc="white", mec=RED)
    _fmt_axes(axs[0], "slider side load F$_{16y}$ (N)"); axs[0].legend(fontsize=6.5)
    axs[1].plot(th, out["T12"], color=BLUE, label="T$_{12}$ (μ = 0, no gravity)")
    axs[1].plot(th, out_g["T12"], color=BLUE, lw=0.8, ls=":", label="with gravity")
    axs[1].plot(th, [r[3]["T12"] for r in an_mu03], color=ORANGE, lw=0.9, ls="--", label="μ = 0.3")
    axs[1].plot(marks, [r[3]["T12"] for r in an16], "o", ms=3, mfc="white", mec=BLUE)
    _fmt_axes(axs[1], "driving torque T$_{12}$ (N·m)"); axs[1].legend(fontsize=6.5)
    axs[1].set_xlabel("crank angle θ$_2$ (deg)", fontsize=8)
    fig.tight_layout(); fig.savefig(f"{outdir}/figures/fig14_sim_torque_sideload.svg"); fig.savefig(f"{outdir}/figures/fig14_sim_torque_sideload.png", dpi=250)
    plt.close(fig)


def run(outdir):
    model, res = run_model(gravity=0.0)
    out, pts, vel, acc = extract(model, res)
    model_g, res_g = run_model(gravity=9.81)
    out_g, _, _, _ = extract(model_g, res_g)
    th = out["theta2"]
    an = analytic_sweep(th)
    an_mu03 = analytic_sweep(th, mu=0.3)
    # comparison simulation vs analytical over the whole revolution
    def maxerr(sim, ana):
        sim, ana = np.asarray(sim), np.asarray(ana)
        return float(np.max(np.abs(sim - ana))), float(np.max(np.abs(ana)))
    cmp = {
        "theta3 (deg)": maxerr(out["theta3"], [np.degrees(r[0]["theta3"]) for r in an]),
        "theta4 (deg)": maxerr(out["theta4"], [np.degrees(r[0]["theta4"]) for r in an]),
        "theta5 (deg)": maxerr(out["theta5"], [np.degrees(r[0]["theta5"]) for r in an]),
        "S (m)": maxerr(out["S"], [r[0]["S"] for r in an]),
        "omega3 (rad/s)": maxerr(out["w3"], [r[1]["omega3"] for r in an]),
        "omega4 (rad/s)": maxerr(out["w4"], [r[1]["omega4"] for r in an]),
        "omega5 (rad/s)": maxerr(out["w5"], [r[1]["omega5"] for r in an]),
        "V_D (m/s)": maxerr(out["VD"], [r[1]["VD"].real for r in an]),
        "alpha3 (rad/s2)": maxerr(out["al3"], [r[2]["alpha3"] for r in an]),
        "alpha4 (rad/s2)": maxerr(out["al4"], [r[2]["alpha4"] for r in an]),
        "alpha5 (rad/s2)": maxerr(out["al5"], [r[2]["alpha5"] for r in an]),
        "a_D (m/s2)": maxerr(out["aD"], [r[2]["aD"].real for r in an]),
        "|F12| (N)": maxerr(np.abs(out["F12"]), [abs(r[3]["F12x"] + 1j * r[3]["F12y"]) for r in an]),
        "|F32| (N)": maxerr(np.abs(out["F32"]), [abs(r[3]["F32x"] + 1j * r[3]["F32y"]) for r in an]),
        "|F43| (N)": maxerr(np.abs(out["F43"]), [abs(r[3]["F43x"] + 1j * r[3]["F43y"]) for r in an]),
        "|F14| (N)": maxerr(np.abs(out["F14"]), [abs(r[3]["F14x"] + 1j * r[3]["F14y"]) for r in an]),
        "|F53| (N)": maxerr(np.abs(out["F53"]), [abs(r[3]["F53x"] + 1j * r[3]["F53y"]) for r in an]),
        "|F65| (N)": maxerr(np.abs(out["F65"]), [abs(r[3]["F65x"] + 1j * r[3]["F65y"]) for r in an]),
        "F16y (N)": maxerr(out["F16y"], [r[3]["F16y"] for r in an]),
        "T12 (N m)": maxerr(out["T12"], [r[3]["T12"] for r in an]),
    }
    figures(outdir, out, pts, vel, acc, out_g, an_mu03)
    # 16-position results table from the simulation
    rows = []
    for k in range(16):
        t = (60 + 22.5 * k) % 360
        i = int(np.argmin(np.abs(th - t)))
        assert abs(th[i] - t) < 1e-6
        rows.append(dict(pos=k + 1, theta2=t, theta3=out["theta3"][i], theta4=out["theta4"][i], theta5=out["theta5"][i],
                         S_mm=out["S"][i] * 1000, w3=out["w3"][i], w4=out["w4"][i], w5=out["w5"][i], VD=out["VD"][i],
                         al3=out["al3"][i], al4=out["al4"][i], al5=out["al5"][i], aD=out["aD"][i], T12=out["T12"][i],
                         F16y=out["F16y"][i], F12=abs(out["F12"][i]), F32=abs(out["F32"][i]), F43=abs(out["F43"][i]),
                         F14=abs(out["F14"][i]), F53=abs(out["F53"][i]), F65=abs(out["F65"][i])))
    with open(f"{outdir}/tables/table_sim_16_positions.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    i60 = int(np.argmin(np.abs(th - 60)))
    summary = dict(
        T12_max=float(out["T12"].max()), T12_min=float(out["T12"].min()), T12_mean=float(np.mean(out["T12"][:-1])),
        T12_rms=float(np.sqrt(np.mean(out["T12"][:-1] ** 2))),
        T12_max_at=float(th[np.argmax(out["T12"])]), T12_min_at=float(th[np.argmin(out["T12"])]),
        F16y_max=float(out["F16y"].max()), F16y_min=float(out["F16y"].min()),
        F16y_absmax=float(np.max(np.abs(out["F16y"]))), F16y_absmax_at=float(th[np.argmax(np.abs(out["F16y"]))]),
        pin_max={k: float(np.max(np.abs(out[k]))) for k in ["F12", "F32", "F43", "F14", "F53", "F65"]},
        pin_max_at={k: float(th[np.argmax(np.abs(out[k]))]) for k in ["F12", "F32", "F43", "F14", "F53", "F65"]},
        residual_max=float(out["residual"].max()),
        gravity_T12_maxdiff=float(np.max(np.abs(out_g["T12"] - out["T12"]))),
        gravity_F16y_maxdiff=float(np.max(np.abs(out_g["F16y"] - out["F16y"]))),
        mu03_T12_max=float(max(r[3]["T12"] for r in an_mu03)),
        mu03_T12_mean=float(np.mean([r[3]["T12"] for r in an_mu03][:-1])),
        at60=dict(T12=float(out["T12"][i60]), F16y=float(out["F16y"][i60]), VD=float(out["VD"][i60]),
                  aD=float(out["aD"][i60]), w3=float(out["w3"][i60]), w4=float(out["w4"][i60]), w5=float(out["w5"][i60]),
                  al3=float(out["al3"][i60]), al4=float(out["al4"][i60]), al5=float(out["al5"][i60]),
                  F12=float(abs(out["F12"][i60])), F32=float(abs(out["F32"][i60])), F43=float(abs(out["F43"][i60])),
                  F14=float(abs(out["F14"][i60])), F53=float(abs(out["F53"][i60])), F65=float(abs(out["F65"][i60])),
                  VA=float(abs(vel["A"][i60])), VB=float(abs(vel["B"][i60])), VC=float(abs(vel["C"][i60])),
                  aA=float(abs(acc["A"][i60])), aB=float(abs(acc["B"][i60])), aC=float(abs(acc["C"][i60])),
                  aG2=float(abs(acc["G2"][i60])), aG3=float(abs(acc["G3"][i60])), aG4=float(abs(acc["G4"][i60])),
                  aG5=float(abs(acc["G5"][i60])), VG3=float(abs(vel["G3"][i60])), VG5=float(abs(vel["G5"][i60])),
                  theta3=float(out["theta3"][i60]), theta4=float(out["theta4"][i60]), theta5=float(out["theta5"][i60]),
                  S=float(out["S"][i60]),
                  T12_gravity=float(out_g["T12"][i60]), F16y_gravity=float(out_g["F16y"][i60])),
    )
    return dict(compare=cmp, summary=summary, rows16=rows)


if __name__ == "__main__":
    import json
    r = run("../output")
    for k, v in r["compare"].items():
        print(f"{k:16s} max|sim-analytic| = {v[0]:.2e}  (max |value| = {v[1]:.3g})")
    print(json.dumps(r["summary"], indent=1))
