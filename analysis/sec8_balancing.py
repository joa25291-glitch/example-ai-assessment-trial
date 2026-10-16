"""Section 8 - shaking force, shaking moment and an analytical balancing strategy.

Shaking force  Fs = F21 + F41 + F61 = -sum(m_i a_Gi)            (frame reaction)
Shaking moment Ms = T21 + R_E x F41 + R_D x F61  (about O)
                  = -sum( r_Gi x m_i a_Gi + I_Gi alpha_i )       (check)

Balancing strategy (counterweights on the two ground-pivoted links 2 and 4):
  1. Replace link 5 by two point masses m5/2 at C and m5/2 at D (statically
     equivalent - exact for the shaking FORCE).
  2. Berkof-Lowen: choose the counterweight "mass-moment" vectors P2 (on the
     crank, about O) and P4 (on the rocker, about E) so that the total mass
     centre of links 2, 3 (+ m5/2 at C) and 4 is STATIONARY.  Derived from the
     loop closure  l3 e^{j th3} = l1 e^{j th1} + l4 e^{j th4} - l2 e^{j th2}.
  3. What remains is the horizontal reciprocating force of m_rec = m6 + m5/2.
     A rotating counterweight can only cancel the forward-rotating half of its
     first harmonic, so the optimum is found by least squares.
  4. Global optimum: complex linear least squares for P2, P4 minimising the
     RMS shaking force over one revolution.
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from params import P
import kinematics as K
import dynamics as Dy
from drawing import INK, GREY, BLUE, RED, GREEN, ORANGE, PURPLE, TEAL

TH = np.radians(np.arange(0.0, 360.0, 0.5))
cross = lambda r, f: r.real * f.imag - r.imag * f.real
dot = lambda u, v: u.real * v.real + u.imag * v.imag


def sweep():
    data = []
    for t in TH:
        pos, vel, acc = K.solve(t)
        data.append((pos, vel, acc))
    return data


def base_quantities(data):
    ms = [P.m2, P.m3, P.m4, P.m5, P.m6]
    Is = [P.I2, P.I3, P.I4, P.I5, 0.0]
    Fs, Ms, T = [], [], []
    for pos, vel, acc in data:
        al = [acc["alpha2"], acc["alpha3"], acc["alpha4"], acc["alpha5"], 0.0]
        om = [vel["omega2"], vel["omega3"], vel["omega4"], vel["omega5"], 0.0]
        G = [pos["G2"], pos["G3"], pos["G4"], pos["G5"], pos["G6"]]
        aG = [acc["aG2"], acc["aG3"], acc["aG4"], acc["aG5"], acc["aG6"]]
        vG = [vel["VG2"], vel["VG3"], vel["VG4"], vel["VG5"], vel["VG6"]]
        Fs.append(-sum(m * a for m, a in zip(ms, aG)))
        Ms.append(-sum(cross(g, m * a) + I * alp for g, m, a, I, alp in zip(G, ms, aG, Is, al)))
        T.append(sum(m * dot(a, v) + I * alp * w for m, a, v, I, alp, w in zip(ms, aG, vG, Is, al, om)) / vel["omega2"])
    return np.array(Fs), np.array(Ms), np.array(T)


def counterweight_effect(data, P2, P4):
    """Extra shaking force, moment and torque produced by point counterweights whose
    mass-moment vectors (body frame) are P2 = m_c2 u2 (about O, x along OA) and
    P4 = m_c4 u4 (about E, x along EB).  Masses enter only through m*u for the force;
    for the moment/torque we use point masses at radius RC2, RC4."""
    dF, dM, dT = [], [], []
    m2c, u2 = P2
    m4c, u4 = P4
    for pos, vel, acc in data:
        th2, th4 = pos["theta2"], pos["theta4"]
        w2, w4, a2, a4 = vel["omega2"], vel["omega4"], acc["alpha2"], acc["alpha4"]
        r2 = u2 * np.exp(1j * th2); r4 = pos["E"] + u4 * np.exp(1j * th4)
        v2 = 1j * w2 * u2 * np.exp(1j * th2); v4 = 1j * w4 * u4 * np.exp(1j * th4)
        A2 = (1j * a2 - w2 ** 2) * u2 * np.exp(1j * th2); A4 = (1j * a4 - w4 ** 2) * u4 * np.exp(1j * th4)
        dF.append(-(m2c * A2 + m4c * A4))
        dM.append(-(cross(r2, m2c * A2) + cross(r4, m4c * A4)))
        dT.append((m2c * dot(A2, v2) + m4c * dot(A4, v4)) / w2)
    return np.array(dF), np.array(dM), np.array(dT)


def berkof_lowen():
    """Required total body-frame mass moments of links 2 and 4 for a stationary
    centre of mass of links 2, 3 (+ m5/2 lumped at C) and 4."""
    l1, l2, l3, l4 = P.l1, P.l2, P.l3, P.l4
    m3p = P.m3 + P.m5 / 2.0
    g3 = (P.m3 * P.g3_local + (P.m5 / 2) * P.AC * np.exp(-1j * P.beta)) / m3p
    M2_req = -m3p * l2 * (1 - g3 / l3)          # total mass moment of link 2 about O (body frame)
    M4_req = -m3p * g3 * l4 / l3                 # total mass moment of link 4 about E (body frame)
    M2_now = P.m2 * l2 / 2
    M4_now = P.m4 * l4 / 2
    return dict(m3p=m3p, g3=g3, P2=M2_req - M2_now, P4=M4_req - M4_now, M2_req=M2_req, M4_req=M4_req)


def lstsq_counterweights(data, Fs0, include_P4=True):
    cols = []
    g2 = np.array([-(1j * acc["alpha2"] - vel["omega2"] ** 2) * np.exp(1j * pos["theta2"]) for pos, vel, acc in data])
    g4 = np.array([-(1j * acc["alpha4"] - vel["omega4"] ** 2) * np.exp(1j * pos["theta4"]) for pos, vel, acc in data])
    cols = [g2, g4] if include_P4 else [g2]
    A = np.stack(cols, axis=1)
    sol, *_ = np.linalg.lstsq(A, -Fs0, rcond=None)
    return sol


def stats(Fs, Ms, T):
    return dict(Fs_rms=float(np.sqrt(np.mean(np.abs(Fs) ** 2))), Fs_max=float(np.max(np.abs(Fs))),
                Fsx_max=float(np.max(np.abs(Fs.real))), Fsy_max=float(np.max(np.abs(Fs.imag))),
                Ms_rms=float(np.sqrt(np.mean(Ms ** 2))), Ms_max=float(np.max(np.abs(Ms))),
                T_max=float(np.max(T)), T_min=float(np.min(T)), T_rms=float(np.sqrt(np.mean(T ** 2))))


def run(outdir, RC2=None, RC4=None):
    # counterweight radius: about 2/3 of the crank length, rounded to 10 mm (example 1: 120 mm)
    if RC2 is None:
        RC2 = round(2.0 * P.OA / 3.0, 2)
    if RC4 is None:
        RC4 = RC2
    data = sweep()
    Fs0, Ms0, T0 = base_quantities(data)
    # check against the matrix method at 60 deg
    pos, vel, acc = K.solve(np.radians(60))
    F = Dy.solve_forces(pos, vel, acc)
    S = Dy.shaking(pos, F)
    i60 = int(np.argmin(np.abs(np.degrees(TH) - 60)))
    assert abs(S["Fs"] - Fs0[i60]) < 1e-6 and abs(S["Ms"] - Ms0[i60]) < 1e-6
    results = {"unbalanced": stats(Fs0, Ms0, T0)}
    designs = {}
    # (1) Berkof-Lowen four-bar balance (with m5/2 lumped at C)
    bl = berkof_lowen()
    def as_cw(Pvec, R):
        m = abs(Pvec) / R
        u = R * Pvec / abs(Pvec)
        return m, u
    m2c, u2 = as_cw(bl["P2"], RC2); m4c, u4 = as_cw(bl["P4"], RC4)
    dF, dM, dT = counterweight_effect(data, (m2c, u2), (m4c, u4))
    F1, M1, T1 = Fs0 + dF, Ms0 + dM, T0 + dT
    results["berkof_lowen"] = stats(F1, M1, T1)
    designs["berkof_lowen"] = dict(m2c=m2c, u2=u2, m4c=m4c, u4=u4)
    resid = -(P.m6 + P.m5 / 2) * np.array([acc["aD"] for _, _, acc in data])
    results["bl_residual_check_max"] = float(np.max(np.abs(F1 - resid)))
    # (2) BL + extra crank counterweight by least squares on the residual
    extra = lstsq_counterweights(data, F1, include_P4=False)[0]
    m2e, u2e = as_cw(bl["P2"] + extra, RC2)
    dF, dM, dT = counterweight_effect(data, (m2e, u2e), (m4c, u4))
    F2, M2, T2 = Fs0 + dF, Ms0 + dM, T0 + dT
    results["bl_plus_crank"] = stats(F2, M2, T2)
    designs["bl_plus_crank"] = dict(m2c=m2e, u2=u2e, m4c=m4c, u4=u4)
    # (3) global least-squares optimum on links 2 and 4
    P2o, P4o = lstsq_counterweights(data, Fs0, include_P4=True)
    m2o, u2o = as_cw(P2o, RC2); m4o, u4o = as_cw(P4o, RC4)
    dF, dM, dT = counterweight_effect(data, (m2o, u2o), (m4o, u4o))
    F3, M3, T3 = Fs0 + dF, Ms0 + dM, T0 + dT
    results["optimal_lsq"] = stats(F3, M3, T3)
    designs["optimal_lsq"] = dict(m2c=m2o, u2=u2o, m4c=m4o, u4=u4o)
    # (4) crank counterweight only (simplest practical option)
    P2c = lstsq_counterweights(data, Fs0, include_P4=False)[0]
    m2s, u2s = as_cw(P2c, RC2)
    dF, dM, dT = counterweight_effect(data, (m2s, u2s), (0.0, 0j))
    F4, M4, T4 = Fs0 + dF, Ms0 + dM, T0 + dT
    results["crank_only"] = stats(F4, M4, T4)
    designs["crank_only"] = dict(m2c=m2s, u2=u2s, m4c=0.0, u4=0j)
    # ------------- figures -------------
    deg = np.degrees(TH)
    fig, axs = plt.subplots(2, 1, figsize=(7.0, 6.0), gridspec_kw=dict(height_ratios=[1.25, 1.0]))
    ax = axs[0]
    for Fc, lab, col, ls in [(Fs0, "unbalanced", RED, "-"), (F1, "Berkof-Lowen on links 2 & 4", BLUE, "--"),
                             (F3, "least-squares optimum", GREEN, "-")]:
        ax.plot(Fc.real, Fc.imag, color=col, ls=ls, lw=1.3, label=lab)
    for Fc, col in [(Fs0, RED), (F3, GREEN)]:
        ax.plot(Fc[i60].real, Fc[i60].imag, "o", color=col, ms=4)
    ax.annotate("θ$_2$ = 60°", (Fs0[i60].real, Fs0[i60].imag), textcoords="offset points", xytext=(5, 5), fontsize=7)
    for k in range(0, 720, 90):     # crank-angle ticks every 45 deg on the unbalanced curve
        ax.annotate("%d°" % (k // 2), (Fs0[k].real, Fs0[k].imag), textcoords="offset points", xytext=(4, -9),
                    fontsize=6, color=RED)
        ax.plot(Fs0[k].real, Fs0[k].imag, ".", color=RED, ms=4)
    ax.set_aspect("equal"); ax.grid(alpha=0.3); ax.axhline(0, color=GREY, lw=0.6); ax.axvline(0, color=GREY, lw=0.6)
    ax.set_xlabel("F$_{s,x}$ (N)", fontsize=8); ax.set_ylabel("F$_{s,y}$ (N)", fontsize=8); ax.tick_params(labelsize=7)
    ax.set_title("Shaking-force polar diagram (tip of F$_s$ over one revolution)", fontsize=8.5)
    ax.legend(fontsize=6.8, loc="best")
    ax = axs[1]
    ax.plot(deg, np.abs(Fs0), color=RED, label="|F$_s$| unbalanced")
    ax.plot(deg, np.abs(F1), color=BLUE, ls="--", label="|F$_s$| Berkof-Lowen")
    ax.plot(deg, np.abs(F3), color=GREEN, label="|F$_s$| optimum")
    ax.set_xlim(0, 360); ax.set_xticks(range(0, 361, 45)); ax.grid(alpha=0.3); ax.axvline(60, color=GREY, ls="--", lw=0.8)
    ax.set_xlabel("crank angle θ$_2$ (deg)", fontsize=8); ax.set_ylabel("shaking force magnitude (N)", fontsize=8)
    ax.tick_params(labelsize=7); ax.legend(fontsize=6.8, ncol=3)
    fig.tight_layout(); fig.savefig(f"{outdir}/figures/fig15_shaking_force.svg"); fig.savefig(f"{outdir}/figures/fig15_shaking_force.png", dpi=250)
    plt.close(fig)
    fig, axs = plt.subplots(2, 1, figsize=(7.0, 4.6), sharex=True)
    axs[0].plot(deg, Ms0, color=RED, label="M$_s$ unbalanced")
    axs[0].plot(deg, M1, color=BLUE, ls="--", label="Berkof-Lowen")
    axs[0].plot(deg, M3, color=GREEN, label="least-squares optimum")
    axs[0].set_ylabel("shaking moment about O (N·m)", fontsize=8)
    axs[1].plot(deg, T0, color=RED, label="T$_{12}$ unbalanced")
    axs[1].plot(deg, T1, color=BLUE, ls="--", label="Berkof-Lowen")
    axs[1].plot(deg, T3, color=GREEN, label="least-squares optimum")
    axs[1].set_ylabel("driving torque (N·m)", fontsize=8); axs[1].set_xlabel("crank angle θ$_2$ (deg)", fontsize=8)
    for ax in axs:
        ax.set_xlim(0, 360); ax.set_xticks(range(0, 361, 60)); ax.grid(alpha=0.3); ax.axvline(60, color=GREY, ls="--", lw=0.8)
        ax.legend(fontsize=6.5, ncol=3); ax.tick_params(labelsize=7)
    fig.tight_layout(); fig.savefig(f"{outdir}/figures/fig16_shaking_moment_torque.svg"); fig.savefig(f"{outdir}/figures/fig16_shaking_moment_torque.png", dpi=250)
    plt.close(fig)
    ser = lambda d: {k: (float(abs(v)) if k.startswith("m") else [float(np.real(v)), float(np.imag(v))]) for k, v in d.items()}
    out = dict(at60=dict(Fsx=float(S["Fs"].real), Fsy=float(S["Fs"].imag), Fs=float(abs(S["Fs"])),
                         Fs_angle=float(np.degrees(np.angle(S["Fs"])) % 360), Ms=float(S["Ms"]), T21=float(S["T21"]),
                         F21=[float(S["F21"].real), float(S["F21"].imag)], F41=[float(S["F41"].real), float(S["F41"].imag)],
                         F61=[float(S["F61"].real), float(S["F61"].imag)],
                         RExF41=float(cross(pos["E"], S["F41"])), RDxF61=float(cross(pos["D"], S["F61"]))),
               stats=results, designs={k: ser(v) for k, v in designs.items()},
               berkof=dict(m3p=bl["m3p"], g3=[float(bl["g3"].real), float(bl["g3"].imag)],
                           P2=[float(bl["P2"].real), float(bl["P2"].imag)], P4=[float(bl["P4"].real), float(bl["P4"].imag)],
                           M2_req=[float(bl["M2_req"].real), float(bl["M2_req"].imag)],
                           M4_req=[float(bl["M4_req"].real), float(bl["M4_req"].imag)]),
               optimal=dict(P2=[float(P2o.real), float(P2o.imag)], P4=[float(P4o.real), float(P4o.imag)]),
               RC2=RC2, RC4=RC4, m_rec=P.m6 + P.m5 / 2)
    return out


if __name__ == "__main__":
    r = run("../output")
    print(json.dumps(r, indent=1))
