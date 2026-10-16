"""Section 6 - free-body diagrams (course notation) and the 14x14 matrix at 60 deg."""
import numpy as np
import matplotlib.pyplot as plt
from params import P
import kinematics as K
import dynamics as Dy
import drawing as Dw
from drawing import INK, GREY, BLUE, RED, GREEN, ORANGE, PURPLE, TEAL


def _dir_arrow(ax, p, v, color, text, L=0.55, off=0.13, lw=1.4, ls="-"):
    """Arrow of FIXED length L in the direction of complex vector v (direction only)."""
    u = v / abs(v)
    q = p + L * u
    ax.annotate("", xy=(q.real, q.imag), xytext=(p.real, p.imag),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, mutation_scale=10, linestyle=ls))
    t = q + off * u
    ha = "left" if u.real > 0.35 else ("right" if u.real < -0.35 else "center")
    va = "bottom" if u.imag > 0.35 else ("top" if u.imag < -0.35 else "center")
    ax.text(t.real, t.imag, text, color=color, fontsize=6.6, ha=ha, va=va)


def _panel(ax, pts, G, title, shape="bar"):
    """Normalise joint positions about G so the link spans about 1.6 units."""
    rel = {k: (v - G) for k, v in pts.items()}
    s = 0.8 / max(abs(z) for z in rel.values())
    rel = {k: z * s for k, z in rel.items()}
    if shape == "tri":
        ax.add_patch(Dw.Polygon([Dw.P(rel[k]) for k in ("A", "B", "C")], closed=True, fc="#e3ecf7", ec=INK, lw=1.3))
    else:
        a, b = list(rel.values())
        ax.plot([a.real, b.real], [a.imag, b.imag], color=INK, lw=5, solid_capstyle="round", alpha=0.85)
    for k, z in rel.items():
        ax.plot(z.real, z.imag, "o", ms=6, mfc="white", mec=INK, zorder=5)
    ax.plot(0, 0, "o", ms=6, mfc=RED, mec=RED, zorder=6)
    ax.set_xlim(-1.75, 1.75); ax.set_ylim(-1.55, 1.55)
    ax.set_title(title, fontsize=8)
    return rel


def _rvec(ax, z, name, side=1):
    ax.annotate("", xy=(z.real * 0.9, z.imag * 0.9), xytext=(0, 0),
                arrowprops=dict(arrowstyle="-|>", color=PURPLE, lw=0.9, linestyle="--", mutation_scale=7))
    n = 1j * z / abs(z) * 0.12 * side
    ax.text(z.real * 0.5 + n.real, z.imag * 0.5 + n.imag, name, color=PURPLE, fontsize=6.3, ha="center", va="center")


def fig_fbd(path, pos, acc, F):
    """Free-body diagrams in the true link orientations.  Arrows show the SOLVED force
    directions (mu = 0, theta2 = 60 deg); magnitudes are written on the labels."""
    fig, axs = plt.subplots(2, 3, figsize=(7.2, 5.0))
    for ax in axs.flat:
        ax.set_aspect("equal"); ax.axis("off")
    c = lambda k: F[k + "x"] + 1j * F[k + "y"]
    N = lambda z: "%.0f N" % abs(z)
    # link 2
    ax = axs[0, 0]
    r = _panel(ax, {"O": pos["O"], "A": pos["A"]}, pos["G2"], "Link 2 (crank)")
    _rvec(ax, r["O"], "R$_{12}$"); _rvec(ax, r["A"], "R$_{32}$")
    _dir_arrow(ax, r["O"], c("F12"), BLUE, "F$_{12}$ = " + N(c("F12")))
    _dir_arrow(ax, r["A"], c("F32"), BLUE, "F$_{32}$ = " + N(c("F32")))
    _dir_arrow(ax, 0j, acc["aG2"], RED, "a$_{G2}$", L=0.4, lw=1.1)
    ax.annotate("", xy=(-0.55, 0.25), xytext=(-0.05, 0.62), arrowprops=dict(arrowstyle="-|>", color=GREEN,
                connectionstyle="arc3,rad=0.45", lw=1.2))
    ax.text(-0.95, 0.55, "T$_{12}$ = %.1f N·m" % F["T12"], color=GREEN, fontsize=6.6, ha="center")
    for k, z in r.items(): ax.text(z.real + 0.12, z.imag - 0.12, k, fontsize=7, weight="bold")
    # link 3
    ax = axs[0, 1]
    r = _panel(ax, {"A": pos["A"], "B": pos["B"], "C": pos["C"]}, pos["G3"], "Link 3 (ternary link)", shape="tri")
    _rvec(ax, r["A"], "R$_{23}$"); _rvec(ax, r["B"], "R$_{43}$"); _rvec(ax, r["C"], "R$_{53}$", side=-1)
    _dir_arrow(ax, r["A"], -c("F32"), BLUE, "F$_{23}$ = -F$_{32}$")
    _dir_arrow(ax, r["B"], c("F43"), BLUE, "F$_{43}$ = " + N(c("F43")))
    _dir_arrow(ax, r["C"], c("F53"), BLUE, "F$_{53}$ = " + N(c("F53")))
    _dir_arrow(ax, 0j, acc["aG3"], RED, "a$_{G3}$", L=0.4, lw=1.1)
    for k, z in r.items(): ax.text(z.real - 0.05, z.imag + 0.14, k, fontsize=7, weight="bold")
    # link 4
    ax = axs[0, 2]
    r = _panel(ax, {"E": pos["E"], "B": pos["B"]}, pos["G4"], "Link 4 (rocker)")
    _rvec(ax, r["B"], "R$_{34}$", side=-1); _rvec(ax, r["E"], "R$_{14}$", side=-1)
    _dir_arrow(ax, r["B"], -c("F43"), BLUE, "F$_{34}$ = -F$_{43}$")
    _dir_arrow(ax, r["E"], c("F14"), BLUE, "F$_{14}$ = " + N(c("F14")))
    _dir_arrow(ax, 0j, acc["aG4"], RED, "a$_{G4}$", L=0.4, lw=1.1)
    for k, z in r.items(): ax.text(z.real + 0.12, z.imag, k, fontsize=7, weight="bold")
    # link 5
    ax = axs[1, 0]
    r = _panel(ax, {"C": pos["C"], "D": pos["D"]}, pos["G5"], "Link 5 (connecting rod)")
    _rvec(ax, r["C"], "R$_{35}$"); _rvec(ax, r["D"], "R$_{65}$")
    _dir_arrow(ax, r["C"], -c("F53"), BLUE, "F$_{35}$ = -F$_{53}$")
    _dir_arrow(ax, r["D"], c("F65"), BLUE, "F$_{65}$ = " + N(c("F65")))
    _dir_arrow(ax, 0j, acc["aG5"], RED, "a$_{G5}$", L=0.4, lw=1.1)
    for k, z in r.items(): ax.text(z.real + 0.05, z.imag + 0.15, k, fontsize=7, weight="bold")
    # link 6
    ax = axs[1, 1]
    ax.add_patch(Dw.Polygon([[-0.45, -0.28], [0.45, -0.28], [0.45, 0.28], [-0.45, 0.28]], closed=True,
                            fc="#f3d9b1", ec=INK, lw=1.0))
    ax.plot([-1.4, 1.4], [-0.28, -0.28], color=GREY, lw=0.8)
    ax.plot(0, 0, "o", ms=6, mfc=RED, mec=RED, zorder=6)
    _dir_arrow(ax, 0j, -c("F65"), BLUE, "F$_{56}$ = -F$_{65}$")
    _dir_arrow(ax, 0j, 1j * F["F16y"], ORANGE, "F$_{16y}$ = %.0f N" % F["F16y"])
    ax.annotate("", xy=(0.9, 0.62), xytext=(0.3, 0.62), arrowprops=dict(arrowstyle="-|>", color=GREY, lw=1.0))
    ax.text(0.95, 0.62, "F$_{16x}$ = μ'F$_{16y}$\n(0 here)", color=GREY, fontsize=6.3, va="center")
    _dir_arrow(ax, 0j, acc["aG6"], RED, "a$_{G6}$ = a$_D$", L=0.4, lw=1.1)
    ax.set_xlim(-1.75, 1.75); ax.set_ylim(-1.55, 1.55)
    ax.set_title("Link 6 (slider, G$_6$ at D)", fontsize=8)
    # notes
    ax = axs[1, 2]
    lines = ["Notation (Week 9 convention)", "F$_{ij}$ : force of link i ON link j", "F$_{ji}$ = -F$_{ij}$ (Newton's 3rd law)",
             "R$_{ij}$ : vector from G$_j$ to the", "        joint where F$_{ij}$ acts", "",
             "Each link:  ΣF = m a$_G$", "               ΣM$_G$ = I$_G$ α", "",
             "Arrows: solved directions at", "θ$_2$ = 60°, μ = 0 (not to scale).", "Gravity neglected."]
    for k, t in enumerate(lines):
        ax.text(0.02, 0.98 - 0.08 * k, t, fontsize=6.8, transform=ax.transAxes, va="top",
                weight="bold" if k == 0 else "normal")
    fig.tight_layout()
    fig.savefig(path + ".svg"); fig.savefig(path + ".png", dpi=250)
    plt.close(fig)


def run(outdir):
    th2 = np.radians(P.theta2_deg)
    pos, vel, acc = K.solve(th2)
    res = {}
    for mu in (0.0, 0.3):
        F = Dy.solve_forces(pos, vel, acc, mu=mu)
        S = Dy.shaking(pos, F)
        res[str(mu)] = dict(F={k: float(F[k]) for k in Dy.UNKNOWNS + ["F16x", "mu_eff"]},
                            matrix=F["matrix"].tolist(), rhs=F["rhs"].tolist(),
                            shaking=dict(Fsx=float(S["Fs"].real), Fsy=float(S["Fs"].imag), Fs=float(abs(S["Fs"])),
                                         Fs_angle=float(np.degrees(np.angle(S["Fs"])) % 360), Ms=float(S["Ms"]),
                                         F21=[float(S["F21"].real), float(S["F21"].imag)],
                                         F41=[float(S["F41"].real), float(S["F41"].imag)],
                                         F61=[float(S["F61"].real), float(S["F61"].imag)], T21=float(S["T21"])))
    F0 = Dy.solve_forces(pos, vel, acc, mu=0.0)
    fig_fbd(f"{outdir}/figures/fig9_fbd", pos, acc, F0)
    chk = Dy.inertia_checks(pos, vel, acc)
    R = Dy.r_vectors(pos)
    res["R"] = {k: [float(v.real), float(v.imag)] for k, v in R.items()}
    res["checks"] = dict(T12_power=float(chk["T12_power"]), Fs_inertia=[float(chk["Fs_inertia"].real),
                         float(chk["Fs_inertia"].imag)], Ms_inertia=float(chk["Ms_inertia"]))
    return res


if __name__ == "__main__":
    import json
    r = run("../output")
    print(json.dumps(r["0.0"]["F"], indent=1))
