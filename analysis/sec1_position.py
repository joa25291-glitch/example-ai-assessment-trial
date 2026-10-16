"""Section 1 - graphical position analysis: space diagram, sample construction,
16 subsequent crank positions, joint paths and limits of motion."""
import json
import numpy as np
from scipy.optimize import brentq
from params import P
import kinematics as K
import drawing as Dw
from drawing import arrow, label, guide, pin, ground, angle_arc, scale_bar, INK, GREY, BLUE, RED, GREEN, ORANGE, PURPLE, TEAL

SCALE = 10.0           # space scale 1:10  ->  1 m = 100 mm on paper
OX, OY = 32.0, 34.0    # paper position of O (mm)


def to_paper(z):
    return complex(OX, OY) + z * 1000.0 / SCALE


def draw_mechanism(ax, pos, lw=1.6, color=INK, shade=True, labels=True, gmarks=True, alpha=1.0, zorder=5):
    Pp = {k: to_paper(pos[k]) for k in ["O", "A", "B", "C", "D", "E", "G2", "G3", "G4", "G5", "G6"]}
    seg = lambda a, b, **kw: ax.plot([Pp[a].real, Pp[b].real], [Pp[a].imag, Pp[b].imag], color=color,
                                    lw=kw.get("lw", lw), alpha=alpha, zorder=zorder, solid_capstyle="round")
    if shade:
        tri = Dw.Polygon([Dw.P(Pp["A"]), Dw.P(Pp["B"]), Dw.P(Pp["C"])], closed=True, fc="#d6e4f5",
                         ec=color, lw=lw * 0.8, zorder=zorder - 1, alpha=alpha)
        ax.add_patch(tri)
    else:
        seg("A", "C"); seg("B", "C")
    seg("O", "A"); seg("A", "B"); seg("E", "B"); seg("C", "D")
    # slider block
    d = Pp["D"]
    blk = Dw.Polygon([[d.real - 5, d.imag - 2.6], [d.real + 5, d.imag - 2.6], [d.real + 5, d.imag + 2.6],
                      [d.real - 5, d.imag + 2.6]], closed=True, fc="#f3d9b1", ec=color, lw=0.9, zorder=zorder - 1,
                     alpha=alpha)
    ax.add_patch(blk)
    for k in ["O", "A", "B", "C", "D", "E"]:
        pin(ax, Pp[k], r=1.0, zorder=zorder + 2)
    if gmarks:
        for k in ["G2", "G3", "G4", "G5"]:
            ax.plot(Pp[k].real, Pp[k].imag, marker="o", ms=4.2, mfc="white", mec=RED, mew=0.9, zorder=zorder + 3)
            ax.plot(Pp[k].real, Pp[k].imag, marker="x", ms=3.0, mec=RED, mew=0.8, zorder=zorder + 4)
    if labels:
        label(ax, Pp["O"], "O", dx=-3.2, dy=1.5, weight="bold")
        label(ax, Pp["A"], "A", dx=-3.0, dy=1.8, weight="bold")
        label(ax, Pp["B"], "B", dx=1.0, dy=3.0, weight="bold")
        label(ax, Pp["C"], "C", dx=-0.8, dy=-3.6, weight="bold")
        label(ax, Pp["D"], "D", dx=0.0, dy=5.0, weight="bold")
        label(ax, Pp["E"], "E", dx=3.2, dy=1.6, weight="bold")
        if gmarks:
            label(ax, Pp["G2"], "G$_2$", dx=-4.6, dy=1.6, color=RED, size=6.5)
            label(ax, Pp["G3"], "G$_3$", dx=0.0, dy=3.4, color=RED, size=6.5)
            label(ax, Pp["G4"], "G$_4$", dx=-4.6, dy=0.0, color=RED, size=6.5)
            label(ax, Pp["G5"], "G$_5$", dx=1.0, dy=-3.8, color=RED, size=6.5)
            label(ax, Pp["D"], "G$_6$", dx=8.0, dy=-1.5, color=RED, size=6.5)
    return Pp


def slider_guide(ax, x_from, x_to, path_line=False):
    y = to_paper(-P.H * 1j).imag
    if path_line:
        ax.plot([x_from - 50, x_to], [y, y], color=RED, lw=0.6, ls=(0, (8, 2, 1.5, 2)), zorder=2)
    ax.plot([x_from, x_to], [y - 2.6, y - 2.6], color=INK, lw=0.9, zorder=2)
    for x in np.arange(x_from + 1, x_to, 3.0):
        ax.plot([x, x - 1.4], [y - 2.6, y - 4.2], color=INK, lw=0.45, zorder=2)


def limits_of_motion():
    """Exact crank angles where omega4, omega3, omega5 and V_D vanish."""
    th = np.radians(np.linspace(0, 360, 3601))
    def roots(fun):
        vals = np.array([fun(t) for t in th])
        out = []
        for k in range(len(th) - 1):
            if np.sign(vals[k]) != np.sign(vals[k + 1]):
                out.append(brentq(fun, th[k], th[k + 1], xtol=1e-14))
        return out
    v = lambda key: (lambda t: np.real(K.solve(t)[1][key]))
    res = {}
    for key, name in [("omega4", "rocker4"), ("omega3", "link3"), ("omega5", "link5"), ("VD", "slider6")]:
        rts = roots(v(key))
        entries = []
        for r in rts:
            pos, _, _ = K.solve(r)
            entries.append(dict(theta2=float(np.degrees(r) % 360), theta3=float(np.degrees(pos["theta3"])),
                                theta4=float(np.degrees(pos["theta4"])), theta5=float(np.degrees(pos["theta5"])),
                                S=float(pos["S"])))
        res[name] = entries
    return res


def fig_space_diagram(path):
    th2 = np.radians(P.theta2_deg)
    pos, _, _ = K.solve(th2)
    fig, ax = Dw.paper(178, 100)
    slider_guide(ax, to_paper(0.38).real, to_paper(1.40).real)
    Pp = draw_mechanism(ax, pos)
    ground(ax, Pp["O"]); ground(ax, Pp["E"])
    # theta1 reference line and OE dashed
    ax.plot([Pp["O"].real, Pp["E"].real], [Pp["O"].imag, Pp["E"].imag], color=GREY, lw=0.7, ls=(0, (5, 3)))
    label(ax, Pp["O"] + 0.72 * (Pp["E"] - Pp["O"]), "$l_1$ = OE", dy=2.8, dx=0, size=6.5, color=GREY)
    ax.plot([Pp["O"].real, Pp["O"].real + 60], [Pp["O"].imag, Pp["O"].imag], color=GREY, lw=0.5)
    ax.plot([Pp["A"].real, Pp["A"].real + 30], [Pp["A"].imag, Pp["A"].imag], color=GREY, lw=0.5)
    ax.plot([Pp["E"].real, Pp["E"].real + 22], [Pp["E"].imag, Pp["E"].imag], color=GREY, lw=0.5)
    ax.plot([Pp["C"].real, Pp["C"].real + 26], [Pp["C"].imag, Pp["C"].imag], color=GREY, lw=0.5)
    angle_arc(ax, Pp["O"], 9, 0, P.theta2_deg, color=BLUE, text=r"$\theta_2$=60°", text_r=13.5)
    angle_arc(ax, Pp["O"], 40, 0, P.theta1_deg, color=GREY, text=r"$\theta_1$", text_r=44)
    t3 = np.degrees(pos["theta3"]); t4 = np.degrees(pos["theta4"]); t5 = np.degrees(pos["theta5"])
    angle_arc(ax, Pp["A"], 22, 0, t3, color=BLUE, text=r"$\theta_3$", text_r=26)
    angle_arc(ax, Pp["A"], 14, t3 - P.beta_deg, t3, color=PURPLE, text=r"$\beta$", text_r=17.5)
    angle_arc(ax, Pp["E"], 8, 0, t4, color=BLUE, text=r"$\theta_4$", text_r=12)
    angle_arc(ax, Pp["C"], 15, t5, 0, color=BLUE, text=r"$\theta_5$", text_r=19)
    # H and S dimensions
    xd = Pp["O"].real + 52
    yD = to_paper(-P.H * 1j).imag
    ax.annotate("", xy=(xd, Pp["O"].imag), xytext=(xd, yD), arrowprops=dict(arrowstyle="<->", lw=0.6, color=INK))
    label(ax, complex(xd + 3.5, (Pp["O"].imag + yD) / 2), "H", size=6.5)
    ys = yD - 12
    ax.plot([Pp["O"].real, Pp["O"].real], [Pp["O"].imag - 5, ys - 1.5], color=GREY, lw=0.5)
    ax.plot([Pp["D"].real, Pp["D"].real], [yD - 3, ys - 1.5], color=GREY, lw=0.5)
    ax.annotate("", xy=(Pp["O"].real, ys), xytext=(Pp["D"].real, ys), arrowprops=dict(arrowstyle="<->", lw=0.6))
    label(ax, complex((Pp["O"].real + Pp["D"].real) / 2, ys - 2.6), "S = %.1f mm" % (pos["S"] * 1000), size=6.5)
    # link numbers
    mid = lambda a, b: (Pp[a] + Pp[b]) / 2
    label(ax, Pp["O"] + 0.8 * (Pp["A"] - Pp["O"]), "2", dx=3.4, dy=-0.5, color=TEAL, weight="bold")
    label(ax, 0.2 * Pp["A"] + 0.4 * Pp["B"] + 0.4 * Pp["C"], "3", dx=0, dy=0, color=TEAL, weight="bold")
    label(ax, Pp["E"] + 0.7 * (Pp["B"] - Pp["E"]), "4", dx=3.4, dy=0, color=TEAL, weight="bold")
    label(ax, Pp["C"] + 0.62 * (Pp["D"] - Pp["C"]), "5", dx=1.8, dy=3.4, color=TEAL, weight="bold")
    label(ax, Pp["D"], "6", dx=6.5, dy=3.5, color=TEAL, weight="bold")
    label(ax, Pp["O"], "1", dx=5.5, dy=-6, color=TEAL, weight="bold")
    label(ax, Pp["E"], "1", dx=5.5, dy=-6, color=TEAL, weight="bold")
    w_a = Pp["O"] + 6.5 * np.exp(1j * np.radians(115)); w_b = Pp["O"] + 6.5 * np.exp(1j * np.radians(205))
    ax.annotate("", xy=(w_b.real, w_b.imag), xytext=(w_a.real, w_a.imag),
                arrowprops=dict(arrowstyle="-|>", connectionstyle="arc3,rad=0.42", lw=0.8, color=RED))
    label(ax, Pp["O"], r"$\omega_2$ (CCW)", dx=-14, dy=4.5, color=RED, size=6.5)
    scale_bar(ax, complex(118, 88), 50, "Space scale 1:10  (50 mm = 0.5 m)")
    Dw.save(fig, path)
    return pos


def fig_construction(path):
    """Sample working for one position (steps a-f of the Week 1 procedure)."""
    th2 = np.radians(P.theta2_deg)
    pos, _, _ = K.solve(th2)
    fig, ax = Dw.paper(178, 80, y0=22)
    Pp = {k: to_paper(pos[k]) for k in ["O", "A", "B", "C", "D", "E"]}
    slider_guide(ax, to_paper(0.38).real, to_paper(1.40).real, path_line=True)
    ground(ax, Pp["O"]); ground(ax, Pp["E"])
    # (a) OA at theta2
    ax.plot([Pp["O"].real, Pp["A"].real], [Pp["O"].imag, Pp["A"].imag], color=INK, lw=1.5)
    label(ax, (Pp["O"] + Pp["A"]) / 2, "(a)", dx=-5, dy=0, color=BLUE, weight="bold")
    # (b) OE at theta1
    ax.plot([Pp["O"].real, Pp["E"].real], [Pp["O"].imag, Pp["E"].imag], color=GREY, lw=0.8, ls=(0, (5, 3)))
    label(ax, (Pp["O"] + Pp["E"]) / 2, "(b)", dx=0, dy=-3.5, color=BLUE, weight="bold")
    # (c) arcs: centre A radius AB, centre E radius BE  -> B
    rAB = P.AB * 100; rBE = P.BE * 100
    aB = np.degrees(np.angle(Pp["B"] - Pp["A"])); eB = np.degrees(np.angle(Pp["B"] - Pp["E"]))
    ax.add_patch(Dw.Arc(Dw.P(Pp["A"]), 2 * rAB, 2 * rAB, theta1=aB - 12, theta2=aB + 12, color=GREEN, lw=0.8))
    ax.add_patch(Dw.Arc(Dw.P(Pp["E"]), 2 * rBE, 2 * rBE, theta1=eB - 25, theta2=eB + 25, color=ORANGE, lw=0.8))
    ax.plot([Pp["A"].real, Pp["B"].real], [Pp["A"].imag, Pp["B"].imag], color=INK, lw=1.2)
    ax.plot([Pp["E"].real, Pp["B"].real], [Pp["E"].imag, Pp["B"].imag], color=INK, lw=1.2)
    label(ax, Pp["B"], "(c)", dx=-9, dy=3.0, color=BLUE, weight="bold")
    label(ax, Pp["A"] + rAB * np.exp(1j * np.radians(aB + 12)), "arc R = AB\ncentre A", dx=-12, dy=-1, color=GREEN, size=6)
    label(ax, Pp["E"] + rBE * np.exp(1j * np.radians(eB - 25)), "arc R = BE\ncentre E", dx=13, dy=0, color=ORANGE, size=6)
    # (d) AC at beta below AB, then join C-B
    ax.plot([Pp["A"].real, Pp["C"].real], [Pp["A"].imag, Pp["C"].imag], color=INK, lw=1.2)
    ax.plot([Pp["B"].real, Pp["C"].real], [Pp["B"].imag, Pp["C"].imag], color=INK, lw=1.2)
    angle_arc(ax, Pp["A"], 16, aB - P.beta_deg, aB, color=PURPLE, text=r"$\beta$", text_r=20)
    label(ax, Pp["C"], "(d)", dx=-2, dy=-5, color=BLUE, weight="bold")
    # (e) horizontal line H below O (already drawn as slider guide)
    label(ax, complex(to_paper(0.15).real, to_paper(-P.H * 1j).imag), "(e)  path of D: horizontal line H below O",
          dx=20, dy=-3.0, color=BLUE, size=6.5)
    # (f) arc centre C radius CD -> D
    rCD = P.CD * 100
    cD = np.degrees(np.angle(Pp["D"] - Pp["C"]))
    ax.add_patch(Dw.Arc(Dw.P(Pp["C"]), 2 * rCD, 2 * rCD, theta1=cD - 7, theta2=cD + 7, color=RED, lw=0.8))
    ax.plot([Pp["C"].real, Pp["D"].real], [Pp["C"].imag, Pp["D"].imag], color=INK, lw=1.2)
    label(ax, Pp["D"], "(f)", dx=5, dy=5.5, color=BLUE, weight="bold")
    label(ax, Pp["C"] + rCD * np.exp(1j * np.radians(cD + 7)), "arc R = CD\ncentre C", dx=11, dy=2, color=RED, size=6)
    for k in Pp:
        pin(ax, Pp[k], r=0.9)
        label(ax, Pp[k], k, dx=-2.8 if k in "OA" else 2.8, dy=2.6, weight="bold")
    scale_bar(ax, complex(118, 88), 50, "Space scale 1:10  (50 mm = 0.5 m)")
    Dw.save(fig, path)


def positions_16():
    rows = []
    for i in range(16):
        th2d = (P.theta2_deg + 22.5 * i) % 360
        pos, vel, acc = K.solve(np.radians(th2d))
        rows.append(dict(i=i + 1, theta2=th2d, theta3=float(np.degrees(pos["theta3"])),
                         theta4=float(np.degrees(pos["theta4"])), theta5=float(np.degrees(pos["theta5"])),
                         S=float(pos["S"]),
                         **{f"{k}x": float(pos[k].real) for k in "ABCD"},
                         **{f"{k}y": float(pos[k].imag) for k in "ABCD"}))
    return rows


def fig_16_positions(path, lim):
    fig, ax = Dw.paper(178, 120, y0=-6)
    slider_guide(ax, to_paper(0.55).real, to_paper(1.42).real)
    thf = np.radians(np.linspace(0, 360, 721))
    posf = K.positions(thf)
    Pf = {k: to_paper(posf[k]) for k in ["A", "B", "C", "D"]}
    O = to_paper(0j); E = to_paper(P.l1 * np.exp(1j * P.theta1))
    # faint links for every one of the 16 positions
    for i in range(16):
        th2d = (P.theta2_deg + 22.5 * i) % 360
        pos, _, _ = K.solve(np.radians(th2d))
        Pp = {k: to_paper(pos[k]) for k in ["O", "A", "B", "C", "D", "E"]}
        for a, b in [("O", "A"), ("A", "B"), ("A", "C"), ("B", "C"), ("E", "B"), ("C", "D")]:
            ax.plot([Pp[a].real, Pp[b].real], [Pp[a].imag, Pp[b].imag], color="#b9c3cf", lw=0.35, zorder=1)
    # paths
    ax.plot(Pf["A"].real, Pf["A"].imag, color=BLUE, lw=1.0, label="S$_A$")
    ax.plot(Pf["B"].real, Pf["B"].imag, color=ORANGE, lw=1.6)
    ax.plot(Pf["C"].real, Pf["C"].imag, color=GREEN, lw=1.0)
    ax.plot(Pf["D"].real, Pf["D"].imag, color=RED, lw=2.2, solid_capstyle="butt")
    for i in range(16):
        th2d = (P.theta2_deg + 22.5 * i) % 360
        pos, _, _ = K.solve(np.radians(th2d))
        for key, col, off in [("A", BLUE, 3.6), ("B", ORANGE, 0), ("C", GREEN, 0), ("D", RED, 0)]:
            z = to_paper(pos[key])
            ax.plot(z.real, z.imag, "o", ms=2.2, color=col, zorder=6)
            if key == "A":
                u = (z - O) / abs(z - O)
                label(ax, z + u * off, str(i + 1), size=5.6, color=col)
    # numbering for B, C, D kept sparse to stay legible: label positions 1,5,9,13
    for i in [0, 4, 8, 12]:
        th2d = (P.theta2_deg + 22.5 * i) % 360
        pos, _, _ = K.solve(np.radians(th2d))
        label(ax, to_paper(pos["B"]), str(i + 1), dx=0, dy=2.6, size=5.6, color=ORANGE)
        label(ax, to_paper(pos["C"]), str(i + 1), dx=-2.4, dy=-2.0, size=5.6, color=GREEN)
        label(ax, to_paper(pos["D"]), str(i + 1), dx=0, dy=3.0, size=5.6, color=RED)
    pin(ax, O); pin(ax, E); ground(ax, O); ground(ax, E)
    label(ax, O, "O", dx=-3, dy=2.5, weight="bold"); label(ax, E, "E", dx=3, dy=2.5, weight="bold")
    # limits of motion
    for ent, tag in zip(sorted(lim["rocker4"], key=lambda e: e["theta4"]), ["B'", "B''"]):
        pB = to_paper(P.l1 * np.exp(1j * P.theta1) + P.BE * np.exp(1j * np.radians(ent["theta4"])))
        ax.plot([E.real, pB.real], [E.imag, pB.imag], color=ORANGE, lw=0.9, ls=(0, (2, 1.5)))
        txt = "%s  θ$_4$ = %.1f°\n(θ$_2$ = %.1f°)" % (tag, ent["theta4"], ent["theta2"])
        if tag == "B'":
            label(ax, pB, txt, dx=13, dy=2.5, size=5.8, color=ORANGE)
        else:
            label(ax, pB, txt, dx=-14, dy=1.5, size=5.8, color=ORANGE)
    for ent in lim["slider6"]:
        pD = to_paper(ent["S"] - 1j * P.H)
        ax.plot([pD.real, pD.real], [pD.imag - 5, pD.imag + 5], color=RED, lw=0.9)
    sl = sorted(lim["slider6"], key=lambda e: e["S"])
    label(ax, to_paper(sl[0]["S"] - 1j * P.H), "D$_L$ (θ$_2$=%.1f°)" % sl[0]["theta2"], dx=-1, dy=-8.0, color=RED, size=6.0)
    label(ax, to_paper(sl[1]["S"] - 1j * P.H), "D$_R$ (θ$_2$=%.1f°)" % sl[1]["theta2"], dx=1, dy=-8.0, color=RED, size=6.0)
    label(ax, complex((to_paper(sl[0]["S"]).real + to_paper(sl[1]["S"]).real) / 2, to_paper(-P.H * 1j).imag),
          "stroke = %.1f mm" % ((sl[1]["S"] - sl[0]["S"]) * 1000), dy=-12.5, color=RED, size=6.0)
    # legend
    lx, ly = 6, 102
    for k, (txt, col) in enumerate([("S$_A$  path of A (circle, R = OA)", BLUE),
                                    ("S$_B$  path of B (arc, R = BE, oscillates)", ORANGE),
                                    ("S$_C$  path of C (coupler curve)", GREEN),
                                    ("S$_D$  path of D (straight stroke D$_L$-D$_R$)", RED)]):
        ax.plot([lx, lx + 6], [ly - 4.2 * k, ly - 4.2 * k], color=col, lw=1.5)
        ax.text(lx + 8, ly - 4.2 * k, txt, va="center", fontsize=6.4)
    ax.text(lx, ly + 6, "Positions 1-16: θ$_2$ = 60° + (i-1)·22.5°  (CCW)", fontsize=6.6, weight="bold")
    scale_bar(ax, complex(118, 106), 50, "Space scale 1:10  (50 mm = 0.5 m)")
    Dw.save(fig, path)


def fig_angles_vs_crank(path):
    import matplotlib.pyplot as plt
    th = np.linspace(0, 360, 721)
    pos = K.positions(np.radians(th))
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.6))
    ax[0].plot(th, np.degrees(pos["theta3"]), label=r"$\theta_3$ (link 3)", color=BLUE)
    ax[0].plot(th, np.degrees(pos["theta4"]), label=r"$\theta_4$ (link 4)", color=ORANGE)
    ax[0].plot(th, np.degrees(pos["theta5"]), label=r"$\theta_5$ (link 5)", color=GREEN)
    ax[0].set_xlabel(r"crank angle $\theta_2$ (deg)"); ax[0].set_ylabel("link angle (deg)")
    ax[1].plot(th, pos["S"] * 1000, color=RED)
    ax[1].set_xlabel(r"crank angle $\theta_2$ (deg)"); ax[1].set_ylabel("slider position S (mm)")
    for a in ax:
        a.set_xlim(0, 360); a.set_xticks(range(0, 361, 45)); a.grid(alpha=0.3)
        a.axvline(60, color=GREY, lw=0.8, ls="--")
    ax[0].legend(fontsize=7, loc="best")
    for k in range(16):
        t = (60 + 22.5 * k) % 360
        p_, _, _ = K.solve(np.radians(t))
        ax[1].plot(t, p_["S"] * 1000, "o", ms=3, color=RED)
    fig.tight_layout()
    fig.savefig(path + ".svg"); fig.savefig(path + ".png", dpi=250)
    plt.close(fig)


def run(outdir):
    lim = limits_of_motion()
    pos = fig_space_diagram(f"{outdir}/figures/fig1_space_diagram")
    fig_construction(f"{outdir}/figures/fig2_position_construction")
    fig_16_positions(f"{outdir}/figures/fig3_16_positions", lim)
    fig_angles_vs_crank(f"{outdir}/figures/fig4_link_angles")
    rows = positions_16()
    import csv
    with open(f"{outdir}/tables/table_16_positions.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    return dict(limits=lim, positions16=rows)


if __name__ == "__main__":
    out = run("../output")
    print(json.dumps(out["limits"], indent=1))
