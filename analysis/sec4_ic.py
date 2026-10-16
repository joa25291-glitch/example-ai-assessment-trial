"""Section 4 - instantaneous centres (Kennedy) and velocities of A, B, C, D."""
import numpy as np
import matplotlib.pyplot as plt
from params import P
import kinematics as K
import ic as IC
import drawing as Dw
import sec1_position as S1
import layout as L
from drawing import arrow, label, pin, ground, INK, GREY, BLUE, RED, GREEN, ORANGE, PURPLE, TEAL

KVI = float(L.KVI)             # velocity arrows on the IC diagram: 1 m/s = KVI mm (example 1: 4 mm)
KIC = 1000.0 / L.N_IC          # paper mm per metre (1:N_IC)
_ix0, _ix1, _iy0, _iy1 = L.S["bbox"]["ic"]
OX, OY = float(round(13.0 - _ix0 * KIC)), float(round(17.5 - _iy0 * KIC))     # example 1: (40, 22)


def to_paper(z):
    return complex(OX, OY) + z * KIC


def fig_ic_diagram(path, pos, I, Vic):
    fig, ax = Dw.paper(178, 142)
    W, Hh = 178, 142
    Pp = {k: to_paper(pos[k]) for k in ["O", "A", "B", "C", "D", "E"]}
    # slider guide
    yD = Pp["D"].imag
    ax.plot([to_paper(S1.S_LO - 2.08 * P.OA).real, to_paper(S1.S_HI + 0.76 * P.OA).real], [yD - 2.6, yD - 2.6],
            color=INK, lw=0.8)
    # links
    tri = Dw.Polygon([Dw.P(Pp["A"]), Dw.P(Pp["B"]), Dw.P(Pp["C"])], closed=True, fc="#e3ecf7", ec=INK, lw=1.1, zorder=3)
    ax.add_patch(tri)
    for a, b in [("O", "A"), ("E", "B"), ("C", "D")]:
        ax.plot([Pp[a].real, Pp[b].real], [Pp[a].imag, Pp[b].imag], color=INK, lw=1.6, zorder=4)
    d = Pp["D"]
    ax.add_patch(Dw.Polygon([[d.real - 4.5, d.imag - 2.4], [d.real + 4.5, d.imag - 2.4], [d.real + 4.5, d.imag + 2.4],
                             [d.real - 4.5, d.imag + 2.4]], closed=True, fc="#f3d9b1", ec=INK, lw=0.8, zorder=3))
    ground(ax, Pp["O"], w=6); ground(ax, Pp["E"], w=6)
    # Kennedy lines
    Ip = {k: (to_paper(v) if not isinstance(v, tuple) else None) for k, v in I.items()}
    def kline(p, q, col=GREY, lw=0.55, ext=(0.0, 0.0)):
        dvec = q - p
        a_, b_ = p - ext[0] * dvec, q + ext[1] * dvec
        ax.plot([a_.real, b_.real], [a_.imag, b_.imag], color=col, lw=lw, ls=(0, (4, 2.5)), zorder=1)
    kline(Ip["12"], Ip["13"], col="#7f8c8d")                    # O-A-I13
    kline(Ip["14"], Ip["13"], col="#7f8c8d")                    # E-B-I13
    kline(Ip["24"], Ip["14"], col="#7f8c8d")                    # I24-O-E
    kline(Ip["24"], Ip["34"], col="#7f8c8d")                    # I24-A-B
    kline(Ip["36"], Ip["56"], col="#7f8c8d")                    # C-D line through I36
    kline(Ip["36"], Ip["13"], col="#7f8c8d")                    # vertical through I13
    kline(Ip["23"], Ip["36"], col="#7f8c8d")                    # A-I36 (gives I26)
    kline(Ip["34"], Ip["36"], col="#7f8c8d", ext=(0.0, 0.0))    # B-I36 (gives I46)
    kline(Ip["26"], Ip["12"], col="#7f8c8d")                    # vertical through O
    kline(Ip["46"], Ip["14"], col="#7f8c8d")                    # vertical through E
    kline(Ip["23"], Ip["25"], col="#7f8c8d", ext=(0.0, 0.3))    # A-C line through I25
    kline(Ip["34"], Ip["45"], col="#7f8c8d", ext=(0.0, 0.4))    # B-C line through I45
    # lines towards the off-sheet I15
    to15 = Ip["15"]
    for start in (Ip["13"], Ip["56"], Ip["12"], Ip["14"]):
        dvec = (to15 - start) / abs(to15 - start)
        tmax = (Hh - 3 - start.imag) / dvec.imag
        end = start + tmax * dvec
        ax.plot([start.real, end.real], [start.imag, end.imag], color=ORANGE, lw=0.55, ls=(0, (4, 2.5)), zorder=1)
    ax.annotate("", xy=(Ip["56"].real, Hh - 1.5), xytext=(Ip["56"].real, Hh - 12),
                arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=0.9))
    label(ax, complex(Ip["56"].real - 2, Hh - 9), "to I$_{15}$ (off sheet)\nx = %.0f mm, y = %.0f mm" %
          (I["15"].real * 1000, I["15"].imag * 1000), ha="right", size=6.0, color=ORANGE)
    # I16 at infinity
    label(ax, complex(8, Hh - 7), "I$_{16}$ at ∞ (⊥ slider path, i.e. vertical)", ha="left", size=6.3, color=TEAL)
    # IC markers
    for k, z in Ip.items():
        if z is None or k == "15":
            continue
        known = k in ("12", "23", "34", "14", "35", "56")
        ax.plot(z.real, z.imag, "o", ms=4.2 if not known else 3.4, mfc=("white" if known else RED), mec=RED,
                mew=0.9, zorder=8)
    offs = {"13": (4.2, 1.8), "24": (0.0, -3.6), "25": (0.8, -3.6), "26": (-5.0, 1.4),
            "36": (4.6, -1.4), "45": (4.8, -0.6), "46": (-4.8, 1.6)}
    for k, (dx, dy) in offs.items():
        label(ax, Ip[k], "I$_{%s}$" % k, dx=dx, dy=dy, color=RED, size=6.6, bbox=True)
    for k in ["O", "A", "B", "C", "D", "E"]:
        pin(ax, Pp[k], r=0.9, zorder=9)
    # joints are the known (permanent) ICs
    for k, ic_name, (dx, dy) in [("O", "12", (-8.0, 1.6)), ("A", "23", (-1.0, 4.6)), ("B", "34", (7.0, 1.2)),
                                 ("C", "35", (-3.0, -4.2)), ("D", "56", (-1.0, 5.2)), ("E", "14", (7.4, 0.6))]:
        label(ax, Pp[k], "%s = I$_{%s}$" % (k, ic_name), dx=dx, dy=dy, weight="bold", size=6.6, bbox=True)
    # velocity vectors to scale (perpendicular to the radius from the IC)
    for key, vkey, col in [("A", "VA", BLUE), ("B", "VB", BLUE), ("C", "VC", BLUE), ("D", "VD", BLUE)]:
        v = Vic[vkey]
        arrow(ax, Pp[key], Pp[key] + KVI * v, color=col, lw=1.5, ms=7, zorder=10)
        if key == "A":      # long sloping arrow: label just beyond its tip, clear of the shaft
            label(ax, Pp[key] + KVI * v + 3.2j, "V$_%s$ = %.2f m/s" % (key, abs(v)), color=col, size=6.2, bbox=True)
            continue
        side = {"B": 1, "C": 1, "D": -1}[key]
        Dw.label_on(ax, Pp[key], Pp[key] + KVI * v, 0.62, "V$_%s$ = %.2f m/s" % (key, abs(v)), side=side,
                    dist=3.4, color=col, size=6.2, bbox=True)
    # radii used
    for f, t in [("12", "A"), ("13", "B"), ("13", "C"), ("13", "A")]:
        z0 = Ip[f]; z1 = Pp[t]
        ax.plot([z0.real, z1.real], [z0.imag, z1.imag], color=GREEN, lw=0.8, zorder=2)
    ax.plot([Ip["12"].real, Ip["26"].real], [Ip["12"].imag, Ip["26"].imag], color=GREEN, lw=1.6, zorder=2)
    label(ax, (Ip["12"] + Ip["26"]) / 2, "r$_{12-26}$", dx=-6.5, dy=0, color=GREEN, size=6.0)
    Dw.scale_bar(ax, complex(8, 122), 40, "Space %s (40 mm = %g m)" % (L.strings()["ic"], 40 * L.N_IC / 1000.0))
    ax.plot([8, 8 + 4 * KVI], [113, 113], color=BLUE, lw=1.4)
    ax.text(8 + 4 * KVI + 2, 113, "velocity: 1 m/s = %g mm" % KVI, va="center", fontsize=6.0, color=BLUE)
    ax.text(8, 107.5, "green = IC radii used;  grey/orange dashed = Kennedy lines", fontsize=5.8, va="center")
    Dw.save(fig, path)


def fig_ic_overview(path, pos, I):
    """All 15 ICs on one sheet at a small scale (example 1: 1:50; I15 is ~5 m above the slider)."""
    sc = 1000.0 / L.N_OV            # paper mm per metre (example 1: 1 m = 20 mm)
    x0, y0 = 14.0, 10.0             # paper position of O
    W, Hh = 84.0, 126.0
    T = lambda z: complex(x0, y0) + z * sc
    fig, ax = Dw.paper(W, Hh)
    Pp = {k: T(pos[k]) for k in ["O", "A", "B", "C", "D", "E"]}
    ax.add_patch(Dw.Polygon([Dw.P(Pp["A"]), Dw.P(Pp["B"]), Dw.P(Pp["C"])], closed=True, fc="#e3ecf7", ec=INK, lw=0.6))
    for a, b in [("O", "A"), ("E", "B"), ("C", "D")]:
        ax.plot([Pp[a].real, Pp[b].real], [Pp[a].imag, Pp[b].imag], color=INK, lw=0.9)
    ax.plot([T(S1.S_LO - 2.08 * P.OA).real, T(S1.S_HI + 0.7 * P.OA).real], [Pp["D"].imag - 0.8, Pp["D"].imag - 0.8],
            color=INK, lw=0.6)
    Ip = {k: (T(v) if not isinstance(v, tuple) else None) for k, v in I.items()}
    # Kennedy lines that locate I15 (I13-I35 and the vertical through D) and check lines
    for a, b in [("13", "15"), ("56", "15"), ("12", "15"), ("14", "15")]:
        ax.plot([Ip[a].real, Ip[b].real], [Ip[a].imag, Ip[b].imag], color=ORANGE, lw=0.5, ls=(0, (3, 2)))
    ax.plot([Ip["35"].real, Ip["13"].real], [Ip["35"].imag, Ip["13"].imag], color=ORANGE, lw=0.5, ls=(0, (3, 2)))
    for k, z in Ip.items():
        if z is None:
            continue
        known = k in ("12", "23", "34", "14", "35", "56")
        ax.plot(z.real, z.imag, "o", ms=2.6, mfc=("white" if known else RED), mec=RED, mew=0.7, zorder=8)
    lab = {"13": (3.0, 0.5), "15": (3.2, 0.0), "24": (0, -2.4), "12": (-3.0, 0.8), "14": (3.0, -0.8)}
    for k, (dx, dy) in lab.items():
        label(ax, Ip[k], "I$_{%s}$" % k, dx=dx, dy=dy, color=RED, size=6.0)
    label(ax, Ip["56"], "I$_{56}$", dx=0.5, dy=-2.6, color=RED, size=6.0)
    x16 = Pp["D"].real + 2.0
    ax.annotate("", xy=(x16, Hh - 3), xytext=(x16, Hh - 16),
                arrowprops=dict(arrowstyle="-|>", color=TEAL, lw=0.8))
    label(ax, complex(x16 + 1.5, Hh - 10), "I$_{16}$ at ∞\n(vertical)", ha="left", size=5.8, color=TEAL)
    label(ax, Ip["15"], "(%.0f, %.0f) mm" % (I["15"].real * 1000, I["15"].imag * 1000), dx=-1, dy=-3.4, size=5.4,
          color=RED, ha="right")
    ax.text(3, 30, "Mechanism\n(ICs I$_{23}$, I$_{25}$, I$_{26}$,\nI$_{34}$, I$_{35}$, I$_{36}$, I$_{45}$, I$_{46}$\nare close together\nhere - see Figure 8)",
            fontsize=5.4, va="bottom")
    Dw.scale_bar(ax, complex(44, 18), 20, "%s (20 mm = %g m)" % (L.strings()["overview"], 20 * L.N_OV / 1000.0))
    Dw.save(fig, path)


KNOWN = ["12", "23", "34", "14", "35", "56", "16"]


def fig_kennedy_steps(path):
    """3x3 panels: the initial Kennedy circle, then each of the 8 steps."""
    nodes = {k: np.exp(1j * (np.pi / 2 - 2 * np.pi * (k - 1) / 6)) for k in range(1, 7)}
    fig, axs = plt.subplots(3, 3, figsize=(7.0, 7.3))
    for ax in axs.flat:
        ax.set_aspect("equal"); ax.axis("off"); ax.set_xlim(-1.45, 1.45); ax.set_ylim(-1.55, 1.45)
    def base(ax, found):
        th = np.linspace(0, 2 * np.pi, 200)
        ax.plot(np.cos(th), np.sin(th), color="#d0d4da", lw=0.8)
        for name in KNOWN + found:
            a, b = int(name[0]), int(name[1])
            ax.plot([nodes[a].real, nodes[b].real], [nodes[a].imag, nodes[b].imag], color=INK, lw=1.0, zorder=2)
        for k, z in nodes.items():
            ax.plot(z.real, z.imag, "o", ms=11, mfc="white", mec=INK, zorder=5)
            ax.text(z.real, z.imag, str(k), ha="center", va="center", fontsize=8, weight="bold", zorder=6)
    unknown = [s[0] for s in IC.STEPS]
    ax = axs.flat[0]
    base(ax, [])
    for name in unknown:
        a, b = int(name[0]), int(name[1])
        ax.plot([nodes[a].real, nodes[b].real], [nodes[a].imag, nodes[b].imag], color=RED, lw=0.8, ls=(0, (3, 2)))
    ax.set_title("Start: 7 known (solid)\n8 unknown (dashed)", fontsize=7.5)
    found = []
    for k, (name, l1, l2, desc) in enumerate(IC.STEPS):
        ax = axs.flat[k + 1]
        base(ax, found)
        a, b = int(name[0]), int(name[1])
        # the two triangles that close the unknown side
        for pair, col in ((l1, BLUE), (l2, GREEN)):
            for side in pair:
                x, y = int(side[0]), int(side[1])
                ax.plot([nodes[x].real, nodes[y].real], [nodes[x].imag, nodes[y].imag], color=col, lw=2.2, zorder=3)
        ax.plot([nodes[a].real, nodes[b].real], [nodes[a].imag, nodes[b].imag], color=RED, lw=2.0, ls=(0, (3, 1.5)),
                zorder=4)
        ax.set_title("Step %d: I$_{%s}$" % (k + 1, name), fontsize=8, color=RED)
        ax.text(0, -1.42, "I$_{%s}$, I$_{%s}$ (blue) ∩ I$_{%s}$, I$_{%s}$ (green)" % (l1[0], l1[1], l2[0], l2[1]),
                ha="center", fontsize=6.3)
        found.append(name)
    fig.tight_layout()
    fig.savefig(path + ".svg"); fig.savefig(path + ".png", dpi=250)
    plt.close(fig)


def run(outdir):
    th2 = np.radians(P.theta2_deg)
    pos, vel, acc = K.solve(th2)
    I, checks = IC.find_ics(pos)
    V = IC.ic_velocities(pos, I, P.omega2)
    fig_ic_diagram(f"{outdir}/figures/fig7_ic_diagram", pos, I, V)
    fig_kennedy_steps(f"{outdir}/figures/fig8_kennedy_circle")
    fig_ic_overview(f"{outdir}/figures/fig7b_ic_overview", pos, I)
    # distances read from the 1:N_IC drawing to the nearest 0.5 mm (1:10 -> nearest 5 mm full size)
    rd = lambda z: float(Dw.round_to(abs(z) * KIC, 0.5) / KIC)
    # I12-A = OA and I14-B = BE are link lengths (known exactly); the other radii are measured
    r = dict(r12A=float(P.OA), r13A=rd(pos["A"] - I["13"]), r13B=rd(pos["B"] - I["13"]),
             r13C=rd(pos["C"] - I["13"]), r14B=float(P.BE), r12_26=rd(I["26"] - I["12"]),
             r15C=float(abs(pos["C"] - I["15"])), r15D=float(abs(pos["D"] - I["15"])))
    VA = P.omega2 * r["r12A"]
    w3 = VA / r["r13A"]
    meas = dict(VA=VA, omega3=w3, VB=w3 * r["r13B"], VC=w3 * r["r13C"], omega4=w3 * r["r13B"] / r["r14B"],
                VD=P.omega2 * r["r12_26"], omega5=(w3 * r["r13C"]) / r["r15C"])
    ics = {k: (None if isinstance(v, tuple) else [float(v.real), float(v.imag)]) for k, v in I.items()}
    return dict(ics=ics, checks={k: float(v) for k, v in checks.items()}, radii=r, measured=meas,
                exact={k: (float(abs(v)) if isinstance(v, complex) else float(v)) for k, v in V.items()})


if __name__ == "__main__":
    import json
    out = run("../output")
    print(json.dumps(out["radii"], indent=1)); print(json.dumps(out["measured"], indent=1))
