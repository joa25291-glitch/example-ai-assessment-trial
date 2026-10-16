"""Section 6 - free-body diagrams (course notation) and the 14x14 matrix at 60 deg."""
import numpy as np
import matplotlib.pyplot as plt
from params import P
import kinematics as K
import dynamics as Dy
import drawing as Dw
from drawing import INK, GREY, BLUE, RED, GREEN, ORANGE, PURPLE, TEAL


FS = 6.6                         # label font size (pt)
CH_W, CH_H = 0.08, 0.2           # character width / line height in panel units at FS (measured on the PNG)


def _text_len(s):
    """Approximate printed width of a mathtext label in characters (subscripts count 0.7)."""
    import re
    subs = re.findall(r"\$_\{?([^}$]*)\}?\$", s)
    plain = re.sub(r"\$_\{?([^}$]*)\}?\$", "", s).replace("$", "")
    return max(len(line) for line in plain.split("\n")) + 0.7 * sum(len(t) for t in subs)


class Placer:
    """Greedy label placement: every drawn item is sampled into obstacle points and each
    label takes the first candidate position (in order of preference) whose text box
    covers the fewest of them.  Keeps the FBD legible for any parameter set."""

    def __init__(self, ax):
        self.ax, self.pts = ax, []

    def seg(self, p, q, step=0.025):
        n = max(2, int(abs(q - p) / step) + 1)
        self.pts.extend(p + (q - p) * k / (n - 1) for k in range(n))

    def dot(self, z, r=0.07):
        self.pts.extend([z] + [z + r * np.exp(1j * k * np.pi / 4) for k in range(8)])

    @staticmethod
    def box(z, w, h, ha, va):
        x0 = z.real - {"center": w / 2, "left": 0.0, "right": w}[ha]
        y0 = z.imag - {"center": h / 2, "bottom": 0.0, "top": h}[va]
        return x0, x0 + w, y0, y0 + h

    def hits(self, bx, pad=0.025):
        if not self.pts:
            return 0
        P_ = np.asarray(self.pts)
        x0, x1, y0, y1 = bx
        # stay within this panel's share of the figure (penalty grows with the overshoot)
        over = max(0.0, -2.0 - x0) + max(0.0, x1 - 2.0) + max(0.0, -1.62 - y0) + max(0.0, y1 - 1.45)
        out = int(np.ceil(over / 0.02)) * 50
        return out + int(((P_.real > x0 - pad) & (P_.real < x1 + pad) & (P_.imag > y0 - pad) & (P_.imag < y1 + pad)).sum())

    def place(self, text, cands, color, size=FS, **kw):
        w, h = CH_W * _text_len(text) * size / FS, CH_H * size / FS
        best = min(range(len(cands)), key=lambda i: (self.hits(self.box(cands[i][0], w, h, *cands[i][1:])), i))
        z, ha, va = cands[best]
        self.ax.text(z.real, z.imag, text, color=color, fontsize=size, ha=ha, va=va, zorder=9, **kw)
        x0, x1, y0, y1 = self.box(z, w, h, ha, va)
        self.pts.extend(complex(x, y) for x in np.linspace(x0, x1, max(2, int((x1 - x0) / 0.025)))
                        for y in np.linspace(y0, y1, 4))


def _align(u):
    ha = "left" if u.real > 0.35 else ("right" if u.real < -0.35 else "center")
    va = "bottom" if u.imag > 0.35 else ("top" if u.imag < -0.35 else "center")
    return ha, va


def _ring(z, r, first):
    """Candidate label positions on a circle around z, starting in direction `first`."""
    out = []
    for k in (0, 1, -1, 2, -2, 3, -3, 4):
        u = first * np.exp(1j * k * np.pi / 4)
        out.append((z + r * u, *_align(u)))
    return out


def _arrow(ax, pl, p, v, color, L=0.55, lw=1.4):
    """Arrow of FIXED length L from p in the direction of v (direction only); returns the tip."""
    u = v / abs(v)
    q = p + L * u
    ax.annotate("", xy=(q.real, q.imag), xytext=(p.real, p.imag),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, mutation_scale=10))
    pl.seg(p, q)
    return q, u


def _force(ax, pl, p, v, color, text, L=0.55):
    q, u = _arrow(ax, pl, p, v, color, L)
    cands = [(q + 0.1 * u, *_align(u))]
    for k in (1, -1, 2, -2):
        w = u * np.exp(1j * k * np.pi / 6)
        cands.append((q + 0.12 * w, *_align(w)))
    return q, u, cands, text, color


def _panel(ax, pl, pts, G, title, shape="bar"):
    """Normalise joint positions about G so the link spans about 1.6 units."""
    rel = {k: (v - G) for k, v in pts.items()}
    s = 0.8 / max(abs(z) for z in rel.values())
    rel = {k: z * s for k, z in rel.items()}
    if shape == "tri":
        ax.add_patch(Dw.Polygon([Dw.P(rel[k]) for k in ("A", "B", "C")], closed=True, fc="#e3ecf7", ec=INK, lw=1.3))
        for a, b in (("A", "B"), ("B", "C"), ("C", "A")):
            pl.seg(rel[a], rel[b])
    else:
        a, b = list(rel.values())
        ax.plot([a.real, b.real], [a.imag, b.imag], color=INK, lw=5, solid_capstyle="round", alpha=0.85)
        pl.seg(a, b); pl.seg(a + 0.04j * (b - a) / abs(b - a), b + 0.04j * (b - a) / abs(b - a))
        pl.seg(a - 0.04j * (b - a) / abs(b - a), b - 0.04j * (b - a) / abs(b - a))
    for k, z in rel.items():
        ax.plot(z.real, z.imag, "o", ms=6, mfc="white", mec=INK, zorder=5)
        pl.dot(z)
    ax.plot(0, 0, "o", ms=6, mfc=RED, mec=RED, zorder=6)
    pl.dot(0j)
    ax.set_xlim(-1.75, 1.75); ax.set_ylim(-1.55, 1.55)
    ax.set_title(title, fontsize=8)
    return rel


def _rvec(ax, pl, z):
    ax.annotate("", xy=(z.real * 0.9, z.imag * 0.9), xytext=(0, 0),
                arrowprops=dict(arrowstyle="-|>", color=PURPLE, lw=0.9, linestyle="--", mutation_scale=7))
    pl.seg(0j, 0.9 * z)


def _rlabel(pl, z, name):
    n = 1j * z / abs(z)
    cands = [(z * t + sd * off * n, "center", "center") for t in (0.5, 0.38, 0.62, 0.28, 0.75)
             for sd in (1, -1) for off in (0.13, 0.2)]
    pl.place(name, cands, PURPLE, size=6.3)


def _torque(ax, pl, T, text, color=GREEN, r=0.5):
    """Curved arrow for a couple about G (CCW if T > 0), put where it is clear of the rest."""
    best = None
    for phi in np.radians([135, 315, 225, 45, 180, 0, 90, 270]):
        th = phi + np.radians(np.linspace(-32, 32, 25)) * (1 if T > 0 else -1)
        arc = r * np.exp(1j * th)
        lab = (1.0 * np.exp(1j * phi), *_align(np.exp(1j * phi)))
        w, h = CH_W * _text_len(text), CH_H
        score = sum(pl.hits((z.real - 0.03, z.real + 0.03, z.imag - 0.03, z.imag + 0.03), pad=0.0) for z in arc) \
            + pl.hits(Placer.box(lab[0], w, h, lab[1], lab[2]))
        if best is None or score < best[0]:
            best = (score, arc, lab)
    _, arc, lab = best
    ax.plot(arc.real[:-2], arc.imag[:-2], color=color, lw=1.2)
    ax.annotate("", xy=(arc[-1].real, arc[-1].imag), xytext=(arc[-3].real, arc[-3].imag),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=1.2, mutation_scale=10))
    for z in arc:
        pl.dot(z, 0.02)
    pl.place(text, [lab], color)


def fig_fbd(path, pos, acc, F):
    """Free-body diagrams in the true link orientations.  Arrows show the SOLVED force
    directions (mu = 0, theta2 = 60 deg); magnitudes are written on the labels."""
    fig, axs = plt.subplots(2, 3, figsize=(7.2, 5.0))
    for ax in axs.flat:
        ax.set_aspect("equal"); ax.axis("off")
    c = lambda k: F[k + "x"] + 1j * F[k + "y"]
    N = lambda z: "%.0f N" % abs(z)

    def finish(ax, pl, rel, forces, rnames, acc_vec, acc_name, torque=None):
        """Draw R vectors, forces and a_G, then place every label (forces first)."""
        for jn, nm in rnames:
            _rvec(ax, pl, rel[jn])
        fl = [_force(ax, pl, *f) for f in forces]
        qa, ua = _arrow(ax, pl, 0j, acc_vec, RED, L=0.4, lw=1.1)
        for q, u, cands, text, color in fl:
            pl.place(text, cands, color)
        if torque is not None:
            _torque(ax, pl, *torque)
        for k, z in rel.items():
            pl.place(k, _ring(z, 0.16, z / abs(z)), INK, size=7, weight="bold")
        for jn, nm in rnames:
            _rlabel(pl, rel[jn], nm)
        pl.place(acc_name, _ring(qa, 0.12, ua) + _ring(qa, 0.2, ua), RED)

    # link 2
    ax = axs[0, 0]; pl = Placer(ax)
    r = _panel(ax, pl, {"O": pos["O"], "A": pos["A"]}, pos["G2"], "Link 2 (crank)")
    finish(ax, pl, r, [(r["O"], c("F12"), BLUE, "F$_{12}$ = " + N(c("F12"))),
                       (r["A"], c("F32"), BLUE, "F$_{32}$ = " + N(c("F32")))],
           [("O", "R$_{12}$"), ("A", "R$_{32}$")], acc["aG2"], "a$_{G2}$",
           torque=(F["T12"], "T$_{12}$ = %.1f N·m" % F["T12"]))
    # link 3
    ax = axs[0, 1]; pl = Placer(ax)
    r = _panel(ax, pl, {"A": pos["A"], "B": pos["B"], "C": pos["C"]}, pos["G3"], "Link 3 (ternary link)", shape="tri")
    finish(ax, pl, r, [(r["A"], -c("F32"), BLUE, "F$_{23}$ = -F$_{32}$"),
                       (r["B"], c("F43"), BLUE, "F$_{43}$ = " + N(c("F43"))),
                       (r["C"], c("F53"), BLUE, "F$_{53}$ = " + N(c("F53")))],
           [("A", "R$_{23}$"), ("B", "R$_{43}$"), ("C", "R$_{53}$")], acc["aG3"], "a$_{G3}$")
    # link 4
    ax = axs[0, 2]; pl = Placer(ax)
    r = _panel(ax, pl, {"E": pos["E"], "B": pos["B"]}, pos["G4"], "Link 4 (rocker)")
    finish(ax, pl, r, [(r["B"], -c("F43"), BLUE, "F$_{34}$ = -F$_{43}$"),
                       (r["E"], c("F14"), BLUE, "F$_{14}$ = " + N(c("F14")))],
           [("B", "R$_{34}$"), ("E", "R$_{14}$")], acc["aG4"], "a$_{G4}$")
    # link 5
    ax = axs[1, 0]; pl = Placer(ax)
    r = _panel(ax, pl, {"C": pos["C"], "D": pos["D"]}, pos["G5"], "Link 5 (connecting rod)")
    finish(ax, pl, r, [(r["C"], -c("F53"), BLUE, "F$_{35}$ = -F$_{53}$"),
                       (r["D"], c("F65"), BLUE, "F$_{65}$ = " + N(c("F65")))],
           [("C", "R$_{35}$"), ("D", "R$_{65}$")], acc["aG5"], "a$_{G5}$")
    # link 6
    ax = axs[1, 1]; pl = Placer(ax)
    ax.add_patch(Dw.Polygon([[-0.45, -0.28], [0.45, -0.28], [0.45, 0.28], [-0.45, 0.28]], closed=True,
                            fc="#f3d9b1", ec=INK, lw=1.0))
    ax.plot([-1.4, 1.4], [-0.28, -0.28], color=GREY, lw=0.8)
    for a_, b_ in ((-0.45 - 0.28j, 0.45 - 0.28j), (0.45 - 0.28j, 0.45 + 0.28j), (0.45 + 0.28j, -0.45 + 0.28j),
                   (-0.45 + 0.28j, -0.45 - 0.28j), (-1.4 - 0.28j, 1.4 - 0.28j)):
        pl.seg(a_, b_)
    ax.plot(0, 0, "o", ms=6, mfc=RED, mec=RED, zorder=6); pl.dot(0j)
    sl = [_force(ax, pl, 0j, -c("F65"), BLUE, "F$_{56}$ = -F$_{65}$"),
          _force(ax, pl, 0j, 1j * F["F16y"], ORANGE, "F$_{16y}$ = %.0f N" % F["F16y"])]
    ax.annotate("", xy=(0.9, 0.62), xytext=(0.3, 0.62), arrowprops=dict(arrowstyle="-|>", color=GREY, lw=1.0))
    pl.seg(0.3 + 0.62j, 0.9 + 0.62j)
    pl.place("F$_{16x}$ = μ'F$_{16y}$\n(0 here)", [(0.95 + 0.62j, "left", "center")], GREY)
    qa, ua = _arrow(ax, pl, 0j, acc["aG6"], RED, L=0.4, lw=1.1)
    for q, u, cands, text, color in sl:
        pl.place(text, cands, color)
    pl.place("a$_{G6}$ = a$_D$", _ring(qa, 0.12, ua) + _ring(qa, 0.2, ua), RED)
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
