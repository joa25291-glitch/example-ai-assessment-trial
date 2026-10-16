"""Section 3 - graphical acceleration analysis at theta2 = 60 deg.

Normal (centripetal) components are CALCULATED from the velocities READ from the
velocity polygon (a_n = V^2 / r, directed towards the centre of relative
rotation).  Tangential components are only known in DIRECTION (perpendicular
to the link); their magnitudes come from the intersections in the polygon,
exactly as in the course steps 1-4.
"""
import numpy as np
from params import P
import kinematics as K
import drawing as Dw
import sec2_velocity as SV
from drawing import arrow, label, guide, intersect, INK, GREY, BLUE, RED, GREEN, ORANGE, PURPLE, TEAL

KA = 0.4                      # acceleration scale: 1 m/s^2 = 0.4 mm  (1 mm = 2.5 m/s^2)
POLE = complex(168.0, 158.0)
READ_MM = 0.5
READ_DEG = 0.5
perp = lambda z: 1j * z / abs(z)
unit = lambda z: z / abs(z)


def read_velocity(meas, name):
    return meas[name]["value"]


def construct(pos, vmeas, vpts):
    o = POLE
    # velocities needed for normal components (read from the velocity polygon)
    VA = P.omega2 * P.OA                                    # exact input
    VB = read_velocity(vmeas, "V_B")
    VBA = read_velocity(vmeas, "V_B/A")
    VCA = read_velocity(vmeas, "V_C/A")
    VCB = read_velocity(vmeas, "V_C/B")
    VDC = read_velocity(vmeas, "V_D/C")
    VG3A = Dw.round_to(abs(vpts["g3"] - vpts["a"]), 0.5) / SV.KV
    VG3B = Dw.round_to(abs(vpts["g3"] - vpts["b"]), 0.5) / SV.KV
    A, B, C, D, E, G3 = pos["A"], pos["B"], pos["C"], pos["D"], pos["E"], pos["G3"]
    rAB, rBE, rAC, rBC, rCD = abs(B - A), abs(B - E), abs(C - A), abs(C - B), abs(D - C)
    rAG3, rBG3 = abs(G3 - A), abs(G3 - B)
    an = dict(
        A=VA ** 2 / P.OA, BA=VBA ** 2 / rAB, B=VB ** 2 / rBE, CA=VCA ** 2 / rAC, CB=VCB ** 2 / rBC,
        DC=VDC ** 2 / rCD, G3A=VG3A ** 2 / rAG3, G3B=VG3B ** 2 / rBG3)
    a = o + KA * an["A"] * unit(pos["O"] - A)                      # a_A = a_A^n (alpha2 = 0)
    n1 = a + KA * an["BA"] * unit(A - B)                           # step 1
    n2 = o + KA * an["B"] * unit(E - B)
    b = intersect(n1, perp(B - A), n2, perp(B - E))
    n3 = a + KA * an["CA"] * unit(A - C)                           # step 2
    n4 = b + KA * an["CB"] * unit(B - C)
    c = intersect(n3, perp(C - A), n4, perp(C - B))
    n5 = c + KA * an["DC"] * unit(C - D)                           # step 3
    d = intersect(n5, perp(D - C), o, 1 + 0j)
    n6 = a + KA * an["G3A"] * unit(A - G3)                         # step 4 (mass centre of link 3)
    n7 = b + KA * an["G3B"] * unit(B - G3)
    g3 = intersect(n6, perp(G3 - A), n7, perp(G3 - B))
    pts = dict(o=o, a=a, b=b, c=c, d=d, n1=n1, n2=n2, n3=n3, n4=n4, n5=n5, n6=n6, n7=n7,
               g2=(o + a) / 2, g3=g3, g4=(o + b) / 2, g5=(c + d) / 2, g6=d)
    vin = dict(VA=VA, VB=VB, VBA=VBA, VCA=VCA, VCB=VCB, VDC=VDC, VG3A=VG3A, VG3B=VG3B)
    radii = dict(rAB=rAB, rBE=rBE, rAC=rAC, rBC=rBC, rCD=rCD, rAG3=rAG3, rBG3=rBG3)
    return pts, an, vin, radii


VECTORS = [
    ("a_A", "o", "a", "= a_A^n, ∥AO towards O (a_A^t = 0)"),
    ("a_B^n", "o", "n2", "∥BE towards E, = V_B²/BE"),
    ("a_B^t", "n2", "b", "⊥BE"),
    ("a_B", "o", "b", "resultant"),
    ("a_B/A^n", "a", "n1", "∥BA towards A, = V_B/A²/AB"),
    ("a_B/A^t", "n1", "b", "⊥AB"),
    ("a_B/A", "a", "b", "resultant"),
    ("a_C/A^n", "a", "n3", "∥CA towards A, = V_C/A²/AC"),
    ("a_C/A^t", "n3", "c", "⊥AC"),
    ("a_C/B^n", "b", "n4", "∥CB towards B, = V_C/B²/BC"),
    ("a_C/B^t", "n4", "c", "⊥BC"),
    ("a_C", "o", "c", "resultant"),
    ("a_D/C^n", "c", "n5", "∥DC towards C, = V_D/C²/CD"),
    ("a_D/C^t", "n5", "d", "⊥CD"),
    ("a_D", "o", "d", "∥x-x"),
    ("a_G2", "o", "g2", "mid-point of o'a'"),
    ("a_G3", "o", "g3", "step 4"),
    ("a_G4", "o", "g4", "mid-point of o'b'"),
    ("a_G5", "o", "g5", "mid-point of c'd'"),
    ("a_G6", "o", "d", "= a_D"),
]


def measure(pts):
    out = {}
    for name, f, t, desc in VECTORS:
        z = pts[t] - pts[f]
        L = abs(z)
        Lr = float(Dw.round_to(L, READ_MM))
        ang = np.degrees(np.angle(z)) % 360
        out[name] = dict(length_exact=float(L), length=Lr, value=Lr / KA, angle=float(Dw.round_to(ang, READ_DEG) % 360),
                         angle_exact=float(ang), desc=desc, frm=f, to=t)
    return out


def angular(meas, pos):
    def al(tname, r_vec, length):
        m = meas[tname]
        v = np.exp(1j * np.radians(m["angle"]))
        return float(np.sign((np.conj(r_vec) * v).imag) * m["value"] / length)
    return dict(alpha3=al("a_B/A^t", pos["B"] - pos["A"], P.AB),
                alpha4=al("a_B^t", pos["B"] - pos["E"], P.BE),
                alpha5=al("a_D/C^t", pos["D"] - pos["C"], P.CD),
                alpha3_check=al("a_C/A^t", pos["C"] - pos["A"], P.AC))


def fig_acceleration_polygon(path, pos, pts):
    fig, ax = Dw.paper(178, 168)
    o, a, b, c, d = pts["o"], pts["a"], pts["b"], pts["c"], pts["d"]
    A, B, C, D, E = pos["A"], pos["B"], pos["C"], pos["D"], pos["E"]
    lo = Dw.label_on
    # construction lines (direction of each unknown tangential component)
    guide(ax, pts["n1"], perp(B - A), -14, 128); label(ax, pts["n1"] - 12 * perp(B - A), "⊥AB", dx=4.5, color=GREY, size=6.2)
    guide(ax, pts["n2"], perp(B - E), -8, 166); label(ax, pts["n2"] + 160 * perp(B - E), "⊥BE", dx=0, dy=-3.2, color=GREY, size=6.2)
    guide(ax, pts["n3"], perp(C - A), -12, 108); label(ax, pts["n3"] - 10 * perp(C - A), "⊥AC", dx=4.5, color=GREY, size=6.2)
    guide(ax, pts["n4"], perp(C - B), -10, 72); label(ax, pts["n4"] + 70 * perp(C - B), "⊥BC", dx=0, dy=3, color=GREY, size=6.2)
    guide(ax, pts["n5"], perp(D - C), -12, 80); label(ax, pts["n5"] + 80 * perp(D - C), "⊥CD", dx=5.5, dy=-0.5, color=GREY, size=6.2)
    guide(ax, o, 1 + 0j, -86, 6); label(ax, o - 86, "∥ x-x", dx=1, dy=2.6, color=GREY, size=6.2, ha="left")
    # normal components (orange) and tangential components (purple)
    for f, t in (("o", "n2"), ("a", "n1"), ("a", "n3"), ("b", "n4"), ("c", "n5")):
        arrow(ax, pts[f], pts[t], color=ORANGE, lw=1.0, ms=5)
    for f, t in (("n2", "b"), ("n1", "b"), ("n3", "c"), ("n4", "c"), ("n5", "d")):
        arrow(ax, pts[f], pts[t], color=PURPLE, lw=0.9, ms=5, ls=(0, (4, 1.5)))
    # absolute accelerations from the pole (blue) and relative resultants (red)
    for q in (a, b, c, d):
        arrow(ax, o, q, color=BLUE, lw=1.4)
    for f, t in ((a, b), (a, c), (b, c), (c, d)):
        arrow(ax, f, t, color=RED, lw=1.1)
    for k in ["g2", "g3", "g4", "g5"]:
        ax.plot(pts[k].real, pts[k].imag, "o", ms=3.0, mfc="white", mec=GREEN, mew=0.9, zorder=9)
    for k in ["g3", "g5"]:   # g2' and g4' lie on o'a' and o'b' (mid-points), so only g3', g5' need own vectors
        arrow(ax, o, pts[k], color=GREEN, lw=0.8, ms=4, ls=(0, (3, 1.5)))
    # labels
    label(ax, o, "o'", dx=3.0, dy=-2.4, weight="bold", size=8.5)
    label(ax, a, "a'", dx=3.0, dy=-1.5, weight="bold", size=8.5)
    label(ax, b, "b'", dx=-3.0, dy=2.0, weight="bold", size=8.5)
    label(ax, c, "c'", dx=-3.4, dy=2.6, weight="bold", size=8.5)
    label(ax, d, "d'", dx=-1.5, dy=3.0, weight="bold", size=8.5)
    label(ax, pts["g2"], "g$_2$'", dx=3.4, dy=-1.0, color=GREEN, size=7)
    label(ax, pts["g3"], "g$_3$'", dx=-3.6, dy=0.5, color=GREEN, size=7)
    label(ax, pts["g4"], "g$_4$'", dx=0.0, dy=3.0, color=GREEN, size=7)
    label(ax, pts["g5"], "g$_5$'", dx=-3.8, dy=-1.5, color=GREEN, size=7)
    lo(ax, o, a, 0.42, "a$_A$", side=-1, color=BLUE, bbox=True)
    lo(ax, o, b, 0.40, "a$_B$", side=-1, dist=3.0, color=BLUE, bbox=True)
    lo(ax, o, c, 0.62, "a$_C$", side=-1, dist=3.0, color=BLUE, bbox=True)
    lo(ax, o, d, 0.55, "a$_D$", side=1, dist=3.0, color=BLUE, bbox=True)
    lo(ax, a, b, 0.62, "a$_{B/A}$", side=-1, dist=4.0, color=RED, bbox=True)
    lo(ax, a, c, 0.55, "a$_{C/A}$", side=-1, dist=4.5, color=RED, bbox=True)
    lo(ax, b, c, 0.45, "a$_{C/B}$", side=-1, dist=3.4, color=RED, bbox=True)
    lo(ax, c, d, 0.72, "a$_{D/C}$", side=1, dist=4.5, color=RED, bbox=True)
    lo(ax, o, pts["n2"], 0.5, "a$^n_B$", side=-1, dist=4.5, color=ORANGE, size=6.6)
    lo(ax, pts["n2"], b, 0.30, "a$^t_B$", side=-1, dist=3.4, color=PURPLE, size=6.6)
    lo(ax, a, pts["n1"], 0.5, "a$^n_{B/A}$", side=1, dist=4.0, color=ORANGE, size=6.6)
    lo(ax, pts["n1"], b, 0.25, "a$^t_{B/A}$", side=1, dist=4.0, color=PURPLE, size=6.6)
    lo(ax, a, pts["n3"], 0.5, "a$^n_{C/A}$", side=-1, dist=4.0, color=ORANGE, size=6.6)
    lo(ax, pts["n3"], c, 0.55, "a$^t_{C/A}$", side=1, dist=3.6, color=PURPLE, size=6.6)
    lo(ax, b, pts["n4"], 0.5, "a$^n_{C/B}$", side=1, dist=4.0, color=ORANGE, size=6.6)
    lo(ax, pts["n4"], c, 0.45, "a$^t_{C/B}$", side=1, dist=3.2, color=PURPLE, size=6.6)
    lo(ax, pts["n5"], d, 0.25, "a$^t_{D/C}$", side=1, dist=3.6, color=PURPLE, size=6.6)
    # scale & legend
    Dw.scale_bar(ax, complex(20, 157), 40, "Acceleration scale: 1 mm = 2.5 m/s² (40 mm = 100 m/s²)")
    leg = [("absolute acceleration (from pole o')", BLUE, "-"), ("relative acceleration (resultant)", RED, "-"),
           ("normal component a$^n$ = V²/r (calculated)", ORANGE, "-"),
           ("tangential component a$^t$ = αr (from polygon)", PURPLE, "--"), ("mass-centre acceleration", GREEN, "--")]
    for k, (t, col, ls) in enumerate(leg):
        y = 146 - 4.2 * k
        ax.plot([6, 13], [y, y], color=col, lw=1.3, ls=ls)
        ax.text(15, y, t, fontsize=6.0, va="center")
    ax.text(5, 14, "g$_2$', g$_4$' = mid-points of o'a' and o'b'", fontsize=5.8, va="center", color=INK)
    ax.text(5, 9.5, "a$^n_{D/C}$ = %.2f mm on paper (too short to draw)" % abs(pts["n5"] - c), fontsize=5.8,
            va="center", color=ORANGE)
    Dw.save(fig, path)


def run(outdir, vres):
    th2 = np.radians(P.theta2_deg)
    pos, vel, acc = K.solve(th2)
    vpts = {k: complex(*v) for k, v in vres["points"].items()}
    pts, an, vin, radii = construct(pos, vres["measured"], vpts)
    meas = measure(pts)
    ang = angular(meas, pos)
    fig_acceleration_polygon(f"{outdir}/figures/fig6_acceleration_polygon", pos, pts)
    return dict(points={k: [v.real, v.imag] for k, v in pts.items()}, measured=meas, angular=ang,
                normal=an, v_inputs=vin, radii=radii)


if __name__ == "__main__":
    vres = SV.run("../output")
    r = run("../output", vres)
    for k, v in r["measured"].items():
        print(f"{k:8s} L={v['length_exact']:7.2f} mm -> {v['length']:6.1f} mm = {v['value']:7.1f} m/s2 at {v['angle']:6.1f} deg")
    print(r["angular"]); print(r["normal"])
