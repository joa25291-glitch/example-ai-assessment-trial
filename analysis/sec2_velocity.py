"""Section 2 - graphical velocity analysis at theta2 = 60 deg.

The polygon is CONSTRUCTED graphically: only V_A (= w2 x OA) is computed; every
other point is found by intersecting direction lines, exactly as with set
squares on paper (course steps 1-4).  Lengths are then READ OFF the drawing to
the nearest 0.5 mm and angles to the nearest 0.5 deg (ruler/protractor
precision) and multiplied by the scale.
"""
import numpy as np
from params import P
import kinematics as K
import drawing as Dw
from drawing import arrow, label, guide, intersect, INK, GREY, BLUE, RED, GREEN, ORANGE, PURPLE, TEAL

KV = 20.0                    # velocity scale: 1 m/s = 20 mm  (1 mm = 0.05 m/s)
POLE = complex(160.0, 32.0)  # paper position of the pole o
READ_MM = 0.5
READ_DEG = 0.5

perp = lambda z: 1j * z / abs(z)       # unit vector perpendicular to z
unit = lambda z: z / abs(z)


def construct(pos):
    """Return paper coordinates of the polygon points o, a, b, c, d, g2..g6."""
    o = POLE
    VA = 1j * P.omega2 * (pos["A"] - pos["O"])          # the only computed input
    a = o + KV * VA
    AB, BE, AC, BC, CD = (pos["B"] - pos["A"], pos["B"] - pos["E"], pos["C"] - pos["A"],
                          pos["C"] - pos["B"], pos["D"] - pos["C"])
    b = intersect(a, perp(AB), o, perp(BE))              # step 1
    c = intersect(a, perp(AC), b, perp(BC))              # step 2
    d = intersect(o, 1 + 0j, c, perp(CD))                # step 3
    g3 = intersect(a, perp(pos["G3"] - pos["A"]), b, perp(pos["G3"] - pos["B"]))   # step 4
    g2 = (o + a) / 2       # G2 is the mid-point of OA, O fixed
    g4 = (o + b) / 2       # G4 is the mid-point of EB, E fixed
    g5 = (c + d) / 2       # G5 is the mid-point of CD
    return dict(o=o, a=a, b=b, c=c, d=d, g2=g2, g3=g3, g4=g4, g5=g5, g6=d)


VECTORS = [  # (name, from, to, description)
    ("V_A", "o", "a", "⊥ OA, = ω2·OA"),
    ("V_B", "o", "b", "⊥ BE"),
    ("V_B/A", "a", "b", "⊥ AB"),
    ("V_C", "o", "c", "from the image abc"),
    ("V_C/A", "a", "c", "⊥ AC"),
    ("V_C/B", "b", "c", "⊥ BC"),
    ("V_D", "o", "d", "∥ x-x (slider path)"),
    ("V_D/C", "c", "d", "⊥ CD"),
    ("V_G2", "o", "g2", "mid-point of oa"),
    ("V_G3", "o", "g3", "from ⊥AG3 & ⊥BG3"),
    ("V_G4", "o", "g4", "mid-point of ob"),
    ("V_G5", "o", "g5", "mid-point of cd"),
    ("V_G6", "o", "d", "= V_D"),
]


def measure(pts):
    out = {}
    for name, f, t, desc in VECTORS:
        z = pts[t] - pts[f]
        L = abs(z)
        Lr = Dw.round_to(L, READ_MM)
        ang = np.degrees(np.angle(z)) % 360
        angr = Dw.round_to(ang, READ_DEG) % 360
        out[name] = dict(length_exact=float(L), length=float(Lr), value=float(Lr / KV),
                         angle=float(angr), angle_exact=float(ang), desc=desc, frm=f, to=t)
    return out


def angular(meas, pos):
    """Angular velocities from relative velocities (sense from the polygon)."""
    def w(rel_name, r_vec, length):
        m = meas[rel_name]
        v = m["value"] * np.exp(1j * np.radians(m["angle"]))
        sense = np.sign((np.conj(r_vec) * v).imag)   # + if v is CCW-perpendicular to r
        return sense * m["value"] / length
    w3 = w("V_B/A", pos["B"] - pos["A"], P.AB)
    w4 = w("V_B", pos["B"] - pos["E"], P.BE)
    w5 = w("V_D/C", pos["D"] - pos["C"], P.CD)
    w3_check = w("V_C/A", pos["C"] - pos["A"], P.AC)
    return dict(omega3=float(w3), omega4=float(w4), omega5=float(w5), omega3_check=float(w3_check))


def fig_velocity_polygon(path, pos, pts):
    fig, ax = Dw.paper(178, 132)
    o, a, b, c, d = pts["o"], pts["a"], pts["b"], pts["c"], pts["d"]
    AB, BE, AC, BC, CD = (pos["B"] - pos["A"], pos["B"] - pos["E"], pos["C"] - pos["A"],
                          pos["C"] - pos["B"], pos["D"] - pos["C"])
    lo = Dw.label_on
    # construction lines (dashed) - each drawn through the known point in the known direction
    guide(ax, a, perp(AB), -8, 122); label(ax, a + 120 * perp(AB), "⊥AB", dx=5.5, dy=0, color=GREY, size=6.3)
    guide(ax, o, perp(BE), -10, 106); label(ax, o + 104 * perp(BE), "⊥BE", dx=-1, dy=-3, color=GREY, size=6.3)
    guide(ax, a, perp(AC), -8, 78); label(ax, a + 32 * perp(AC), "⊥AC", dx=-5.5, dy=0, color=GREY, size=6.3)
    guide(ax, b, perp(BC), -10, 62); label(ax, b - 8 * perp(BC), "⊥BC", dx=-1, dy=-3.2, color=GREY, size=6.3)
    guide(ax, o, 1 + 0j, -150, 12); label(ax, o - 150 + 0j, "∥ x-x", dx=1, dy=2.6, color=GREY, size=6.3, ha="left")
    guide(ax, c, perp(CD), -28, 12); label(ax, c - 26 * perp(CD), "⊥CD", dx=4.5, dy=-1.0, color=GREY, size=6.3)
    guide(ax, a, perp(pos["G3"] - pos["A"]), 60, 108, color="#9bc9a9")
    guide(ax, b, perp(pos["G3"] - pos["B"]), -55, -20, color="#9bc9a9")
    # vectors: absolute from the pole (blue), relative (red)
    for q in (a, b, c, d):
        arrow(ax, o, q, color=BLUE, lw=1.4)
    for p_, q_ in ((a, b), (a, c), (b, c), (c, d)):
        arrow(ax, p_, q_, color=RED, lw=1.2)
    arrow(ax, o, pts["g3"], color=GREEN, lw=1.0, ls=(0, (3, 1.5)))
    for k in ["g2", "g3", "g4", "g5"]:
        ax.plot(pts[k].real, pts[k].imag, "o", ms=3.0, mfc="white", mec=GREEN, mew=0.9, zorder=9)
    # point labels
    label(ax, o, "o", dx=2.8, dy=-2.4, weight="bold", size=8.5)
    label(ax, a, "a", dx=-2.8, dy=2.4, weight="bold", size=8.5)
    label(ax, b, "b", dx=1.5, dy=-3.0, weight="bold", size=8.5)
    label(ax, c, "c", dx=-3.0, dy=1.4, weight="bold", size=8.5)
    label(ax, d, "d", dx=-3.0, dy=-2.4, weight="bold", size=8.5)
    label(ax, pts["g2"], "g$_2$", dx=-1.5, dy=-3.0, color=GREEN, size=7)
    label(ax, pts["g3"], "g$_3$", dx=0.5, dy=3.0, color=GREEN, size=7)
    label(ax, pts["g4"], "g$_4$", dx=0.5, dy=-3.0, color=GREEN, size=7)
    label(ax, pts["g5"], "g$_5$", dx=-3.2, dy=0.0, color=GREEN, size=7)
    # vector names (offset perpendicular to each vector)
    lo(ax, o, a, 0.30, "V$_A$", side=-1, color=BLUE, bbox=True)
    lo(ax, o, b, 0.30, "V$_B$", side=1, color=BLUE, bbox=True)
    lo(ax, o, c, 0.55, "V$_C$", side=-1, color=BLUE, bbox=True)
    lo(ax, o, d, 0.80, "V$_D$", side=1, dist=3.0, color=BLUE, bbox=True)
    lo(ax, a, b, 0.42, "V$_{B/A}$", side=1, dist=5.0, color=RED, bbox=True)
    lo(ax, a, c, 0.45, "V$_{C/A}$", side=-1, dist=5.5, color=RED, bbox=True)
    lo(ax, b, c, 0.72, "V$_{C/B}$", side=-1, dist=3.2, color=RED, bbox=True)
    lo(ax, c, d, 0.50, "V$_{D/C}$", side=1, dist=6.0, color=RED, bbox=True)
    lo(ax, o, pts["g3"], 0.30, "V$_{G3}$", side=-1, dist=2.8, color=GREEN, size=6.5, bbox=True)
    # scale and notes (top-right, clear of the polygon)
    Dw.scale_bar(ax, complex(118, 122), 40, "Velocity scale: 1 m/s = 20 mm")
    notes = ["Step 1:  V$_B$ = V$_A$ + V$_{B/A}$       (⊥BE) = (⊥OA) + (⊥AB)",
             "Step 2:  V$_A$ + V$_{C/A}$ = V$_B$ + V$_{C/B}$   (⊥AC), (⊥BC)",
             "Step 3:  V$_D$ = V$_C$ + V$_{D/C}$        (∥x-x) = (known) + (⊥CD)",
             "Step 4:  V$_A$ + V$_{G3/A}$ = V$_B$ + V$_{G3/B}$   (⊥AG$_3$), (⊥BG$_3$)",
             "V$_A$ = ω$_2$·OA = %.3f × %.3f = %.3f m/s, ⊥OA" % (P.omega2, P.OA, P.omega2 * P.OA),
             "g$_2$, g$_4$, g$_5$: mid-points of oa, ob, cd (velocity image)"]
    for k, t in enumerate(notes):
        ax.text(92, 110 - 4.4 * k, t, fontsize=6.2, va="center")
    Dw.save(fig, path)


def run(outdir):
    th2 = np.radians(P.theta2_deg)
    pos, vel, acc = K.solve(th2)
    pts = construct(pos)
    meas = measure(pts)
    ang = angular(meas, pos)
    fig_velocity_polygon(f"{outdir}/figures/fig5_velocity_polygon", pos, pts)
    return dict(points={k: [v.real, v.imag] for k, v in pts.items()}, measured=meas, angular=ang)


if __name__ == "__main__":
    r = run("../output")
    for k, v in r["measured"].items():
        print(f"{k:6s} L={v['length_exact']:7.2f} mm -> read {v['length']:6.1f} mm = {v['value']:6.3f} m/s at {v['angle']:6.1f} deg")
    print(r["angular"])
