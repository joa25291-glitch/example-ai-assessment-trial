"""Drawing scales for the graphical constructions.

Each scale is picked from a list of standard engineering scales so that the
construction fills, but fits, its sheet (A4 width, 178 mm frame) for whatever
parameter set is active.  The values are exported to results.json and quoted
in the report, so the figures and the text always agree.

    space diagram, 16 positions, IC diagram   1:N
    velocity polygon                          1 m/s = KV mm
    acceleration polygon                      1 mm = 1/KA m/s^2
    velocity arrows on the IC diagram         1 m/s = KVI mm
"""
import numpy as np
from params import P
import kinematics as K
import ic as IC

SPACE_N = [5, 10, 15, 20, 25, 30, 40, 50, 75, 100, 150, 200]                  # 1:N
OVERVIEW_N = [20, 25, 30, 40, 50, 75, 100, 150, 200, 250, 300, 400, 500, 1000]
KV_MM = [50, 40, 25, 20, 10, 5, 4, 2.5, 2, 1, 0.5]                              # mm per m/s
KA_MM = [1, 0.5, 0.4, 0.25, 0.2, 0.1, 0.05, 0.04, 0.025, 0.02, 0.01]            # mm per m/s^2
KVI_MM = [10, 5, 4, 2.5, 2, 1, 0.5]                                             # mm per m/s


def bbox(points):
    pts = np.concatenate([np.atleast_1d(np.asarray(z, dtype=complex)) for z in points])
    return pts.real.min(), pts.real.max(), pts.imag.min(), pts.imag.max()


def _largest(choices, fits):
    """First (largest drawing) entry of `choices` for which fits(value) is true."""
    for v in choices:
        if fits(v):
            return v
    return choices[-1]


def _at60():
    return K.solve(np.radians(P.theta2_deg))


def mechanism_bbox():
    """Bounding box (m) of every joint over a full revolution, plus O and E."""
    ps = K.positions(np.radians(np.arange(0.0, 360.0, 1.0)))
    return bbox([ps[k] for k in "ABCD"] + [0j, ps["E"]])


def velocity_points(pos, vel):
    return [0j, vel["VA"], vel["VB"], vel["VC"], vel["VD"], vel["VG3"]]


def acceleration_points(pos, vel, acc):
    """Polygon points and the tips of the normal components (exact values, m/s^2)."""
    A, B, C, D, E, G3 = (pos[k] for k in ("A", "B", "C", "D", "E", "G3"))
    w3, w4, w5 = vel["omega3"], vel["omega4"], vel["omega5"]
    aA, aB, aC = acc["aA"], acc["aB"], acc["aC"]
    tips = [aA - w3 ** 2 * (B - A), -w4 ** 2 * (B - E), aA - w3 ** 2 * (C - A), aB - w3 ** 2 * (C - B),
            aC - w5 ** 2 * (D - C), aA - w3 ** 2 * (G3 - A), aB - w3 ** 2 * (G3 - B)]
    return [0j, aA, aB, aC, acc["aD"], acc["aG3"]] + tips


def compute():
    pos, vel, acc = _at60()
    x0, x1, y0, y1 = mechanism_bbox()
    # space diagram: the joint paths must fit in ~145 mm of the 178 mm frame
    N = _largest(SPACE_N, lambda n: (x1 - x0) * 1000.0 / n <= 145.0 and (y1 - y0) * 1000.0 / n <= 80.0)
    # IC diagram (everything except I15 and I16, which are far away / at infinity)
    I, _ = IC.find_ics(pos)
    near = [pos[k] for k in ("O", "A", "B", "C", "D", "E")] + [I[k] for k in ("13", "24", "36", "25", "45", "26", "46")]
    ix0, ix1, iy0, iy1 = bbox(near)
    N_ic = _largest(SPACE_N, lambda n: (ix1 - ix0) * 1000.0 / n <= 150.0 and (iy1 - iy0) * 1000.0 / n <= 112.0)
    # overview of all ICs including I15
    allp = near + [I["15"]]
    ox0, ox1, oy0, oy1 = bbox(allp)
    N_ov = _largest(OVERVIEW_N, lambda n: (ox1 - ox0) * 1000.0 / n <= 62.0 and (oy1 - oy0) * 1000.0 / n <= 105.0)
    # velocity polygon
    vx0, vx1, vy0, vy1 = bbox(velocity_points(pos, vel))
    KV = _largest(KV_MM, lambda k: (vx1 - vx0) * k <= 165.0 and (vy1 - vy0) * k <= 118.0)
    # acceleration polygon
    ax0, ax1, ay0, ay1 = bbox(acceleration_points(pos, vel, acc))
    KA = _largest(KA_MM, lambda k: (ax1 - ax0) * k <= 160.0 and (ay1 - ay0) * k <= 152.0)
    # velocity arrows drawn on the IC diagram
    vmax = max(abs(vel[k]) for k in ("VA", "VB", "VC", "VD"))
    KVI = _largest(KVI_MM, lambda k: vmax * k <= 36.0)
    return dict(N=N, N_ic=N_ic, N_ov=N_ov, KV=KV, KA=KA, KVI=KVI,
                bbox=dict(mech=(x0, x1, y0, y1), ic=(ix0, ix1, iy0, iy1), overview=(ox0, ox1, oy0, oy1),
                          vel=(vx0, vx1, vy0, vy1), acc=(ax0, ax1, ay0, ay1)))


S = compute()
N, N_IC, N_OV = S["N"], S["N_ic"], S["N_ov"]
KV, KA, KVI = S["KV"], S["KA"], S["KVI"]


def fmt(x):
    """2.5 -> '2.5', 20.0 -> '20'"""
    return ("%g" % x)


def strings():
    """Human-readable scale statements used in figures and in the report."""
    bar = 50.0                                    # mm, length of the space-scale bar
    return dict(
        space="1:%d" % N, ic="1:%d" % N_IC, overview="1:%d" % N_OV,
        space_bar="Space scale 1:%d  (%g mm = %g m)" % (N, bar, bar * N / 1000.0),
        vel="1 m/s = %s mm" % fmt(KV), vel_inv="1 mm = %s m/s" % fmt(1.0 / KV),
        acc="1 mm = %s m/s²" % fmt(1.0 / KA), acc_k="1 m/s² = %s mm" % fmt(KA),
        ic_vel="1 m/s = %s mm" % fmt(KVI),
        read_full=float(0.5 * N_IC),              # 0.5 mm on the IC drawing, in mm full size
    )


if __name__ == "__main__":
    import json
    print(P.name, json.dumps({k: (v if not isinstance(v, dict) else {a: [round(t, 4) for t in b] for a, b in v.items()})
                              for k, v in S.items()}, indent=1))
    print(strings())
