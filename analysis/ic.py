"""Instantaneous centres (ICs) of the six-bar by the Aronhold-Kennedy theorem.

n = 6 links  ->  C = n(n-1)/2 = 15 ICs.
Known by inspection (7): I12=O, I23=A, I34=B, I14=E, I35=C, I56=D (pins) and
I16 at infinity in the direction perpendicular to the slider path (vertical).
Unknown (8): I13, I24, I15, I36, I25, I45, I26, I46 - each found as the
intersection of two Kennedy lines (three ICs of any three links are collinear).
"""
import numpy as np

INF_VERTICAL = ("inf", 1j)  # point at infinity in the vertical direction


def _line(p, q):
    """Line through IC p and IC q.  Returns (point, direction)."""
    if isinstance(p, tuple):
        p, q = q, p
    if isinstance(q, tuple):            # q at infinity: line through p with q's direction
        return p, q[1]
    return p, q - p


def _intersect(L1, L2):
    (p1, d1), (p2, d2) = L1, L2
    M = np.array([[d1.real, -d2.real], [d1.imag, -d2.imag]])
    t = np.linalg.solve(M, np.array([(p2 - p1).real, (p2 - p1).imag]))
    return p1 + t[0] * d1


# Kennedy steps: (unknown IC, (pair of ICs on line 1), (pair on line 2), (check pair))
STEPS = [
    ("13", ("12", "23"), ("14", "34"), "links 1-2-3 and 1-4-3"),
    ("24", ("12", "14"), ("23", "34"), "links 2-1-4 and 2-3-4"),
    ("15", ("13", "35"), ("16", "56"), "links 1-3-5 and 1-6-5"),
    ("36", ("35", "56"), ("13", "16"), "links 3-5-6 and 3-1-6"),
    ("25", ("23", "35"), ("12", "15"), "links 2-3-5 and 2-1-5"),
    ("45", ("34", "35"), ("14", "15"), "links 4-3-5 and 4-1-5"),
    ("26", ("12", "16"), ("23", "36"), "links 2-1-6 and 2-3-6"),
    ("46", ("14", "16"), ("34", "36"), "links 4-1-6 and 4-3-6"),
]

CHECKS = {  # a third Kennedy line through each unknown IC, used as a check
    "13": ("15", "35"), "24": ("25", "45"), "15": ("12", "25"), "36": ("23", "26"),
    "25": ("24", "45"), "45": ("46", "56"), "26": ("25", "56"), "46": ("45", "56"),
}


def find_ics(pos):
    I = {"12": pos["O"], "23": pos["A"], "34": pos["B"], "14": pos["E"],
         "35": pos["C"], "56": pos["D"], "16": INF_VERTICAL}
    for name, l1, l2, _ in STEPS:
        I[name] = _intersect(_line(I[l1[0]], I[l1[1]]), _line(I[l2[0]], I[l2[1]]))
    # collinearity check of the third Kennedy line
    checks = {}
    for name, (a, b) in CHECKS.items():
        pa, pb, pc = I[a], I[b], I[name]
        if isinstance(pa, tuple) or isinstance(pb, tuple):
            continue
        d = pb - pa
        dist = abs((pc - pa).real * d.imag - (pc - pa).imag * d.real) / abs(d)
        checks[name] = dist
    return I, checks


def ic_velocities(pos, I, omega2):
    """Velocities of A, B, C, D (and omegas) from the ICs only."""
    rA = abs(pos["A"] - I["12"])
    VA = omega2 * rA
    # link 3 rotates about I13: omega3 = VA / |I13 A|, sign from geometry
    def omega_about(ic, point, v_point):
        r = point - ic
        # v = j*omega*r  ->  omega = Im(v * conj(j r)) / |r|^2
        return (v_point * np.conj(1j * r)).real / abs(r) ** 2
    vA_vec = 1j * omega2 * (pos["A"] - I["12"])
    w3 = omega_about(I["13"], pos["A"], vA_vec)
    vB_vec = 1j * w3 * (pos["B"] - I["13"])
    vC_vec = 1j * w3 * (pos["C"] - I["13"])
    w4 = omega_about(I["14"], pos["B"], vB_vec)
    w5 = omega_about(I["15"], pos["C"], vC_vec)
    vD_vec = 1j * w5 * (pos["D"] - I["15"])
    vD_alt = 1j * omega2 * (I["26"] - I["12"])        # I26 as a point on link 2
    # angular-velocity-ratio theorem: I12, I14, I24 are collinear and I24 has the
    # same velocity on links 2 and 4 -> w2 (s24 - s12) = w4 (s24 - s14)
    u = (I["14"] - I["12"]) / abs(I["14"] - I["12"])
    s = lambda z: ((z - I["12"]) * np.conj(u)).real
    w4_ratio = omega2 * (s(I["24"]) - s(I["12"])) / (s(I["24"]) - s(I["14"]))
    return dict(VA=vA_vec, VB=vB_vec, VC=vC_vec, VD=vD_vec, VD_from_I26=vD_alt,
                omega3=w3, omega4=w4, omega5=w5, omega4_ratio=w4_ratio,
                rA=rA, r13A=abs(pos["A"] - I["13"]), r13B=abs(pos["B"] - I["13"]),
                r13C=abs(pos["C"] - I["13"]), r15C=abs(pos["C"] - I["15"]),
                r15D=abs(pos["D"] - I["15"]))
