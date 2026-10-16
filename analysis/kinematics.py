"""Closed-form (vector loop / complex number) kinematics of the six-bar linkage.

Follows the course template ("Approach for Analytical Part of Major Assignment",
Week 7): Part 1 crank (A), Part 2 coupler & rocker (B, C), Part 3 slider (D).

Coordinate system: origin at the crank pivot O, x to the right, y up.
All angles are measured counter-clockwise from +x.  theta5 is the angle of the
vector C->D.  Units: SI (m, rad, s).
"""
import numpy as np
from params import P as DEFAULT_P

J = 1j


def positions(theta2, p=DEFAULT_P, branch=-1):
    """Return dict of joint/CG positions (complex numbers) and link angles."""
    l1, l2, l3, l4, l5 = p.l1, p.l2, p.l3, p.l4, p.l5
    th1 = p.theta1
    # Part 1 - crank
    A = l2 * np.exp(J * theta2)
    E = l1 * np.exp(J * th1)
    # Part 2 - four-bar O-A-B-E (course template)
    Re = l1 * np.cos(th1) - l2 * np.cos(theta2)      # x-component of A->E
    Im = l1 * np.sin(th1) - l2 * np.sin(theta2)      # y-component of A->E
    Cc = (Re ** 2 + Im ** 2 + l4 ** 2 - l3 ** 2) / (2.0 * l4)
    disc = Re ** 2 + Im ** 2 - Cc ** 2
    if np.any(np.asarray(disc) < 0):
        raise ValueError("Four-bar cannot be assembled at this crank angle")
    t4 = (-Im + branch * np.sqrt(disc)) / (Cc - Re)
    theta4 = 2.0 * np.arctan(t4)
    theta3 = 2.0 * np.arctan((Im + l4 * np.sin(theta4)) / (l3 + Re + l4 * np.cos(theta4)))
    B = E + l4 * np.exp(J * theta4)
    # point C on the ternary link (C lies BELOW AB, so theta_AC = theta3 - beta)
    C = A + p.AC * np.exp(J * (theta3 - p.beta))
    # Part 3 - slider: C + l5 e^{j th5} = S - jH
    s5 = -(p.H + C.imag) / l5
    theta5 = np.arcsin(s5)                          # D to the right of C
    S = C.real + l5 * np.cos(theta5)
    D = S - J * p.H
    # mass centres
    G2 = 0.5 * l2 * np.exp(J * theta2)
    G3 = A + p.g3_local * np.exp(J * theta3)
    G4 = E + 0.5 * l4 * np.exp(J * theta4)
    G5 = C + 0.5 * l5 * np.exp(J * theta5)
    G6 = D
    return dict(theta2=theta2, theta3=theta3, theta4=theta4, theta5=theta5, S=S,
                O=0j, A=A, B=B, C=C, D=D, E=E, G2=G2, G3=G3, G4=G4, G5=G5, G6=G6)


def velocities(pos, omega2, p=DEFAULT_P):
    l2, l3, l4, l5 = p.l2, p.l3, p.l4, p.l5
    th2, th3, th4, th5 = pos["theta2"], pos["theta3"], pos["theta4"], pos["theta5"]
    w3 = l2 * omega2 * np.sin(th4 - th2) / (l3 * np.sin(th3 - th4))
    w4 = l2 * omega2 * np.sin(th2 - th3) / (l4 * np.sin(th4 - th3))
    VA = J * l2 * omega2 * np.exp(J * th2)
    VB = J * l4 * w4 * np.exp(J * th4)
    VC = VA + J * p.AC * w3 * np.exp(J * (th3 - p.beta))
    w5 = -VC.imag / (l5 * np.cos(th5))
    VD = VC.real - l5 * w5 * np.sin(th5) + 0j
    VG2 = 0.5 * VA
    VG3 = VA + J * w3 * p.g3_local * np.exp(J * th3)
    VG4 = 0.5 * VB
    VG5 = VC + J * 0.5 * l5 * w5 * np.exp(J * th5)
    return dict(omega2=omega2, omega3=w3, omega4=w4, omega5=w5,
                VA=VA, VB=VB, VC=VC, VD=VD,
                VBA=VB - VA, VCA=VC - VA, VCB=VC - VB, VDC=VD - VC,
                VG2=VG2, VG3=VG3, VG4=VG4, VG5=VG5, VG6=VD)


def accelerations(pos, vel, alpha2=0.0, p=DEFAULT_P):
    a, b, c, l5 = p.l2, p.l3, p.l4, p.l5
    th2, th3, th4, th5 = pos["theta2"], pos["theta3"], pos["theta4"], pos["theta5"]
    w2, w3, w4, w5 = vel["omega2"], vel["omega3"], vel["omega4"], vel["omega5"]
    # course constants (Week 7, acceleration analysis)
    Ak = c * np.sin(th4)
    Bk = b * np.sin(th3)
    Ck = a * alpha2 * np.sin(th2) + a * w2 ** 2 * np.cos(th2) + b * w3 ** 2 * np.cos(th3) - c * w4 ** 2 * np.cos(th4)
    Dk = c * np.cos(th4)
    Ek = b * np.cos(th3)
    Fk = a * alpha2 * np.cos(th2) - a * w2 ** 2 * np.sin(th2) - b * w3 ** 2 * np.sin(th3) + c * w4 ** 2 * np.sin(th4)
    al3 = (Ck * Dk - Ak * Fk) / (Ak * Ek - Bk * Dk)
    al4 = (Ck * Ek - Bk * Fk) / (Ak * Ek - Bk * Dk)
    aA = (J * alpha2 - w2 ** 2) * a * np.exp(J * th2)
    aB = (J * al4 - w4 ** 2) * c * np.exp(J * th4)
    aC = aA + (J * al3 - w3 ** 2) * p.AC * np.exp(J * (th3 - p.beta))
    al5 = (l5 * w5 ** 2 * np.sin(th5) - aC.imag) / (l5 * np.cos(th5))
    aD = aC.real - l5 * al5 * np.sin(th5) - l5 * w5 ** 2 * np.cos(th5) + 0j
    aG2 = 0.5 * aA
    aG3 = aA + (J * al3 - w3 ** 2) * p.g3_local * np.exp(J * th3)
    aG4 = 0.5 * aB
    aG5 = aC + (J * al5 - w5 ** 2) * 0.5 * l5 * np.exp(J * th5)
    return dict(alpha2=alpha2, alpha3=al3, alpha4=al4, alpha5=al5,
                aA=aA, aB=aB, aC=aC, aD=aD,
                aBA=aB - aA, aCA=aC - aA, aCB=aC - aB, aDC=aD - aC,
                aG2=aG2, aG3=aG3, aG4=aG4, aG5=aG5, aG6=aD,
                consts=dict(A=Ak, B=Bk, C=Ck, D=Dk, E=Ek, F=Fk))


def solve(theta2, p=DEFAULT_P, omega2=None, alpha2=0.0, branch=-1):
    if omega2 is None:
        omega2 = p.omega2
    pos = positions(theta2, p, branch)
    vel = velocities(pos, omega2, p)
    acc = accelerations(pos, vel, alpha2, p)
    return pos, vel, acc
