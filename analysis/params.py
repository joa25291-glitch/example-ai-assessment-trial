"""Example parameters for the MIET1077 six-bar linkage practice assignment.

The handout leaves OA, beta, theta1 and n2 blank: each student is GIVEN these
four values by the lecturer.  The remaining lengths are SELECTED by the student
using the "data selection" (non-equality) equations from the Week 1 lecture:

    OE = CD = 4 OA
    OA + AB < OE + BE
    OA + OE < AB + BE
    AC = 3/5 AB
    H  = OA / 4

All values below are EXAMPLE values chosen for this worked practice solution.
Two parameter sets are defined in PRESETS; pick one with the MIET_PRESET
environment variable (default "example1"):

    MIET_PRESET=mock2 python3 analysis/run_all.py

To practise with your own made-up values, add a preset with the four "given"
values (and AB, BE re-selected to satisfy the inequalities).
"""
from dataclasses import dataclass
import math
import os


@dataclass
class Params:
    # ---- given by lecturer (example values) ----
    OA: float = 0.180          # m      crank length l2
    beta_deg: float = 20.0     # deg    angle BAC of the ternary link 3
    theta1_deg: float = 10.0   # deg    angle of OE above the horizontal
    n2_rpm: float = 440.0      # rpm    crank speed (counter-clockwise, alpha2 = 0)
    # ---- selected by student ----
    AB: float = 0.660          # m      l3
    BE: float = 0.360          # m      l4
    # ---- masses / inertias from the handout (item 6) ----
    m2: float = 1.0
    I2: float = 0.002
    m3: float = 2.5
    I3: float = 0.008
    m4: float = 1.5
    I4: float = 0.005
    m5: float = 1.8
    I5: float = 0.006
    m6: float = 0.9
    theta2_deg: float = 60.0   # analysis position
    # Location of G3 on the ternary link.  The handout says "CG at beta/2 and AB/2":
    #   "bisector": distance AB/2 from A along the bisector of angle beta (literal reading, used here)
    #   "centroid": centroid of triangle ABC (as sketched on the Week 9 slide / WM polygon default)
    G3_mode: str = "bisector"
    # ---- bookkeeping for the generated report ----
    name: str = "example1"                       # preset key; also names the output folder
    label: str = "Example 1"
    note: str = "example values — replace with the ones your lecturer gives you"

    @property
    def out_dir(self):
        """Folder (relative to the repository root) for figures, tables and results.json."""
        return "output" if self.name == "example1" else "output_" + self.name

    # ---- derived (data selection equations) ----
    @property
    def OE(self):
        return 4.0 * self.OA

    @property
    def CD(self):
        return 4.0 * self.OA

    @property
    def AC(self):
        return 0.6 * self.AB

    @property
    def H(self):
        return self.OA / 4.0

    @property
    def beta(self):
        return math.radians(self.beta_deg)

    @property
    def theta1(self):
        return math.radians(self.theta1_deg)

    @property
    def omega2(self):
        return self.n2_rpm * 2.0 * math.pi / 60.0

    @property
    def g3_local(self):
        """G3 relative to A in a frame fixed to link 3 (x along A->B, complex number)."""
        import cmath
        if self.G3_mode == "centroid":
            return (self.AB + self.AC * cmath.exp(-1j * self.beta)) / 3.0
        return 0.5 * self.AB * cmath.exp(-1j * self.beta / 2.0)

    # handy aliases used in the course notation
    @property
    def l1(self):
        return self.OE

    @property
    def l2(self):
        return self.OA

    @property
    def l3(self):
        return self.AB

    @property
    def l4(self):
        return self.BE

    @property
    def l5(self):
        return self.CD

    def selection_checks(self):
        """Return list of (description, lhs, rhs, ok) for the selection inequalities
        plus the Grashof check of the four-bar O-A-B-E."""
        OA, AB, BE, OE = self.OA, self.AB, self.BE, self.OE
        links = sorted([OA, AB, BE, OE])
        s, l = links[0], links[-1]
        p, q = links[1], links[2]
        return [
            ("OA + AB < OE + BE", OA + AB, OE + BE, OA + AB < OE + BE),
            ("OA + OE < AB + BE", OA + OE, AB + BE, OA + OE < AB + BE),
            ("Grashof: s + l <= p + q", s + l, p + q, s + l <= p + q),
            ("OA is the shortest link (crank-rocker)", OA, s, abs(OA - s) < 1e-12),
        ]


PRESETS = {
    # worked example of the first report
    "example1": Params(),
    # mock test: made-up values (not taken from any class allocation sheet)
    "mock2": Params(OA=0.265, beta_deg=16.5, theta1_deg=12.5, n2_rpm=472.0, AB=0.970, BE=0.530,
                    name="mock2", label="Practice set 2 (mock test)",
                    note="made-up mock-test values, for practice only"),
}

PRESET = os.environ.get("MIET_PRESET", "example1")
if PRESET not in PRESETS:
    raise SystemExit(f"unknown MIET_PRESET {PRESET!r}; choose one of {sorted(PRESETS)}")
P = PRESETS[PRESET]
