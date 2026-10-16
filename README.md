# MIET1077 six-bar linkage: worked practice solution

A complete worked solution of the MIET1077 *Mechanics of Machines* major assignment, "Kinematic and dynamic analysis of a linkage mechanism" (six-bar: crank, ternary coupler, rocker, connecting rod, slider). It is meant as a study reference before the in-person assessment.

Two parameter sets are worked in full. In the real assessment you get your own four values, and every number changes.

| Set | Given: OA, β, θ₁, n₂ | Selected: AB, BE | Report |
|---|---|---|---|
| Example 1 | 180 mm, 20°, 10°, 440 rpm | 660 mm, 360 mm | `report/MIET1077_SixBar_Worked_Solution.pdf` |
| Practice set 2 (mock test) | 265 mm, 16.5°, 12.5°, 472 rpm | 970 mm, 530 mm | `report/MIET1077_SixBar_MockTest2_Worked_Solution.pdf` |

Both sets are made-up values, not anyone's allocated data. For the mock test, try the whole analysis by hand first, then check against the PDF.

## What to open

| File | What it is |
|---|---|
| `report/MIET1077_SixBar_Worked_Solution.pdf` | Example 1: the full report, laid out like the rubric (cover, TOC, one-page summary, objectives, theory, data selection, Sections 1–8, conclusions, references) |
| `report/MIET1077_SixBar_MockTest2_Worked_Solution.pdf` | Practice set 2 (mock test): the same full report for the mock-test values |
| `report/MIET1077_InPerson_Workflow_Guide.pdf` | 3-page condensed workflow: the order of the steps, the formulas to know, the Kennedy-circle steps and the common traps |
| `report/*.docx` | Editable Word versions of the two reports |
| `output/figures/`, `output_mock2/figures/` | Every drawing (SVG + PNG). The graphical constructions are true to scale when printed at 100 % |
| `output*/tables/`, `output*/results.json` | All numerical results |

## What is in the solution

1. Graphical position analysis: data-selection equations, a sample construction, 16 positions, joint paths, limits of motion.
2. Velocity polygon at θ₂ = 60° (drawn to scale, constructed by intersections, values read at ruler precision).
3. Acceleration polygon at θ₂ = 60° (normal components tabulated first).
4. All 15 instantaneous centres (Kennedy circle step by step) and the velocities of A, B, C and D.
5. Analytical vector-loop / complex-number solution (course template Parts 1–3), and a comparison table.
6. Free-body diagrams and the 14 × 14 matrix of joint forces and driving torque (frictionless, and μ = 0.3).
7. Computational model over one revolution: an independent multibody simulation that works the way Working Model does, plus step-by-step instructions for building the model in Working Model 2D.
8. Shaking force and moment, and an analytical balancing strategy (Berkof–Lowen counterweights and a least-squares optimum).

Self-checks built into the code:
- analytical vs numerical differentiation;
- analytical vs the independent simulation (agree to < 10⁻⁷);
- power balance for the driving torque;
- third-line collinearity for every IC.

## Generating a practice set

Parameter sets live in `PRESETS` in `analysis/params.py` and are chosen with the `MIET_PRESET` environment variable (default `example1`):

```bash
pip install -r requirements.txt                              # numpy, scipy, matplotlib, jinja2, pypdf
MIET_PRESET=mock2 python3 analysis/run_all.py                # figures, tables, results.json -> output_mock2/
MIET_PRESET=mock2 python3 report/build_report.py             # PDF + DOCX (needs pandoc, node and playwright/chromium)
MIET_PRESET=mock2 python3 analysis/verify_independent.py     # independent recheck of every reported number
```

To make a fresh answer key, add a preset with made-up values for the four "given" parameters (and AB, BE). `analysis/params.py` lists the data-selection rules. Pick AB and BE that satisfy both inequalities; about 3.5–3.8·OA and 2·OA works well.

The drawing scales are chosen automatically from standard scales so each construction fits an A4 page (`analysis/layout.py`): 1:10 and 1:15 for the space diagrams, 1 m/s = 20 or 10 mm for the velocity polygons, 1 mm = 2.5 or 5 m/s² for the acceleration polygons. Every direction, rotation sense and scale quoted in the report text is derived from the results, so a new preset gives a consistent report. The 3-page guide is written around example 1 and is only built for it.
