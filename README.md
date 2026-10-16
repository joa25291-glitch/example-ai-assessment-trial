# MIET1077 six-bar linkage: worked practice solution

A complete worked solution of the MIET1077 *Mechanics of Machines* major assignment, "Kinematic and dynamic analysis of a linkage mechanism" (six-bar: crank, ternary coupler, rocker, connecting rod, slider). It is meant as a study reference before the in-person assessment.

The four lecturer-given parameters are **example values** (OA = 180 mm, β = 20°, θ₁ = 10°, n₂ = 440 rpm). In the real assessment you get your own, and every number changes.

## What to open

| File | What it is |
|---|---|
| `report/MIET1077_SixBar_Worked_Solution.pdf` | The full report, laid out like the rubric: cover, TOC, one-page summary, objectives, theory, data selection, Sections 1–8, conclusions, references |
| `report/MIET1077_InPerson_Workflow_Guide.pdf` | 3-page condensed workflow: the order of the steps, the formulas to know, the Kennedy-circle steps and the common traps |
| `report/MIET1077_SixBar_Worked_Solution.docx` | Editable Word version of the report |
| `output/figures/` | Every drawing (SVG + PNG). The graphical constructions are true to scale when printed at 100 % |
| `output/tables/`, `output/results.json` | All numerical results |

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

## Re-running with your own parameters

Edit the four "given" values (and AB, BE) in `analysis/params.py`, then:

```bash
pip install -r requirements.txt          # numpy, scipy, matplotlib, jinja2, pypdf
python3 analysis/run_all.py              # figures, tables, results.json
python3 report/build_report.py           # PDF + DOCX (needs pandoc, node and playwright/chromium)
```

`analysis/params.py` lists the data-selection rules. Pick AB and BE that satisfy both inequalities; around 3.5–3.8·OA and 2·OA works well.
