"""Build the worked-solution report (Markdown -> HTML/PDF via Chromium, and DOCX).

    python3 report/build_report.py            # uses output/results.json (run analysis/run_all.py first)

Every number in the report text is pulled from output/results.json, so changing
the parameters in analysis/params.py and re-running both scripts regenerates a
consistent report.
"""
import cmath
import json
import math
import os
import re
import shutil
import subprocess
import sys

import jinja2

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "output")
sys.path.insert(0, os.path.join(ROOT, "analysis"))

KATEX_DIR = os.environ.get("KATEX_DIR", "")
PDF_NAME = "MIET1077_SixBar_Worked_Solution.pdf"
GUIDE_PDF_NAME = "MIET1077_InPerson_Workflow_Guide.pdf"
DOCX_NAME = "MIET1077_SixBar_Worked_Solution.docx"


# ---------------------------------------------------------------- formatting helpers
def n(x, d=2):
    x = float(x)
    if abs(x) < 0.5 * 10 ** (-d):
        x = 0.0
    s = f"{x:,.{d}f}"
    return s.replace("-", "−")


def c(z):
    return complex(*z) if isinstance(z, (list, tuple)) else complex(z)


def mag(z):
    return abs(c(z))


def ang(z):
    return math.degrees(cmath.phase(c(z))) % 360


def pct(a, b):
    return 100.0 * (float(a) - float(b)) / float(b)


def sense(w):
    return "CCW" if w > 0 else "CW"


def vname(name):
    """'V_B/A' -> '$V_{B/A}$'"""
    return "$V_{%s}$" % name[2:]


def aname(name):
    """'a_B/A^n' -> '$a^{n}_{B/A}$', 'a_G3' -> '$a_{G3}$'"""
    base = name[2:]
    sup = ""
    if "^" in base:
        base, sup = base.split("^")
    return "$a^{%s}_{%s}$" % (sup, base) if sup else "$a_{%s}$" % base


def build_context(R):
    a = R["analytical60"]
    ctx = dict(R=R, a=a, n=n, c=c, mag=mag, ang=ang, pct=pct, sense=sense, abs=abs, math=math, len=len,
               vname=vname, aname=aname,
               p=R["params"], s1=R["sec1"], s2=R["sec2"], s3=R["sec3"], s4=R["sec4"], s6=R["sec6"],
               s7=R["sec7"], s8=R["sec8"], wm=R["wm"])
    # ---------------- comparison table at theta2 = 60 deg ----------------
    vm, am = R["sec2"]["measured"], R["sec3"]["measured"]
    va, aa = R["sec2"]["angular"], R["sec3"]["angular"]
    icm = R["sec4"]["measured"]
    sim = R["sec7"]["summary"]["at60"]
    rows = []
    def row(name, unit, g, ic, an, cp):
        rows.append(dict(name=name, unit=unit, g=g, ic=ic, an=an, cp=cp,
                         dg=(pct(abs(g), abs(an)) if g is not None else None),
                         dic=(pct(abs(ic), abs(an)) if ic is not None else None),
                         dcp=(pct(abs(cp), abs(an)) if cp is not None else None)))
    row("$V_A$", "m/s", vm["V_A"]["value"], icm["VA"], mag(a["VA"]), sim["VA"])
    row("$V_B$", "m/s", vm["V_B"]["value"], icm["VB"], mag(a["VB"]), sim["VB"])
    row("$V_C$", "m/s", vm["V_C"]["value"], icm["VC"], mag(a["VC"]), sim["VC"])
    row("$V_D$", "m/s", vm["V_D"]["value"], icm["VD"], mag(a["VD"]), abs(sim["VD"]))
    row("$\\omega_3$", "rad/s", abs(va["omega3"]), icm["omega3"], abs(a["omega3"]), abs(sim["w3"]))
    row("$\\omega_4$", "rad/s", abs(va["omega4"]), icm["omega4"], abs(a["omega4"]), abs(sim["w4"]))
    row("$\\omega_5$", "rad/s", abs(va["omega5"]), icm["omega5"], abs(a["omega5"]), abs(sim["w5"]))
    row("$a_B$", "m/s²", am["a_B"]["value"], None, mag(a["aB"]), sim["aB"])
    row("$a_C$", "m/s²", am["a_C"]["value"], None, mag(a["aC"]), sim["aC"])
    row("$a_D$", "m/s²", am["a_D"]["value"], None, mag(a["aD"]), abs(sim["aD"]))
    row("$\\alpha_3$", "rad/s²", abs(aa["alpha3"]), None, abs(a["alpha3"]), abs(sim["al3"]))
    row("$\\alpha_4$", "rad/s²", abs(aa["alpha4"]), None, abs(a["alpha4"]), abs(sim["al4"]))
    row("$\\alpha_5$", "rad/s²", abs(aa["alpha5"]), None, abs(a["alpha5"]), abs(sim["al5"]))
    row("$a_{G2}$", "m/s²", am["a_G2"]["value"], None, mag(a["aG2"]), sim["aG2"])
    row("$a_{G3}$", "m/s²", am["a_G3"]["value"], None, mag(a["aG3"]), sim["aG3"])
    row("$a_{G4}$", "m/s²", am["a_G4"]["value"], None, mag(a["aG4"]), sim["aG4"])
    row("$a_{G5}$", "m/s²", am["a_G5"]["value"], None, mag(a["aG5"]), sim["aG5"])
    ctx["cmp"] = rows
    ctx["max_dg_vel"] = max(abs(r["dg"]) for r in rows[:7])
    ctx["max_dg_acc"] = max(abs(r["dg"]) for r in rows[7:])
    ctx["max_dic"] = max(abs(r["dic"]) for r in rows if r["dic"] is not None)
    g = 9.81
    ratios = [mag(a[k]) / g for k in ("aG2", "aG3", "aG4", "aG5", "aG6")]
    ctx["gmin"], ctx["gmax"] = min(ratios), max(ratios)
    ctx["g3s"] = R["g3_sensitivity"]
    # table keys must not contain '|' (it would split the markdown table)
    R["sec7"]["compare"] = {k.replace("|", "").replace(" (N)", " magnitude (N)") if k.startswith("|") else k: v
                            for k, v in R["sec7"]["compare"].items()}
    # forces with magnitudes/directions
    F0 = R["sec6"]["0.0"]["F"]; F3 = R["sec6"]["0.3"]["F"]
    pins = []
    for key, joint in [("F12", "O (ground–crank)"), ("F32", "A (crank–coupler)"), ("F43", "B (coupler–rocker)"),
                       ("F14", "E (ground–rocker)"), ("F53", "C (coupler–rod)"), ("F65", "D (rod–slider)")]:
        z0 = complex(F0[key + "x"], F0[key + "y"]); z3 = complex(F3[key + "x"], F3[key + "y"])
        pins.append(dict(key=key, joint=joint, x=z0.real, y=z0.imag, m=abs(z0), ang=math.degrees(cmath.phase(z0)) % 360,
                         m3=abs(z3)))
    ctx["pins"] = pins
    ctx["F0"], ctx["F3"] = F0, F3
    # time ratio of slider
    sl = sorted(R["sec1"]["limits"]["slider6"], key=lambda e: e["S"])
    th_L, th_R = sl[0]["theta2"], sl[1]["theta2"]
    ret = (th_L - th_R) % 360         # D moves from D_R to D_L (leftwards)
    fwd = 360 - ret
    ctx.update(th_DL=th_L, th_DR=th_R, S_min=sl[0]["S"], S_max=sl[1]["S"], stroke=sl[1]["S"] - sl[0]["S"],
               ang_ret=ret, ang_fwd=fwd, Q=fwd / ret)
    rk = sorted(R["sec1"]["limits"]["rocker4"], key=lambda e: e["theta4"])
    ctx.update(rk_min=rk[0], rk_max=rk[1], rk_swing=rk[1]["theta4"] - rk[0]["theta4"],
               rk_ang1=(rk[1]["theta2"] - rk[0]["theta2"]) % 360)
    ctx["rk_ang2"] = 360 - ctx["rk_ang1"]
    # matrix (numeric) rows for display
    A = R["sec6"]["0.0"]["matrix"]; rhs = R["sec6"]["0.0"]["rhs"]
    ctx["Amat"], ctx["rhs"] = A, rhs
    ctx["unknowns"] = ["F12x", "F12y", "F32x", "F32y", "F43x", "F43y", "F53x", "F53y", "F14x", "F14y", "F65x", "F65y",
                       "F16y", "T12"]
    # balancing
    st = R["sec8"]["stats"]
    ctx["bal_rows"] = [
        ("Unbalanced (as designed)", st["unbalanced"], None),
        ("Crank counterweight only (least squares)", st["crank_only"], R["sec8"]["designs"]["crank_only"]),
        ("Berkof–Lowen on links 2 & 4 (exact 4-bar balance)", st["berkof_lowen"], R["sec8"]["designs"]["berkof_lowen"]),
        ("Berkof–Lowen + extra crank weight (least squares)", st["bl_plus_crank"], R["sec8"]["designs"]["bl_plus_crank"]),
        ("Global least-squares optimum on links 2 & 4", st["optimal_lsq"], R["sec8"]["designs"]["optimal_lsq"]),
    ]
    return ctx


def fmt_matrix_cell(v):
    if abs(v) < 1e-12:
        return "0"
    if abs(v - round(v)) < 1e-12:
        return n(v, 0)
    return n(v, 4)


def render_markdown(ctx, figext, name="template.md.j2"):
    env = jinja2.Environment(loader=jinja2.FileSystemLoader(HERE), variable_start_string="<<",
                             variable_end_string=">>", block_start_string="<%", block_end_string="%>",
                             comment_start_string="<#", comment_end_string="#>", trim_blocks=True, lstrip_blocks=True,
                             undefined=jinja2.StrictUndefined)
    env.globals.update(fm=fmt_matrix_cell)
    tpl = env.get_template(name)
    return tpl.render(figext=figext, **ctx)


def make_toc(html, pages=None):
    """Build the table of contents from the <h1>/<h2> ids of the pandoc HTML."""
    items = []
    for m in re.finditer(r'<h([12]) id="([^"]+)"[^>]*>(.*?)</h\1>', html, flags=re.S):
        lvl, hid, inner = int(m.group(1)), m.group(2), m.group(3)
        text = re.sub(r"<[^>]+>", "", inner).strip()
        if hid in ("toc",):
            continue
        items.append((lvl, hid, text))
    li = []
    for lvl, hid, text in items:
        pg = str(pages.get(hid, "")) if pages else ""
        li.append(f'<li class="lvl{lvl}"><a class="t" href="#{hid}">{text}</a><span class="dots"></span>'
                  f'<span class="pg">{pg}</span></li>')
    return "<ol>" + "".join(li) + "</ol>", items


def _norm(t):
    return re.sub(r"[^a-z0-9]", "", t.lower())


def find_pages(pdf_path, items, start_page=2):
    from pypdf import PdfReader
    reader = PdfReader(pdf_path)
    texts = [_norm(pg.extract_text() or "") for pg in reader.pages]
    pages, cursor = {}, start_page
    for lvl, hid, text in items:
        key = _norm(text)[:28]
        for i in range(cursor, len(texts)):
            if key and key in texts[i]:
                pages[hid] = i + 1
                cursor = i
                break
    return pages


def print_pdf(html_path, pdf_path, footer="MIET1077 · Six-bar linkage · worked practice solution (example parameters)"):
    env = dict(os.environ)
    env["NODE_PATH"] = env.get("NODE_PATH", "/opt/node22/lib/node_modules")
    subprocess.run(["node", os.path.join(HERE, "print_pdf.js"), html_path, pdf_path, footer], check=True, env=env)


def main():
    with open(os.path.join(OUT, "results.json")) as f:
        R = json.load(f)
    ctx = build_context(R)
    build = os.path.join(HERE, "build")
    os.makedirs(build, exist_ok=True)
    figdir = os.path.join(build, "figures")
    if os.path.exists(figdir):
        shutil.rmtree(figdir)
    shutil.copytree(os.path.join(OUT, "figures"), figdir)
    katex_src = KATEX_DIR or os.path.join(HERE, "vendor", "katex")
    katex_dst = os.path.join(build, "katex")
    if not os.path.exists(katex_dst):
        shutil.copytree(katex_src, katex_dst)
    md_html = render_markdown(dict(ctx, target="html"), "svg")
    md_docx = render_markdown(dict(ctx, target="docx"), "png")
    with open(os.path.join(build, "report_html.md"), "w") as f:
        f.write(md_html)
    with open(os.path.join(build, "report_docx.md"), "w") as f:
        f.write(md_docx)
    common = ["--from", "markdown+pipe_tables+tex_math_dollars+fenced_divs+link_attributes+implicit_figures+raw_html+raw_attribute",
              "--resource-path", build]
    html_path = os.path.join(build, "report.html")
    subprocess.run(["pandoc", os.path.join(build, "report_html.md"), *common, "--standalone", "--katex=katex/",
                    "--css", "style.css", "--metadata", "pagetitle=MIET1077 Six-Bar Linkage – Worked Solution",
                    "-o", html_path], check=True)
    shutil.copy(os.path.join(HERE, "style.css"), os.path.join(build, "style.css"))
    with open(html_path) as f:
        html = f.read()
    pdf_path = os.path.join(HERE, PDF_NAME)
    # pass 1: TOC without page numbers -> find the page of every heading
    toc, items = make_toc(html)
    with open(html_path, "w") as f:
        f.write(html.replace("<!--TOC-->", toc))
    print_pdf(html_path, pdf_path)
    pages = find_pages(pdf_path, items)
    # pass 2: TOC with page numbers (same length, so pagination is unchanged)
    toc, _ = make_toc(html, pages)
    with open(html_path, "w") as f:
        f.write(html.replace("<!--TOC-->", toc))
    print_pdf(html_path, pdf_path)
    missing = [hid for _, hid, _ in items if hid not in pages]
    if missing:
        print("WARNING: TOC page not found for", missing)
    # companion guide for the in-person assessment
    guide_md = os.path.join(build, "guide.md")
    with open(guide_md, "w") as f:
        f.write(render_markdown(dict(ctx, target="html"), "svg", name="guide.md.j2"))
    guide_html = os.path.join(build, "guide.html")
    subprocess.run(["pandoc", guide_md, *common, "--standalone", "--katex=katex/", "--css", "style.css",
                    "--metadata", "pagetitle=MIET1077 Six-Bar – In-person Workflow Guide", "-o", guide_html], check=True)
    print_pdf(guide_html, os.path.join(HERE, GUIDE_PDF_NAME), "MIET1077 · six-bar linkage · in-person workflow guide")
    subprocess.run(["pandoc", os.path.join(build, "report_docx.md"), *common, "--toc", "--toc-depth=2",
                    "--metadata", "title=MIET1077 Major Assignment – Six-Bar Linkage (Worked Practice Solution)",
                    "-o", os.path.join(HERE, DOCX_NAME)], check=True)
    print("done:", pdf_path, os.path.join(HERE, DOCX_NAME))


if __name__ == "__main__":
    main()
