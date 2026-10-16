"""Small helpers for drawings that are TRUE TO SCALE on paper.

Every scaled figure is created with its exact physical size (mm) and an axes
that fills the figure, so one data unit = 1 mm on the printed page when the
figure is printed at 100 %.  Physical quantities are converted to paper mm
with the drawing scale (e.g. 1:10 for lengths, 1 m/s = 20 mm for velocities).
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Circle, Polygon, Arc

MM = 1 / 25.4
INK = "#1d1d1f"
GREY = "#8a8f98"
BLUE = "#1f5fbf"
RED = "#c0392b"
GREEN = "#1e8449"
ORANGE = "#d35400"
PURPLE = "#7d3c98"
TEAL = "#117a8b"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 7.5,
    "mathtext.fontset": "dejavusans",
    "svg.fonttype": "none",
})


def paper(w_mm, h_mm, x0=0.0, y0=0.0, frame=True):
    fig = plt.figure(figsize=(w_mm * MM, h_mm * MM))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(x0, x0 + w_mm)
    ax.set_ylim(y0, y0 + h_mm)
    ax.set_aspect("equal")
    ax.axis("off")
    if frame:
        ax.add_patch(plt.Rectangle((x0 + 0.3, y0 + 0.3), w_mm - 0.6, h_mm - 0.6, fill=False,
                                   lw=0.6, ec=GREY))
    return fig, ax


def P(z):
    return np.array([z.real, z.imag])


def arrow(ax, p, q, color=INK, lw=1.2, ms=7, zorder=5, style="-|>", ls="-"):
    a = FancyArrowPatch(P(p), P(q), arrowstyle=style, mutation_scale=ms, color=color, lw=lw,
                        shrinkA=0, shrinkB=0, zorder=zorder, linestyle=ls)
    ax.add_patch(a)
    return a


def label(ax, z, text, dx=0.0, dy=0.0, color=INK, size=7.5, ha="center", va="center", weight="normal",
          bbox=False, zorder=8, rotation=0):
    kw = {}
    if bbox:
        kw["bbox"] = dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85)
    ax.text(z.real + dx, z.imag + dy, text, color=color, fontsize=size, ha=ha, va=va,
            fontweight=weight, zorder=zorder, rotation=rotation, **kw)


def guide(ax, p, d, t0=-1.0, t1=1.0, color=GREY, lw=0.6, ls=(0, (4, 3)), zorder=1):
    """Construction line through point p with direction d (complex), parameter range in mm."""
    d = d / abs(d)
    a, b = p + t0 * d, p + t1 * d
    ax.plot([a.real, b.real], [a.imag, b.imag], color=color, lw=lw, ls=ls, zorder=zorder)
    return a, b


def along(p, u, q):
    """Signed distance from p to q measured along the unit vector u."""
    return ((q - p) * np.conj(u)).real


def cline(ax, p, u, q, before=8.0, after=13.0, **kw):
    """Construction line through the known point p in direction +-u (unit vector) that
    passes through the intersection point q: it starts `before` mm behind p and ends
    `after` mm beyond q.  Returns (t_start, t_end), measured along u from p."""
    t = along(p, u, q)
    s = 1.0 if t >= 0 else -1.0
    t_start, t_end = -s * before, t + s * after
    guide(ax, p, u, min(t_start, t_end), max(t_start, t_end), **kw)
    return t_start, t_end


def end_tag(ax, p, u, ts, text, back=2.0, at="end", **kw):
    """Label a construction line near its free end beyond the intersection (at="end")
    or near its start behind the known point (at="start")."""
    t_start, t_end = ts
    s = 1.0 if t_end >= t_start else -1.0
    t = t_end - s * back if at == "end" else t_start + s * back
    label(ax, p + t * u, text, **kw)


def intersect(p1, d1, p2, d2):
    M = np.array([[d1.real, -d2.real], [d1.imag, -d2.imag]])
    t = np.linalg.solve(M, np.array([(p2 - p1).real, (p2 - p1).imag]))
    return p1 + t[0] * d1


def pin(ax, z, r=1.1, color=INK, fc="white", zorder=7):
    ax.add_patch(Circle(P(z), r, fc=fc, ec=color, lw=0.9, zorder=zorder))


def ground(ax, z, w=7.0, h=2.2, angle=0.0, color=INK, zorder=3, tri=True):
    """Fixed-pivot symbol (triangle + hatching) below point z (paper mm)."""
    rot = np.exp(1j * np.radians(angle))
    if tri:
        tri_pts = [z, z + rot * (-w * 0.35 - 1j * h * 1.3), z + rot * (w * 0.35 - 1j * h * 1.3)]
        ax.add_patch(Polygon([P(t) for t in tri_pts], closed=True, fc="white", ec=color, lw=0.7, zorder=zorder))
        base = z - rot * 1j * h * 1.3
    else:
        base = z
    a, b = base - rot * w / 2, base + rot * w / 2
    ax.plot([a.real, b.real], [a.imag, b.imag], color=color, lw=0.9, zorder=zorder)
    for k in range(6):
        s = a + (b - a) * (k + 0.5) / 6
        e = s + rot * (-1.2 - 1.4j)
        ax.plot([s.real, e.real], [s.imag, e.imag], color=color, lw=0.5, zorder=zorder)


def angle_arc(ax, centre, r, a0_deg, a1_deg, color=INK, lw=0.6, text=None, text_r=None, size=7, arrow_end=False):
    lo, hi = (a0_deg, a1_deg) if a1_deg >= a0_deg else (a1_deg, a0_deg)
    ax.add_patch(Arc(P(centre), 2 * r, 2 * r, angle=0, theta1=lo, theta2=hi, color=color, lw=lw, zorder=4))
    if text:
        mid = np.radians((a0_deg + a1_deg) / 2)
        tr = text_r if text_r else r + 3.0
        ax.text(centre.real + tr * np.cos(mid), centre.imag + tr * np.sin(mid), text, fontsize=size,
                ha="center", va="center", color=color, zorder=8)


def scale_bar(ax, z, length_mm, text, color=INK):
    ax.plot([z.real, z.real + length_mm], [z.imag, z.imag], color=color, lw=1.0)
    for k in range(0, int(round(length_mm)) + 1, int(max(1, round(length_mm / 5)))):
        ax.plot([z.real + k, z.real + k], [z.imag - 0.8, z.imag + 0.8], color=color, lw=0.6)
    ax.text(z.real + length_mm / 2, z.imag + 2.0, text, ha="center", va="bottom", fontsize=7, color=color)


def label_on(ax, p, q, t, text, side=1.0, dist=3.2, **kw):
    """Label placed at fraction t along segment p->q, offset perpendicular to it.
    side=+1 puts the text on the left of the direction p->q, -1 on the right."""
    d = (q - p) / abs(q - p)
    z = p + t * (q - p) + side * dist * 1j * d
    label(ax, z, text, **kw)


def save(fig, path_no_ext):
    fig.savefig(path_no_ext + ".svg")
    fig.savefig(path_no_ext + ".png", dpi=300)
    plt.close(fig)


def round_to(x, step):
    return np.round(np.asarray(x) / step) * step
