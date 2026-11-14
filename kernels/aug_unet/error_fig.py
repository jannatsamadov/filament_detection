"""Where the validation PQ goes: one figure from error_analysis.json, candidate_pool.json, keep_curve.json."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "kernels" / "aug_post" / "out"
INK, INK2, GRID, BAR, KEY = "#0b0b0b", "#52514e", "#e4e3df", "#2a78d6", "#eb6834"
PAPER = "#fcfcfb"
GREY1, GREY2, GREY3 = "#8d8c87", "#b5b4af", "#d5d4cf"

e = json.load(open(OUT / "error_analysis.json"))
pool = json.load(open(OUT / "candidate_pool.json"))
curve = json.load(open(OUT / "keep_curve.json"))
g, f = e["gt"], e["fp"]
tp = e["overall"]["TP"]

rows = [
    ("Annotasiyadakı filamentlər\n(1849)", [
        ("uyğun gəldi", g["matched"], BAR),
        ("forma səhvi", g["prediction too large"] + g["prediction too small"] + g["fragmented"], KEY),
        ("zəif\n(p < 0.65)", g["faint (seen below threshold)"], GREY1),
        ("görmür", g["not seen"], GREY2)]),
    ("Proqnozlarımız\n(1956)", [
        ("uyğun gəldi", tp, BAR),
        ("forma səhvi", f["overlaps this annotator's filament, IoU 0.2-0.5"], KEY),
        ("başqa\nannotator", f["another annotator drew it"], GREY1),
        ("heç kim\nçəkməyib", f["no annotator of this frame drew it"] + f["single-annotator frame, not drawn"], GREY2)]),
]
ladder = [
    ("Təlimsiz tünd piksel (\"0.46\" notebook)", e["dark_pixel_notebook"]["PQ"], GREY2),
    ("Bir annotator digərinə qarşı", e["reach"]["one_other_annotator"]["PQ"], GREY2),
    ("5 modelin ansamblı", e["overall"]["PQ"], BAR),
    ("+ keep-rule, 17 əlamət", curve["curve"]["all17_100"]["PQ"], BAR),
    ("Oracle: bir parametrdə seçim", pool["oracle_one_setting"]["PQ"], GREY1),
    ("Oracle: 32 parametrdən kontur", pool["oracle_pool"]["PQ"], GREY1),
]

fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 7.4), facecolor=PAPER, gridspec_kw=dict(height_ratios=[1, 1.25], hspace=0.42))
for y, (name, parts) in enumerate(reversed(rows)):
    left = 0
    total = sum(p[1] for p in parts)
    for label, n, color in parts:
        a1.barh(y, n, left=left, color=color, edgecolor=PAPER, linewidth=2, height=0.62)
        dark = color in (BAR, KEY, GREY1)
        a1.text(left + n / 2, y, f"{n}\n{100 * n / total:.0f}%", ha="center", va="center", fontsize=9,
                color="white" if dark else INK)
        a1.text(left + n / 2, y + 0.35, label, ha="center", va="bottom", fontsize=9, color=INK2, linespacing=1.0)
        left += n
a1.set_yticks(range(len(rows)), [r[0] for r in reversed(rows)], fontsize=10, color=INK)
a1.set_xlim(0, 2000)
a1.set_ylim(-0.45, 1.85)
a1.set_xticks([])
for s in a1.spines.values():
    s.set_visible(False)
a1.tick_params(left=False)
a1.set_title("Eyni problem iki tərəfdən: filamenti tapırıq, amma forma IoU 0.5-i keçmir", loc="left", fontsize=12, color=INK, fontweight="bold")

for y, (name, v, color) in enumerate(reversed(ladder)):
    a2.barh(y, v, color=color, height=0.62)
    a2.text(v + 0.006, y, f"{v:.3f}", va="center", fontsize=10, color=INK)
a2.set_yticks(range(len(ladder)), [r[0] for r in reversed(ladder)], fontsize=10, color=INK)
a2.set_xlim(0, 0.65)
a2.xaxis.grid(True, color=GRID)
a2.set_axisbelow(True)
a2.tick_params(left=False, axis="x", colors=INK2)
for s in a2.spines.values():
    s.set_visible(False)
a2.set_xlabel("Validasiya PQ (141 foto, öz skorerimiz; oracle cavabları görür, yalnız tavandır)", color=INK2, fontsize=9)
a2.set_title("Oracle boşluğu böyükdür, amma öyrədilmiş seçici çox az hissəsini tutur", loc="left", fontsize=12, color=INK, fontweight="bold")
fig.savefig(ROOT / "figures" / "error_breakdown.png", dpi=120, bbox_inches="tight", facecolor=PAPER)
