"""Fig. 1 of the manuscript: training-free protocol of the equal-depth GPU study."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUT = Path(__file__).resolve().parents[1] / "paper" / "figures" / "fig01_protocol.png"
INK, MUTED, BLUE, ORANGE = "#0b0b0b", "#52514e", "#2a78d6", "#eb6834"
boxes = [
    ("Train split\n8 restarts × (p, method)\nadjoint L-BFGS-B, c64", "#e8f1fb"),
    ("Angle templates\nnormalised angles,\nmedoid of best optima", "#e8f1fb"),
    ("Validation split\nselect α scaling and\nQAOA baseline", "#e8f1fb"),
    ("FROZEN_MODEL.json\n+ sha256\n(before any test call)", "#fdecc8"),
    ("Test split\nangles from D, s_J only;\nno objective calls", "#e6f5ee"),
    ("Paired analysis\nsame depth & equal\ntwo-qubit budget", "#efedf8"),
]
fig, ax = plt.subplots(figsize=(12, 2.6))
ax.set_xlim(0, 12); ax.set_ylim(0, 2.6); ax.axis("off")
w, h, y = 1.55, 1.3, 0.9
for i, (txt, col) in enumerate(boxes):
    x = 0.15 + i * 2.0
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.03,rounding_size=0.08",
                                fc=col, ec=MUTED, lw=1))
    ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=7.8, color=INK)
    if i < len(boxes) - 1:
        ax.annotate("", xy=(x + 2.0, y + h / 2), xytext=(x + w + 0.02, y + h / 2),
                    arrowprops=dict(arrowstyle="->", color=INK, lw=1.2))
ax.plot([7.92, 7.92], [0.55, 2.35], ls="--", color=ORANGE, lw=1.4)
ax.text(7.92, 0.38, "freeze boundary: test instances never choose templates, signs, coefficients or candidates",
        fontsize=8, color=ORANGE, ha="center")
ax.text(0.15, 2.42, "independent seeds per split; 10 graph families × 5 positive weight laws × n; 20 / 10 / 20 instances per cell",
        fontsize=8, color=MUTED)
fig.savefig(OUT, dpi=200, bbox_inches="tight", facecolor="white")
print(OUT)
