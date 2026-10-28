"""Isolated S2/S3 review assets; never modifies the active manuscript."""
from pathlib import Path
import json, hashlib, shutil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'editing_audit/side_S2_S3_2026-10-27'
OUT.mkdir(parents=True, exist_ok=True)
archive = OUT / 'archive'
archive.mkdir(exist_ok=True)
data = pd.read_csv(ROOT / 'inputs/original_effective_exponents.csv')
model = json.loads((ROOT / 'models/U1_power_law.json').read_text())
fig, axes = plt.subplots(2, 1, figsize=(6.5, 5.5), sharex=True)
for i, (ax, key, title, color) in enumerate(zip(axes, ['alpha', 'gamma'],
        [r'CD angle $\alpha_1 s_J$', r'Cost angle $\gamma_1 s_J$'], ['#d95f02', '#1f77b4'])):
    rows = data[(data.layer == 1) & (data.quantity == key + '_l*s_J')].sort_values('p')
    x = rows.p.to_numpy(); y = rows.fitted_exponent.to_numpy()
    expected = [model['templates'][str(p)]['layers'][0][key]['b'] for p in x]
    assert np.allclose(y, expected, atol=1e-12, rtol=0)
    ax.errorbar(x, y, yerr=[y-rows.ci_lo.to_numpy(), rows.ci_hi.to_numpy()-y],
                fmt='o-', color=color, capsize=4, lw=1.8, ms=5,
                label='Fitted exponent (95% bootstrap CI)')
    ax.set_title(title, loc='left', fontsize=10)
    ax.set_ylabel('Degree exponent'); ax.grid(axis='y', color='.88', lw=.7)
    if i == 0: ax.legend(frameon=False, fontsize=8, loc='lower right')
axes[-1].set_xticks([1,2,3,4,6]); axes[-1].set_xlabel(r'Circuit depth $p$ (layer 1)')
fig.suptitle('First-layer angle scaling with mean degree', fontsize=11)
fig.text(.5, .005, 'p = 1–4: 6,988 graphs each; p = 6: 4,988 graphs', ha='center', fontsize=8, color='.35')
fig.tight_layout(rect=(0,.035,1,.96))
for ext in ['png','pdf']: fig.savefig(OUT / ('FigureS2_U1.'+ext), dpi=220)
plt.close(fig)
