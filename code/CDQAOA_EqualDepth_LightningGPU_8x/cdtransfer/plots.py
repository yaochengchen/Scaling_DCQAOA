"""Minimal reproducible figures for pilot and manuscript scaffolding."""
from __future__ import annotations
from pathlib import Path
import numpy as np


def make_figures(out: Path, records: list[dict], test: list[dict]) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    images=[]
    figdir=out/"figures";figdir.mkdir(exist_ok=True,parents=True)
    if records:
        depth=sorted({r["p"] for r in records if r["cd"]})[0]
        chosen=[r for r in records if r["cd"] and r["p"]==depth]
        if chosen:
            fig,ax=plt.subplots(figsize=(7,4))
            a=np.array([x["angles"]["alpha"] for x in chosen])
            rms=np.array([x["features"]["rms"] for x in chosen])
            for j in range(a.shape[1]):
                ax.scatter(np.arange(len(chosen)),a[:,j]*rms,label=f"alpha[{j}]*RMS",s=14)
            ax.set(xlabel="offline restart index",ylabel="weight-normalized alpha",
                   title=f"Raw (un-aligned) optimized CD angles, p={depth}")
            ax.legend(fontsize=8)
            fig.tight_layout(); fig.savefig(figdir/"fig02_parameter_concentration.png",dpi=160);plt.close(fig)
            images.append("figures/fig02_parameter_concentration.png")
    if test:
        fig,ax=plt.subplots(figsize=(7,4))
        gains=[r["gain_equal_budget_norm"] for r in test]
        ax.axhline(0,color="black",linewidth=0.8)
        ax.plot(np.arange(len(gains)),gains,"o")
        ax.set(xlabel="held-out instance-depth pair",ylabel="(E_QAOA - E_CD) / sum|J|",
               title="Paired performance: fixed decomposition CX-matched")
        fig.tight_layout();fig.savefig(figdir/"fig05_equal_budget_paired.png",dpi=160);plt.close(fig)
        images.append("figures/fig05_equal_budget_paired.png")
        fig,ax=plt.subplots(figsize=(6,4))
        x=np.arange(len(test))
        ax.plot(x,[r["energy_qaoa_best_budget"]/r["norm_weights"] for r in test],"o",label="QAOA p'=3p")
        ax.plot(x,[r["energy_cd_proposed"]/r["norm_weights"] for r in test],"s",label="CD-QAOA p")
        ax.set(xlabel="held-out instance-depth pair",ylabel="Energy / sum|J|",
               title="Fixed-angle generalization")
        ax.legend()
        fig.tight_layout();fig.savefig(figdir/"fig04_generalization.png",dpi=160);plt.close(fig)
        images.append("figures/fig04_generalization.png")
    return images
