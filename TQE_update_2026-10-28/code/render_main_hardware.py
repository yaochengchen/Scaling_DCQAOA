from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
DEST=ROOT/'reproduced/figures';DEST.mkdir(parents=True,exist_ok=True)
d=pd.read_csv(ROOT/'hardware/tqe_hw_results.csv')
assert len(d)==84 and d.shots.eq(8000).all()
plt.rcParams.update({'font.size':8,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
fig,axs=plt.subplots(1,2,figsize=(7.25,2.8),layout='constrained')
colors={'u1':{1:'#6BA6E5',2:'#2477D1'},'qaoa':{1:'#F1AE87',2:'#E87837'}}
for p,ax in zip([1,2],axs[:2]):
    for m,nam in [('u1','Power-law DC-QAOA'),('qaoa','Fixed-angle QAOA')]:
        g=d[d.method==f'{m}_p{p}'].groupby('n').mean(numeric_only=True)
        c=colors[m][2]
        ax.plot(g.index,100*g.p_opt_hw,'o-',color=c,ms=3,lw=1.2,label=nam+' | hardware')
        ax.plot(g.index,100*g.p_opt_ideal,'^--',color=c,ms=2.5,lw=1,alpha=.8,label=nam+' | ideal')
    ax.set_yscale('log'); ax.set_ylim(.015,70)
    ax.set_title(f'({"ab"[p-1]}) $p={p}$',loc='left',fontsize=9)
    ax.set_xlabel('Qubits, $n$');ax.grid(axis='y',alpha=.2)
    ax.set_xticks(sorted(d.n.unique()))
axs[0].set_ylabel('Mean optimal-solution\nprobability (%)')
handles, labels = axs[0].get_legend_handles_labels()
fig.legend(handles, labels, fontsize=6.1, handlelength=2.6, loc='lower center', bbox_to_anchor=(.5,-.095), ncol=2, frameon=False)
for ext in ['pdf','png']: fig.savefig(DEST/f'fig04_hardware_probability.{ext}',dpi=300,bbox_inches='tight')
plt.close(fig)