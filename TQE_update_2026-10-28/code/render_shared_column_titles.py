"""Prepare U1 figure artwork from frozen data; no simulations or refits."""
from pathlib import Path
import json, hashlib, shutil, subprocess
import numpy as np
import pandas as pd
import study
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

S=Path(__file__).resolve().parents[1]
OUT=S/'figures/u1_revision_1452s1'
AUDIT=S/'editing_audit/u1_figures_1452s1'
OUT.mkdir(exist_ok=True); AUDIT.mkdir(parents=True,exist_ok=True)
model=json.loads((S/'models/U1_power_law.json').read_text())
oldmodel=json.loads((S/'models/A_joint_candidate.json').read_text())
assert model['templates']==oldmodel['templates']
D0=model['reference_degree']
# F2/S1: same training angles, new per-layer degree normalization.
df=study.load_training(); ft=study.figtools
stats={}; low=[]; plotted=[]
for p,name in [(1,'fig02_powerlaw_scaling_p1'),(4,'figS1_powerlaw_scaling_p4')]:
    d=ft.layer_values(df,p,1,D0)
    coeff=model['templates'][str(p)]['layers'][0]
    for key,prefix in [('gamma','g'),('alpha','a')]:
        A,b=coeff[key]['A'],coeff[key]['b']
        d[prefix+'_new']=d[prefix+'_raw']*(d.D/D0)**(-b)
        d[prefix+'_prediction']=A*(d.D/D0)**b
        assert np.allclose(d[prefix+'_prediction']*(d.D/D0)**(-b),A,atol=1e-14,rtol=0)
    cols=['id','p','family','D','s','g_raw','g_new','g_prediction','a_raw','a_new','a_prediction']
    low.append(d[d.D<=1][cols]); d=d[d.D>1].copy(); plotted.append(d[cols])
    assert len(d)==6988
    stats[str(p)]={'records':len(d),'low_degree_excluded':12,'coefficients':coeff,'panels':{}}
    D=np.linspace(d.D.min(),d.D.max(),400)
    g=coeff['gamma'];a=coeff['alpha']
    curves=[g['A']*(D/D0)**g['b'],np.full_like(D,g['A']),a['A']*(D/D0)**a['b'],np.full_like(D,a['A'])]
    labs=[r'$\gamma_1s_J$',r'$\gamma_1s_J(D/D_0)^{-b_{\gamma,1,p}}$',r'$\alpha_1s_J$',r'$\alpha_1s_J(D/D_0)^{-b_{\alpha,1,p}}$']
    fig,axes=plt.subplots(2,2,figsize=(8.4,6.2),sharex=True)
    for ax,col,lab,curve in zip(axes.flat,['g_raw','g_new','a_raw','a_new'],labs,curves):
        for j,fam in enumerate(ft.FAMILIES):
            z=d[d.family==fam];ax.scatter(z.D,z[col],s=5,alpha=.25,color=plt.get_cmap('tab10')(j),linewidths=0)
        ft._binned(ax,d.D,d[col])
        ax.plot(D,curve,'--',color='crimson',lw=1.4,zorder=6)
        lo,hi=np.quantile(d[col],[.02,.98]);pad=.25*(hi-lo)
        ax.set(ylim=(lo-pad,hi+pad),ylabel=lab)
        outside=int(((d[col]<lo-pad)|(d[col]>hi+pad)).sum());spread=float(ft._spread(d[col]))
        ax.text(.02,.04,f'IQR/|median| = {spread:.2f}',transform=ax.transAxes,fontsize=8)
        ax.text(.98,.12,f'{outside} of {len(d):,} outside view',ha='right',transform=ax.transAxes,fontsize=7,color='.4')
        xs,med,q25,q75,n=ft.bin_stats(d.D,d[col])
        pd.DataFrame({'D_median':xs,'angle_median':med,'q25':q25,'q75':q75,'count':n}).to_csv(OUT/f'{name}_{col}_bins.csv',index=False)
        stats[str(p)]['panels'][col]={'spread':spread,'median':float(d[col].median()),'outside':outside,'bins':len(xs),'records_in_bins':int(n.sum()),'ylim':[lo-pad,hi+pad]}
    # Shared column headings sit above the individual panel labels.
    axes[0,0].set_title('Weight scale removed', fontsize=10, loc='center', pad=23)
    axes[0,1].set_title('Weight and degree scales removed', fontsize=10, loc='center', pad=23)
    for ax, label in zip(axes.flat, ['(a)', '(b)', '(c)', '(d)']):
        ax.text(0, 1.015, label, transform=ax.transAxes,
                ha='left', va='bottom', fontsize=10, clip_on=False)
    for ax in axes[1]:ax.set_xlabel(r'Mean degree $D$')
    handles=[Line2D([],[],marker='o',ls='',color=plt.get_cmap('tab10')(j),label=f,markersize=4) for j,f in enumerate(ft.FAMILIES)]
    handles += [Line2D([],[],marker='o',color='k',lw=1,label='Degree-bin median / IQR',markersize=4),Line2D([],[],ls='--',color='crimson',label='Formula prediction')]
    fig.legend(handles=handles,loc='lower center',ncol=4,fontsize=7,frameon=False)
    fig.suptitle(f'DC-QAOA angle scaling: formula predictions and offline-optimized angles\n$p={p}$, layer 1; 6,988 training graphs with $D>1$',fontsize=10)
    fig.tight_layout(rect=(0,.1,1,.92))
    for ext in ('pdf','png'):fig.savefig(OUT/f'{name}.{ext}',dpi=300)
    plt.close(fig)
pd.concat(low).to_csv(OUT/'low_degree_training_diagnostic.csv',index=False)
pd.concat(plotted).to_csv(OUT/'plotted_training_coordinates.csv',index=False)
(OUT/'diagnostic_statistics.json').write_text(json.dumps(stats,indent=2),encoding='utf-8')
print(json.dumps(stats,indent=2))
