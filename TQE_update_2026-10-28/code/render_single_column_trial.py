from pathlib import Path
import json,csv,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parents[1];OUT=R/'figures/vertical_single_column_trial';OUT.mkdir(exist_ok=True)
plt.rcParams.update({'font.size':8,'axes.titlesize':8,'axes.labelsize':8,'xtick.labelsize':7,'ytick.labelsize':7,'legend.fontsize':6.8,'pdf.fonttype':42})
B='#2476cf';O='#e87837';G='#1baf7a'
def load(p):return json.loads((R/p).read_text(encoding='utf-8'))
def style(ax):ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',color='.9',lw=.5);ax.set_axisbelow(True)
def save(fig,n):
 for ext in ['pdf','png']:fig.savefig(OUT/f'fig{n:02d}_single_column.{ext}',dpi=300,facecolor='white')
 plt.close(fig)
S=load('results/pennylane_probability_1050/summary.json');SZ=load('figures/figure3_depth_average_probability/statistics.json')['size_statistics']
fig,axes=plt.subplots(2,1,figsize=(3.5,4.65),layout='constrained')
for ax,x,entries,ttl,xlabel in [(axes[0],[1,2,3,4],[{'mean':[S['depth'][str(p)]['mean'][k] for k in [0,2]],'ci95':[S['depth'][str(p)]['ci95'][k] for k in [0,2]]} for p in [1,2,3,4]],'(a) By depth: pooled over $n=10$–$22$','Layers, $p$'),(axes[1],[10,12,14,16,18,20,22],[SZ[str(n)] for n in [10,12,14,16,18,20,22]],'(b) By size: averaged over $p=1$–$4$','Qubits, $n$')]:
 for k,lab,col,mark in [(1,'Fixed-angle QAOA',O,'s'),(0,'Power-law DC-QAOA',B,'o')]:
  y=np.array([v['mean'][k] for v in entries])*100;ci=np.array([v['ci95'][k] for v in entries])*100
  ax.errorbar(x,y,yerr=[y-ci[:,0],ci[:,1]-y],fmt=mark,ls='-' if ax==axes[0] else 'none',color=col,label=lab,ms=3.5,lw=1.2,capsize=2)
 ax.set(title=ttl,xlabel=xlabel,ylabel='Optimal-solution probability (%)',xticks=x,ylim=(0,30));style(ax)
axes[0].legend(frameon=False);save(fig,3)
# Figure 4: same archived trajectories and hit records.
Q=load('results/benchmark_new_summary.json');colors={'L-BFGS-B':'#6750a4','COBYLA':'#21867a','Adam':'#cf6b43'}
fig,(a,b)=plt.subplots(2,1,figsize=(3.5,4.65),layout='constrained');x=np.arange(1,501)
a.axhline(Q['L-BFGS-B']['mean_fixed_ratio'],color='#dd9b12',lw=1.5,label='Formula-based DC-QAOA')
for m,col in colors.items():
 v=Q[m];assert v['runs']==240 and v['budget_evaluations']==500
 a.fill_between(x,v['p10_best_curve'],v['p90_best_curve'],color=col,alpha=.13,lw=0);a.plot(x,v['mean_best_curve'],color=col,label=m,lw=1.2)
 xx=np.arange(501);hits=v['run_hits'];y=np.array([sum(h is not None and h<=i for h in hits) for i in xx])/240*100
 b.step(xx,y,where='post',color=col,lw=1.2,label=f'{m}: {sum(h is not None for h in hits)}/240')
a.set(title='(a) Mean best-so-far',ylabel='Mean MaxCut approximation ratio',ylim=(.45,.92));b.set(title='(b) Runs attaining own-graph target',ylabel='Runs attaining target (%)',ylim=(0,5))
for ax in [a,b]:ax.set(xlabel='Objective evaluations',xlim=(0,500));style(ax);ax.legend(frameon=False,loc='lower right' if ax==a else 'upper left')
save(fig,4)
# Figure 5: best-so-far endpoint, not returned angle probability.
runs=[json.loads(p.read_text(encoding='utf-8')) for p in sorted((R/'results/figure5_ajoint_c128_cpu/runs').glob('*.json'))];assert len(runs)==12
pad=lambda c:np.array((list(c)+[c[-1]]*60)[:60])
F=np.array([pad(r['chains']['formula']['best_so_far_ratio']) for r in runs]);rnd=[c for r in runs for k,c in r['chains'].items() if k.startswith('random')];assert len(rnd)==240
V=np.array([pad(c['best_so_far_ratio']) for c in rnd]);fixed=np.mean([r['fixed_ratio'] for r in runs]);refined=np.mean([r['chains']['formula']['final_ratio'] for r in runs])
fig,(a,b)=plt.subplots(2,1,figsize=(3.5,5.45),gridspec_kw={'height_ratios':[1,1.15]},layout='constrained');x=np.arange(1,61)
a.fill_between(x,F.min(0),F.max(0),color='tab:orange',alpha=.18,lw=0);a.plot(x,F.mean(0),color='tab:orange',lw=1.4,label='Formula start (12 graphs)')
a.fill_between(x,np.percentile(V,10,axis=0),np.percentile(V,90,axis=0),color='tab:blue',alpha=.15,lw=0);a.plot(x,V.mean(0),color='tab:blue',lw=1.4,label='Random start (240 runs)')
a.axhline(fixed,color='tab:orange',ls=':',lw=1);a.axhline(refined,color='k',ls='--',lw=.8)
a.text(59,refined+.007,f'Formula refined {refined:.4f}',ha='right',va='bottom',fontsize=6.5);a.text(59,fixed-.007,f'Formula fixed {fixed:.4f}',ha='right',va='top',fontsize=6.5,color='tab:orange')
a.set(title='(a) $n=12$, $p=2$, double precision, adjoint',xlabel='L-BFGS-B value-and-gradient calls',ylabel='Mean best-so-far approximation ratio',xlim=(1,60),ylim=(.45,.95));a.legend(frameon=False,loc='lower right');style(a)
order=np.argsort([r['fixed_ratio'] for r in runs])
for j,i in enumerate(order):
 r=runs[i];rr=[c['final_ratio'] for k,c in r['chains'].items() if k.startswith('random')];fr=r['chains']['formula']['final_ratio']
 b.scatter(rr,[j]*len(rr),s=7,color='tab:blue',alpha=.35,lw=0);b.plot([r['fixed_ratio'],fr],[j,j],color='tab:orange',lw=1);b.scatter(r['fixed_ratio'],j,s=15,color='tab:orange');b.scatter(fr,j,marker='D',s=13,color='k')
b.scatter([],[],s=15,color='tab:orange',label='Formula fixed');b.scatter([],[],s=13,marker='D',color='k',label='Formula refined');b.scatter([],[],s=7,color='tab:blue',label='Random-start best-so-far')
b.set(title='(b) Per graph',xlabel='Approximation ratio',xlim=(.45,.95));b.set_yticks(range(12),[runs[i]['id'].replace('test-n12-','') for i in order],fontsize=6.5);b.legend(frameon=False,loc='upper center',bbox_to_anchor=(.5,-.22),ncol=2,fontsize=6);style(b);save(fig,5)
# Figure 7 compact 3.5-inch single-column bar chart.
fig,ax=plt.subplots(figsize=(3.5,2.55),layout='constrained');weights=['bimodal','heavy_tail','lognormal','uniform','unit'];xs=np.arange(5)
for offset,p,col,hatch in [(-.18,1,B,''),(.18,4,G,'///')]:
 e=[S['weights'][f'{w}_p{p}'] for w in weights];y=np.array([v['delta_new_old_new_qaoa'][1] for v in e])*100;ci=np.array([v['delta_ci95'][1] for v in e])*100
 ax.bar(xs+offset,y,width=.34,color=col,edgecolor='.2',lw=.5,hatch=hatch,yerr=[y-ci[:,0],ci[:,1]-y],capsize=2,error_kw={'elinewidth':.8},label=f'$p={p}$')
ax.set_xticks(xs,['Bimodal','Heavy\ntail','Lognormal','Uniform','Unit']);ax.set(title='Same-layer gain by edge-weight distribution',ylabel='Mean probability difference\n(percentage points)',ylim=(0,12));ax.legend(frameon=False,ncol=2,loc='upper left');style(ax);save(fig,7)
manifest={'description':'Single-column vertical layout trial. No new quantum simulations. Original artwork preserved.','figure5_fixed':fixed,'figure5_refined':refined,'sources':{}}
for rel in ['results/pennylane_probability_1050/summary.json','figures/figure3_depth_average_probability/statistics.json','results/benchmark_new_summary.json','figures/figure6_depth_average_probability/statistics_and_negative_audit.json']:
 manifest['sources'][rel]=hashlib.sha256((R/rel).read_bytes()).hexdigest()
(OUT/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8');print(OUT)
