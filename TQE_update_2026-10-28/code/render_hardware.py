from pathlib import Path
import json,hashlib,sys
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parents[1];P=R/'hardware';O=R/'reproduced/hardware';O.mkdir(parents=True,exist_ok=True)
F=O/'figures';F.mkdir(exist_ok=True)
df=pd.read_csv(P/'tqe_hw_results.csv');plan=json.loads((P/'tqe_hw_plan.json').read_text())['items'];raw=json.loads((P/'raw_dbgjqlgte5oc738iauug/counts.json').read_text());assert len(df)==len(raw)==len(plan)==84
assert df.id.nunique()==21 and set(df.shots)=={8000};assert set(df.groupby(['n','method']).size())=={3}
model=json.loads((R/'models/U1_power_law.json').read_text());manifest=json.loads((R/'results/pennylane_probability_1050/manifest.json').read_text());models=manifest['models'];reference=pd.read_csv(R/'results/pennylane_probability_1050/paired_probabilities_1050.csv').set_index(['id','p']);cache={};validated=[]
def energy(x,it):
 e=np.zeros(len(x))
 for (i,j),w in zip(it['edges'],it['weights']):e+=w*(1-2*(((x>>i)^(x>>j))&1))
 return e
for it,rr,row in zip(plan,raw,df.to_dict('records')):
 assert it['id']==rr['id']==row['id'] and f"{it['method']}_p{it['p']}"==rr['method']==row['method']
 gid=it['id'];n=it['n'];p=it['p'];assert 'regular3-unit' in gid and set(it['weights'])=={1.}
 if gid not in cache:
  ev=energy(np.arange(1<<n,dtype=np.int64),it);cache[gid]=(float(ev.min()),float(ev.max()),int(np.count_nonzero(abs(ev-ev.min())<1e-9)))
 emin,emax,nopt=cache[gid];x=np.fromiter((int(b,2) for b in rr['counts']),dtype=np.int64);c=np.array(list(rr['counts'].values()));e=energy(x,it);shots=c.sum();assert shots==row['shots']
 successes=int(c[abs(e-emin)<1e-9].sum());prob=successes/shots;ratio=(emax-np.dot(c,e)/shots)/(emax-emin)
 np.testing.assert_allclose([prob,ratio,nopt/2**n],[row['p_opt_hw'],row['ratio_hw'],row['p_opt_random']],atol=1e-12,rtol=0)
 ref=reference.loc[(gid,p)];key='u1' if it['method']=='u1' else 'qaoa';np.testing.assert_allclose([row['p_opt_ideal'],row['ratio_ideal']],[ref['probability_'+key],ref['ratio_'+key]],atol=1e-12,rtol=0)
 D=2*len(it['edges'])/n;s=np.sqrt(np.mean(np.square(it['weights'])))
 if key=='u1':
  t=model['templates'][str(p)];r=D/model['reference_degree'];expected={'gamma':[z['gamma']['A']*r**z['gamma']['b']/s for z in t['layers']],'alpha':[z['alpha']['A']*r**z['alpha']['b']/s for z in t['layers']],'beta':t['beta']}
 else:
  t=models['small' if n<=16 else 'large'];assert t['qaoa_fixed_best_by_depth'][str(p)]=='qaoa_trained';t=t['templates'][str(p)]['qaoa'];expected={'gamma':np.array(t['gamma_unit'])*np.arctan(1/np.sqrt(D-1))/s,'alpha':np.zeros(p),'beta':t['beta']}
 for k in expected:np.testing.assert_allclose(it['angles'][k],expected[k],atol=1e-12,rtol=0)
 validated.append(dict(**row,successes=successes,recomputed_probability=prob,recomputed_ratio=ratio))
df=pd.DataFrame(validated);df.to_csv(O/'validated_circuit_results.csv',index=False)
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42});colors={'u1':'#2476cf','qaoa':'#e87837'};names={'u1':'Power-law DC-QAOA','qaoa':'Fixed-angle QAOA'};ns=sorted(df.n.unique())
def save(fig,name):
 fig.savefig(F/(name+'.png'),dpi=220,bbox_inches='tight');fig.savefig(F/(name+'.pdf'),bbox_inches='tight');plt.close(fig)
def axes2(title):
 fig,axes=plt.subplots(1,2,figsize=(11,4.1),layout='constrained');fig.suptitle(title)
 for j,ax in enumerate(axes):ax.set_title(f'({"ab"[j]}) $p={j+1}$',loc='left');ax.set_xlabel('Qubits, $n$');ax.set_xticks(ns);ax.grid(axis='y',alpha=.2)
 return fig,axes
fig,axes=axes2('IBM Fez: optimal-solution probability on 21 regular3-unit graphs')
for p,ax in enumerate(axes,1):
 for key in ['u1','qaoa']:
  d=df[df.method==f'{key}_p{p}'];g=d.groupby('n');ax.plot(ns,g.p_opt_hw.mean()*100,'o-',color=colors[key],label=names[key]+' | hardware');ax.plot(ns,g.p_opt_ideal.mean()*100,'o--',ms=3,color=colors[key],alpha=.65,label=names[key]+' | ideal')
 ax.plot(ns,d.groupby('n').p_opt_random.mean()*100,':',color='.4',label='Uniform random');ax.set_yscale('log');ax.set_ylabel('Mean optimal-solution probability (%)')
axes[0].legend(fontsize=7.7);save(fig,'H01_probability_by_size')
fig,axes=axes2('Same-graph probability differences: hardware, equal layer count');paired=[]
for p,ax in enumerate(axes,1):
 a=df[df.method==f'u1_p{p}'].set_index('id');b=df[df.method==f'qaoa_p{p}'].set_index('id');delta=100*(a.p_opt_hw-b.p_opt_hw)
 for n in ns:
  ids=sorted(a[a.n==n].index);v=delta.loc[ids].values;ax.scatter(np.array([n]*3)+[-.16,0,.16],v,color=colors['u1'],s=28,alpha=.7);ax.plot([n-.32,n+.32],[v.mean()]*2,color='black',lw=2)
 for gid in a.index:paired.append({'id':gid,'n':int(a.loc[gid,'n']),'p':p,'delta_pp':delta[gid],'delta_ratio':a.loc[gid,'ratio_hw']-b.loc[gid,'ratio_hw'],'ideal_delta_pp':100*(a.loc[gid,'p_opt_ideal']-b.loc[gid,'p_opt_ideal'])})
 ax.axhline(0,color='.4',lw=1);ax.set_ylabel('Power-law minus QAOA (percentage points)');ax.text(.03,.97,f'Mean: {delta.mean():+.3f} pp; higher on {(delta>0).sum()}/21 graphs',transform=ax.transAxes,va='top',fontsize=9)
fig.text(.5,-.035,'Dots: individual graphs; black bars: three-graph means. Descriptive differences, not significance tests.',ha='center',fontsize=9);save(fig,'H02_paired_probability_differences');pairs=pd.DataFrame(paired);pairs.to_csv(O/'paired_differences.csv',index=False)
fig,axes=plt.subplots(1,2,figsize=(11,4.1),layout='constrained');fig.suptitle('Circuit resources and probability retained relative to ideal simulation')
# Orange p=1 is a 35% white tint of the base orange used for p=2.
depth_colors={'u1':{1:'#6BA6E5',2:'#2477D1'},'qaoa':{1:'#F1AE87',2:'#E87837'}}
for p in [1,2]:
 for key in ['u1','qaoa']:
  d=df[df.method==f'{key}_p{p}'].groupby('n').mean(numeric_only=True);marker='o' if p==1 else 's';lab=names[key]+f', $p={p}$'
  axes[0].plot(ns,d.twoq,marker=marker,ms=5,ls='-',color=depth_colors[key][p],label=lab);axes[1].plot(ns,100*d.p_opt_hw/d.p_opt_ideal,marker=marker,ms=5,ls='-',color=depth_colors[key][p],label=lab)
for ax in axes:ax.set_xlabel('Qubits, $n$');ax.set_xticks(ns);ax.grid(axis='y',alpha=.2)
axes[0].set_title('(a) Transpiled two-qubit gates',loc='left');axes[0].set_ylabel('Mean two-qubit gate count');axes[0].legend(fontsize=9,handlelength=3,markerscale=1.2,labelspacing=.6);axes[1].set_title('(b) Hardware / ideal probability',loc='left');axes[1].set_ylabel('Ratio of three-graph means (%)');save(fig,'H03_resources_and_retention')
fig,axes=axes2('Energy-based companion: MaxCut approximation ratio')
for p,ax in enumerate(axes,1):
 for key in ['u1','qaoa']:
  d=df[df.method==f'{key}_p{p}'].groupby('n').mean(numeric_only=True);ax.plot(ns,d.ratio_hw,'o-',color=colors[key],label=names[key]+' | hardware');ax.plot(ns,d.ratio_ideal,'o--',ms=3,color=colors[key],alpha=.65,label=names[key]+' | ideal')
 ax.set_ylim(.6,.93);ax.set_ylabel('Mean approximation ratio')
axes[0].legend(fontsize=7.5);save(fig,'H04_approximation_ratio')
summary={'job':'dbgjqlgte5oc738iauug','graphs':21,'circuits':84,'family':'regular3','weights':'unit','shots':8000,'counts_recomputed':True,'angles_match_frozen_rules':True,'ideal_values_match_archived_pennylane':True,'depths':{}}
for p in [1,2]:
 z=pairs[pairs.p==p];a=df[df.method==f'u1_p{p}'];b=df[df.method==f'qaoa_p{p}'];summary['depths'][str(p)]={'powerlaw_probability_percent':a.p_opt_hw.mean()*100,'qaoa_probability_percent':b.p_opt_hw.mean()*100,'mean_delta_pp':z.delta_pp.mean(),'wins':int((z.delta_pp>0).sum()),'ties':int((z.delta_pp==0).sum()),'mean_ratio_delta':z.delta_ratio.mean(),'ideal_delta_pp':z.ideal_delta_pp.mean(),'mean_twoq_powerlaw':a.twoq.mean(),'mean_twoq_qaoa':b.twoq.mean()}
(O/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
