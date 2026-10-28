"""Isolated A-joint study. No optimization; originals are never written."""
from pathlib import Path
import os,sys
os.environ['OMP_NUM_THREADS']='4'
os.environ['OPENBLAS_NUM_THREADS']='4'
sys.dont_write_bytecode=True
S=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(S/'code/reference'),str(S/'code/reference_analysis')]
import json,gzip,time,hashlib,copy
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import prr_figures as figtools
from evaluate_unified_table import _instance,energy_qulacs,predict_unified
from cdtransfer.simulator import Angles

def readgz(path):
    with gzip.open(path,'rt',encoding='utf-8') as f:return json.load(f)
def dump(path,obj):path.write_text(json.dumps(obj,indent=2,ensure_ascii=False),encoding='utf-8')
def load_training():
    frames=[]
    for name in ['small','large','extra']:
        d=pd.read_csv(S/f'inputs/{name}_best_optima.csv.gz',dtype={'cd':str,'gamma':str,'alpha':str,'beta':str})
        d=d[d.cd.str.strip().str.lower()=='true']
        if name=='extra':d=d[d.p==6]
        frames.append(d)
    d=pd.concat(frames,ignore_index=True);d=d.loc[d.groupby(['id','p']).energy.idxmin()]
    rows=[]
    for r in d.itertuples(index=False):
        g=np.asarray(json.loads(r.gamma));b=np.asarray(json.loads(r.beta));a=np.asarray(json.loads(r.alpha))
        if np.mean(g)>0:g=-g;b=-b
        rows.append(dict(id=r.id,p=int(r.p),D=float(r.degree),s=float(r.rms),family=r.family,gamma=g,beta=b,alpha=a))
    return pd.DataFrame(rows)

def fit():
    original=json.loads((S/'inputs/original_model.json').read_text());D0=original['reference_degree'];df=load_training()
    model={'name':'A-joint-layerwise-training-fit-v1','status':'exploratory_not_validation_selected','reference_degree':D0,'low_degree_policy':'use original rule for D<=1','original_model_sha256':hashlib.sha256((S/'inputs/original_model.json').read_bytes()).hexdigest(),'templates':{},'fit_method':'count-weighted log absolute degree-bin medians; width=.5,min_count=10,D>1; beta retained'}
    stats=[]
    for p in sorted(df.p.unique()):
        layers=[]
        for l in range(1,p+1):
            d=figtools.layer_values(df,p,l,D0);d=d[d.D>1];entry={}
            for key,col in [('gamma','g_raw'),('alpha','a_raw')]:
                x,y,lo,hi,n=figtools.bin_stats(d.D,d[col]);keep=(x>1)&(y!=0);x,y,n=x[keep],y[keep],n[keep]
                sign=float(np.sign(np.median(y)));support=float(n[np.sign(y)==sign].sum()/n.sum())
                if support<.95:raise ValueError(f'Mixed-sign bin medians: p={p},layer={l},{key}, support={support}')
                X=np.column_stack([np.ones(len(x)),np.log(x/D0)]);w=np.sqrt(n)
                coef=np.linalg.lstsq(X*w[:,None],np.log(abs(y))*w,rcond=None)[0]
                entry[key]={'A':float(sign*np.exp(coef[0])),'b':float(coef[1]),'sign_support':support}
                stats.append({'p':p,'layer':l,'quantity':key,**entry[key],'bins':len(x),'graphs_in_bins':int(n.sum())})
            layers.append(entry)
        model['templates'][str(p)]={'layers':layers,'beta':original['templates'][str(p)]['beta']}
    dump(S/'models/A_joint_candidate.json',model);pd.DataFrame(stats).to_csv(S/'results/fitted_coefficients.csv',index=False)
    print('FIT',pd.DataFrame(stats).to_string(index=False),flush=True)
    for p,name in [(1,'Figure2_A_joint'),(4,'FigureS1_A_joint')]:
        d=figtools.layer_values(df,p,1,D0);D=np.linspace(max(1.001,d.D.min()),d.D.max(),300);e=model['templates'][str(p)]['layers'][0];t=original['templates'][str(p)]
        g=e['gamma']['A']*(D/D0)**e['gamma']['b'];a=e['alpha']['A']*(D/D0)**e['alpha']['b']
        new=[g,g/figtools.f_D(D),a,a*np.sqrt(D/D0)];old=[t['gamma_unit'][0]*figtools.f_D(D),np.full_like(D,t['gamma_unit'][0]),t['alpha_unit'][0]*np.sqrt(D0/D),np.full_like(D,t['alpha_unit'][0])]
        fig,axes=plt.subplots(2,2,figsize=(11,8.2),sharex=True)
        labels=[r'$\gamma_1s_J$',r'$\gamma_1s_J/f_D(D)$',r'$\alpha_1s_J$',r'$\alpha_1s_J\sqrt{D/D_0}$']
        for i,(ax,col,lab) in enumerate(zip(axes.flat,['g_raw','g_col','a_raw','a_col'],labels)):
            for j,fam in enumerate(figtools.FAMILIES):
                z=d[d.family==fam];ax.scatter(z.D,z[col],s=5,alpha=.25,c=[plt.get_cmap('tab10')(j)],linewidths=0)
            figtools._binned(ax,d.D,d[col]);ax.plot(D,old[i],':',color='.5',label='Original rule');ax.plot(D,new[i],'--',color='crimson',label='A-joint candidate')
            lo,hi=np.quantile(d[col],[.02,.98]);pad=.25*(hi-lo);ax.set(ylim=(lo-pad,hi+pad),ylabel=lab);ax.legend(fontsize=7)
        axes[0,0].set_title('Weight scale removed');axes[0,1].set_title('Original rescaling held fixed')
        for ax in axes[1]:ax.set_xlabel('Mean degree D')
        fig.suptitle(f'A-joint exploratory rule | depth {p}, layer 1 | {len(d)} training graphs\nNo performance validation; candidate curves shown for D > 1',fontsize=12)
        from matplotlib.lines import Line2D
        handles=[Line2D([],[],marker='o',ls='',color=plt.get_cmap('tab10')(j),label=fam,markersize=5) for j,fam in enumerate(figtools.FAMILIES)]
        fig.legend(handles=handles,loc='lower center',ncol=5,fontsize=8,bbox_to_anchor=(.5,.01))
        fig.tight_layout(rect=(0,.08,1,.93))
        for ext in ['png','pdf']:fig.savefig(S/f'figures/new/{name}.{ext}',dpi=170)
        plt.close(fig)
    # Existing bootstrap intervals remain descriptive; candidate exponent is the same fitted statistic.
    ex=pd.read_csv(S/'inputs/original_effective_exponents.csv');fig,axes=plt.subplots(2,1,figsize=(7,6),sharex=True)
    for ax,key in zip(axes,['alpha','gamma']):
        z=ex[(ex.layer==1)&ex.quantity.str.startswith(key)].sort_values('p');b=[model['templates'][str(p)]['layers'][0][key]['b'] for p in z.p]
        ax.errorbar(z.p,z.fitted_exponent,yerr=[z.fitted_exponent-z.ci_lo,z.ci_hi-z.fitted_exponent],fmt='ko',capsize=3,label='Existing fit / bootstrap interval')
        ax.plot(z.p,z.rule_exponent,':',color='.5',label='Original rule effective slope');ax.plot(z.p,b,'rx',ms=8,label='A-joint fitted exponent (same statistic)');ax.set_ylabel(key+' effective exponent');ax.legend(fontsize=8)
    axes[-1].set_xlabel('Circuit depth p');fig.suptitle('Descriptive training exponents, not independent validation\nA-joint markers coincide with the fitted slopes by construction',fontsize=10);fig.tight_layout(rect=(0,0,1,.92))
    for ext in ['png','pdf']:fig.savefig(S/f'figures/new/FigureS2_A_joint.{ext}',dpi=170)
    plt.close(fig)

def predict(inst,p,candidate,original):
    D=2*inst.m/inst.n
    if D<=1:return predict_unified(inst,p,original)
    t=candidate['templates'][str(p)];s=np.sqrt(np.mean(inst.weights**2));r=D/candidate['reference_degree']
    return Angles(np.array([x['gamma']['A']*r**x['gamma']['b']/s for x in t['layers']]),np.array(t['beta']),np.array([x['alpha']['A']*r**x['alpha']['b']/s for x in t['layers']]))

def benchmark():
    original=json.loads((S/'inputs/original_model.json').read_text());candidate=json.loads((S/'models/A_joint_candidate.json').read_text())
    candidate_hash=hashlib.sha256((S/'models/A_joint_candidate.json').read_bytes()).hexdigest()
    manifest=json.loads((S/'inputs/optimizer_benchmark/manifest.json').read_text());ids=manifest['test_graph_ids']
    full=readgz(S/'inputs/eqdepth_small_N20_manifest.json.gz');rows={r['id']:r for r in full['splits']['test']}
    oldrecords=readgz(S/'inputs/eqdepth_small_N20_heldout_results.json.gz');refs={(r['id'],r['p']):r for r in oldrecords}
    from pennylane_n12p2 import energy_pennylane
    output=[]
    for graphid in ids:
        row=rows[graphid];inst=_instance(row);hist=refs[(graphid,2)];oldangle=predict_unified(inst,2,original);newangle=predict(inst,2,candidate,original)
        start=time.perf_counter();eo=energy_qulacs(inst,oldangle);en=energy_qulacs(inst,newangle);ep=energy_pennylane(inst,newangle);elapsed=time.perf_counter()-start
        assert abs(en-ep)<1e-7*hist['norm_weights'],(en,ep)
        ratio=lambda e:(hist['max_energy']-e)/(hist['max_energy']-hist['ground_energy'])
        output.append(dict(id=graphid,fingerprint=inst.fingerprint,energy_old=eo,energy_new=en,ratio_old=ratio(eo),ratio_new=ratio(en),delta_ratio=ratio(en)-ratio(eo),seconds_three_calls=elapsed))
        print('BENCH',graphid,ratio(eo),ratio(en),flush=True)
    byid={x['id']:x for x in output};traces=[]
    for file in (S/'inputs/optimizer_benchmark/runs').rglob('*.json'):
        r=json.loads(file.read_text());assert r['fingerprint']==byid[r['id']]['fingerprint'];assert abs(r['fixed_ratio']-byid[r['id']]['ratio_old'])<1e-6
        r['fixed_ratio']=byid[r['id']]['ratio_new'];r['fixed_energy']=byid[r['id']]['energy_new'];traces.append(r)
    from unified_variational_n16p2 import summarize_runs
    methods=['L-BFGS-B','COBYLA','Adam'];summaries={m:summarize_runs([r for r in traces if r['method']==m],500) for m in methods}
    oldsummary=json.loads((S/'inputs/optimizer_benchmark/summary.json').read_text())
    for m in methods:assert np.allclose(summaries[m]['mean_best_curve'],oldsummary[m]['mean_best_curve'],rtol=0,atol=1e-14)
    dump(S/'results/benchmark_12graphs.json',{'candidate_hash':candidate_hash,'rows':output,'calls':{'old_qulacs':12,'new_qulacs':12,'new_pennylane_crosscheck':12},'optimization_runs':0})
    dump(S/'results/benchmark_new_summary.json',summaries)
    from plot_optimizer_comparison_n12p2_v3 import build_figure
    fig=build_figure(summaries);fig.axes[0].axhline(np.mean([x['ratio_old'] for x in output]),color='.4',ls=':',lw=1,label='Original frozen rule');fig.axes[0].legend(fontsize=7)
    fig.suptitle('A-joint exploratory target | same 720 archived optimizer trajectories',fontsize=10);fig.set_size_inches(9,4.3)
    for ext in ['png','pdf']:fig.savefig(S/f'figures/new/Figure4_A_joint.{ext}',dpi=190)
    plt.close(fig)
    print('BENCH_SUMMARY',json.dumps({m:{'old_hits':sum(x is not None for x in oldsummary[m]['run_hits']),'new_hits':sum(x is not None for x in summaries[m]['run_hits'])} for m in methods}),flush=True)
    print('MEANS',np.mean([x['ratio_old'] for x in output]),np.mean([x['ratio_new'] for x in output]),flush=True)

def timing():
    model=json.loads((S/'models/A_joint_candidate.json').read_text());orig=json.loads((S/'inputs/original_model.json').read_text());out=[]
    for cohort in ['eqdepth_small_N20','eqdepth_large_N20']:
        manifest=readgz(S/f'inputs/{cohort}_manifest.json.gz');rows=manifest['splits']['test'];pmax=max(manifest['config']['depths'])
        for n in sorted(set(r['n'] for r in rows)):
            group=sorted([r for r in rows if r['n']==n],key=lambda r:len(r['edges']));chosen=[group[0],group[len(group)//2],group[-1]]
            for row in chosen:
                inst=_instance(row)
                for p in [1,pmax]:
                    a=predict(inst,p,model,orig);start=time.perf_counter();energy=energy_qulacs(inst,a);sec=time.perf_counter()-start
                    out.append({'id':row['id'],'n':n,'m':inst.m,'p':p,'seconds':sec,'energy':energy});print('TIMING',n,inst.m,p,round(sec,4),flush=True)
    dump(S/'results/timing_samples.json',{'omp_threads':4,'samples':out,'purpose':'workload estimate only; not representative performance assessment'})

if __name__=='__main__':
    {'fit':fit,'benchmark':benchmark,'timing':timing}[sys.argv[1]]()
