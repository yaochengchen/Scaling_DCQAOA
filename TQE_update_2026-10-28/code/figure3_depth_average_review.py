"""Depth-averaged size comparison and four-depth companion from stored PennyLane results."""
from pathlib import Path
import csv,json,hashlib,sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'figures/figure3_depth_average_probability'
DATA=ROOT/'results/pennylane_probability_1050'
SIZES=(10,12,14,16,18,20,22);DEPTHS=(1,2,3,4)
BLUE='#2476cf';ORANGE='#e87837'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(fig,name):
    if '--main-only' in sys.argv and name.startswith('figS'):
        plt.close(fig)
        return
    for ext in ('pdf','png'):fig.savefig(OUT/f'{name}.{ext}',dpi=300,bbox_inches='tight',facecolor='white')
    plt.close(fig)
def panel(ax,x,means,ci,connect,ylim):
    for k,label,color,marker in ((1,'Fixed-angle QAOA',ORANGE,'s'),(0,'Power-law DC-QAOA',BLUE,'o')):
        y=means[:,k]*100;bounds=ci[:,k,:]*100
        ax.errorbar(x,y,yerr=np.maximum(0,[y-bounds[:,0],bounds[:,1]-y]),fmt=marker,
          linestyle='-' if connect else 'none',color=color,label=label,linewidth=1.6,markersize=5,elinewidth=1,capsize=2.5,zorder=3)
    ax.set_xticks(x);ax.set_ylim(0,ylim);ax.grid(axis='y',color='#e5e5e5',linewidth=.7)
    ax.set_axisbelow(True);ax.spines[['top','right']].set_visible(False);ax.tick_params(labelsize=8)

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    source=DATA/'paired_probabilities_1050.csv';original_hash=sha(source)
    rows=list(csv.DictReader(source.open(encoding='utf-8',newline='')))
    assert len(rows)==4200 and len({(r['id'],int(r['p'])) for r in rows})==4200
    assert {r['backend'] for r in rows}=={'pennylane.lightning.qubit'}
    summary=json.loads((DATA/'summary.json').read_text());bykey={(r['id'],int(r['p'])):r for r in rows}
    bysize={};export=[];rng=np.random.default_rng(20261027)
    for n in SIZES:
        ids=sorted({r['id'] for r in rows if int(r['n'])==n});assert len(ids)==150
        for p in DEPTHS:assert {r['id'] for r in rows if int(r['n'])==n and int(r['p'])==p}==set(ids)
        v=np.array([[[float(bykey[g,p]['probability_u1']),float(bykey[g,p]['probability_qaoa'])] for p in DEPTHS] for g in ids])
        assert v.shape==(150,4,2) and np.isfinite(v).all() and ((v>=0)&(v<=1)).all()
        avg=v.mean(axis=1)
        assert np.allclose(avg.mean(axis=0),v.mean(axis=(0,1)),atol=1e-14)
        boot=np.empty((2000,4,2));boot_avg=np.empty((2000,2))
        for start in range(0,2000,100):
            take=rng.integers(0,150,(100,150))
            boot[start:start+100]=v[take].mean(axis=1)
            boot_avg[start:start+100]=avg[take].mean(axis=1)
        assert np.allclose(boot.mean(axis=1),boot_avg,atol=1e-14)
        bysize[str(n)]={'graphs':150,'graph_depth_records':600,'mean':avg.mean(axis=0).tolist(),
          'ci95':np.quantile(boot_avg,[.025,.975],axis=0).T.tolist(),
          'paired_difference':float(np.mean(avg[:,0]-avg[:,1])),
          'paired_difference_ci95':np.quantile(boot_avg[:,0]-boot_avg[:,1],[.025,.975]).tolist(),
          'depth_mean':v.mean(axis=0).tolist(),'depth_ci95':np.quantile(boot,[.025,.975],axis=0).transpose(1,2,0).tolist()}
        for g,a in zip(ids,avg):export.append({'id':g,'n':n,'depths':'1,2,3,4','mean_probability_powerlaw':a[0],'mean_probability_qaoa':a[1]})
    means=np.array([bysize[str(n)]['mean'] for n in SIZES]);cis=np.array([bysize[str(n)]['ci95'] for n in SIZES])
    # Left panel preserves the previous point estimates and intervals exactly.
    dm=np.array([[summary['depth'][str(p)]['mean'][k] for k in (0,2)] for p in DEPTHS])
    dc=np.array([[summary['depth'][str(p)]['ci95'][k] for k in (0,2)] for p in DEPTHS])
    ymax=max(30,float(np.ceil(max(cis.max(),dc.max())*100/5)*5))
    fig,axes=plt.subplots(1,2,figsize=(7.8,3.35),sharey=True,layout='constrained')
    panel(axes[0],DEPTHS,dm,dc,True,ymax);panel(axes[1],SIZES,means,cis,False,ymax)
    axes[0].set_title('(a) By depth: pooled over $n=10$–$22$',loc='left',fontsize=9)
    axes[1].set_title('(b) By size: averaged over $p=1$–$4$',loc='left',fontsize=9)
    axes[0].set_xlabel('Layers, $p$',fontsize=9);axes[1].set_xlabel('Qubits, $n$',fontsize=9)
    axes[0].set_ylabel('Mean optimal-solution probability (%)',fontsize=9)
    axes[0].legend(frameon=False,loc='upper left',fontsize=7.3)
    save(fig,'fig03_probability_depth_averaged_size')
    allci=np.array([bysize[str(n)]['depth_ci95'] for n in SIZES]);ally=np.array([bysize[str(n)]['depth_mean'] for n in SIZES])
    limit=float(np.ceil(allci.max()*100/5)*5)
    fig,axes=plt.subplots(2,2,figsize=(8.3,6),sharex=True,sharey=True,layout='constrained')
    for j,(ax,p) in enumerate(zip(axes.flat,DEPTHS)):
        panel(ax,SIZES,ally[:,j,:],allci[:,j,:,:],True,limit)
        ax.set_title(f'({"abcd"[j]}) $p={p}$',loc='left',fontsize=10)
        if j>=2:ax.set_xlabel('Qubits, $n$',fontsize=9)
        if j%2==0:ax.set_ylabel('Mean optimal-solution probability (%)',fontsize=9)
    axes[0,0].legend(frameon=False,fontsize=8,loc='upper right')
    fig.suptitle('Size dependence at each depth: 150 graphs per size',fontsize=11)
    save(fig,'figS09_probability_size_by_depth')
    result={'graphs':1050,'records':4200,'backend':'pennylane.lightning.qubit','methods':['powerlaw','qaoa'],
      'size_statistics':bysize,'bootstrap_replicates':2000,'seed':20261027,
      'resampling':'Within each size, resample 150 graph IDs with replacement; carry all four depths and both methods together.',
      'averaging':'Equal weights 1/4 for depths 1,2,3,4 within each graph; then equal graph weights.',
      'interval':'Pointwise 95% percentile intervals; no multiple-comparison adjustment.',
      'left_panel_statistics_unchanged':True,'main_ylim_percent':ymax,'companion_ylim_percent':limit,
      'source_sha256':original_hash,'source_summary_sha256':sha(DATA/'summary.json'),'quantum_simulations':0}
    (OUT/'statistics.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    with (OUT/'per_graph_depth_averages.csv').open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=export[0]);w.writeheader();w.writerows(export)
    assert sha(source)==original_hash
    print(json.dumps({'graphs':1050,'records':4200,'main_ylim':ymax,'supplement_ylim':limit,'source_unchanged':True}))

if __name__=='__main__':main()
