"""Render depth-averaged family/size probability gaps and audit hidden negative means."""
from pathlib import Path
import csv,json,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'figures/figure6_depth_average_probability'
SOURCE=ROOT/'results/pennylane_probability_1050/paired_probabilities_1050.csv'
FAMILIES=('tree','ring','regular3','regular4','regular5','regular6','er3','er05','community','bipartite')
SIZES=(10,12,14,16,18,20,22);DEPTHS=(1,2,3,4)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    source_sha=sha(SOURCE)
    rows=list(csv.DictReader(SOURCE.open(encoding='utf-8',newline='')))
    assert len(rows)==4200 and len({(r['id'],int(r['p'])) for r in rows})==4200
    assert {r['backend'] for r in rows}=={'pennylane.lightning.qubit'}
    left=np.zeros((10,4));cube=np.zeros((10,7,4));audit=[];pergraph=[]
    for i,f in enumerate(FAMILIES):
        for j,n in enumerate(SIZES):
            group=[r for r in rows if r['family']==f and int(r['n'])==n]
            ids=sorted({r['id'] for r in group});assert len(ids)==15 and len(group)==60
            lookup={(r['id'],int(r['p'])):r for r in group}
            values=np.array([[100*(float(lookup[g,p]['probability_u1'])-float(lookup[g,p]['probability_qaoa'])) for p in DEPTHS] for g in ids])
            assert values.shape==(15,4) and np.isfinite(values).all()
            cube[i,j]=values.mean(axis=0)
            for g,v in zip(ids,values):
                pergraph.append({'id':g,'n':n,'family':f,**{f'delta_p{p}_pp':v[k] for k,p in enumerate(DEPTHS)},'depth_mean_delta_pp':v.mean()})
            for k,p in enumerate(DEPTHS):
                audit.append({'family':f,'n':n,'p':p,'graphs':15,'mean_delta_pp':cube[i,j,k],
                  'negative_graphs':int((values[:,k]<-1e-12).sum()),'min_graph_delta_pp':values[:,k].min(),
                  'max_graph_delta_pp':values[:,k].max()})
        for k,p in enumerate(DEPTHS):
            g=[r for r in rows if r['family']==f and int(r['p'])==p];assert len(g)==105
            left[i,k]=np.mean([100*(float(r['probability_u1'])-float(r['probability_qaoa'])) for r in g])
    right=cube.mean(axis=2)
    assert np.allclose(left,cube.mean(axis=1),atol=1e-12)
    original=json.loads((ROOT/'results/pennylane_probability_1050/summary.json').read_text())
    before=np.array(original['family_depth_mean_percent'])
    assert np.allclose(left,before[0]-before[2],atol=1e-12)
    scale=max(float(np.abs(left).max()),float(np.abs(right).max()))
    fig,axes=plt.subplots(1,2,figsize=(12.8,5.1),sharey=True)
    for ax,m,xs,title,xlabel in zip(axes,(left,right),(DEPTHS,SIZES),
       ('(a) Depth dependence · pooled over $n=10$–$22$','(b) Size dependence · averaged over $p=1$–$4$'),('Layers $p$','Qubits $n$')):
        im=ax.imshow(m,cmap='RdBu',vmin=-scale,vmax=scale,aspect='auto')
        ax.set_yticks(range(10),FAMILIES);ax.set_xticks(range(len(xs)),xs)
        ax.set_xlabel(xlabel);ax.set_title(title,loc='left',fontsize=10)
        ax.set_xticks(np.arange(-.5,len(xs),1),minor=True);ax.set_yticks(np.arange(-.5,10,1),minor=True)
        ax.grid(which='minor',color='white',linewidth=.7);ax.tick_params(which='minor',bottom=False,left=False)
        for i in range(10):
            for j in range(len(xs)):
                ax.text(j,i,f'{m[i,j]:+.1f}',ha='center',va='center',fontsize=7.4,color='white' if abs(m[i,j])>.6*scale else '#202020')
    fig.subplots_adjust(left=.12,right=.87,bottom=.13,top=.9,wspace=.22)
    ca=fig.add_axes([.9,.18,.018,.64]);fig.colorbar(im,cax=ca,label='Mean paired probability difference (percentage points)')
    for ext in ('pdf','png'):fig.savefig(OUT/f'fig06_family_probability_depth_averaged_size.{ext}',dpi=300,bbox_inches='tight',facecolor='white')
    plt.close(fig)
    neg=[r for r in audit if r['mean_delta_pp']<-1e-12]
    masked=[r for r in neg if right[FAMILIES.index(r['family']),SIZES.index(r['n'])]>0]
    counts={str(p):sum(r['negative_graphs'] for r in audit if r['p']==p) for p in DEPTHS}
    summary={'graphs':1050,'records':4200,'backend':'pennylane.lightning.qubit','source_sha256':source_sha,
      'left_graphs_per_cell':105,'right_graphs_per_cell':15,'right_depths_per_graph':4,'units':'percentage points',
      'left_matrix':left.tolist(),'right_matrix':right.tolist(),'size_depth_cube':cube.tolist(),
      'families':FAMILIES,'sizes':SIZES,'depths':DEPTHS,'color_limit_pp':scale,
      'negative_family_size_depth_cells':neg,'negative_cells_masked_by_positive_depth_mean':masked,
      'total_family_size_depth_cells':280,'minimum_cell':min(audit,key=lambda r:r['mean_delta_pp']),
      'negative_individual_graphs_by_depth':counts,'individual_graph_depth_records':4200,
      'negative_depth_averaged_graphs':int(sum(r['depth_mean_delta_pp']<-1e-12 for r in pergraph)),
      'left_statistics_unchanged':True,'quantum_simulations':0}
    (OUT/'statistics_and_negative_audit.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    for name,data in [('family_size_depth_audit.csv',audit),('per_graph_depth_differences.csv',pergraph)]:
        with (OUT/name).open('w',encoding='utf-8',newline='') as f:
            w=csv.DictWriter(f,fieldnames=data[0]);w.writeheader();w.writerows(data)
    assert sha(SOURCE)==source_sha
    print(json.dumps({k:summary[k] for k in ('negative_family_size_depth_cells','minimum_cell','negative_individual_graphs_by_depth','negative_depth_averaged_graphs','color_limit_pp')}))

if __name__=='__main__':main()
