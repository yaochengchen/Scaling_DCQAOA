from pathlib import Path
import sys,json
import numpy as np,pandas as pd
S=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(S/"code/reference_analysis"))
import prr_figures as figtools
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
