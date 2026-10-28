import numpy as np
from types import SimpleNamespace
core=SimpleNamespace(KINDS=("u1","old","qaoa"))
def stats(rows):
    v=np.array([[r['methods'][k]['probability'] for k in core.KINDS] for r in rows]);ns=np.array([r['n'] for r in rows])
    boot=np.zeros((2000,3));rng=np.random.default_rng(20261027)
    for n in sorted(set(ns)):
        z=v[ns==n]
        for start in range(0,2000,100):boot[start:start+100]+=z[rng.integers(0,len(z),(100,len(z)))].sum(axis=1)/len(v)
    dif=np.column_stack((v[:,0]-v[:,1],v[:,0]-v[:,2]))
    b=np.column_stack((boot[:,0]-boot[:,1],boot[:,0]-boot[:,2]))
    return {'count':len(rows),'mean':v.mean(axis=0).tolist(),'ci95':np.quantile(boot,[.025,.975],axis=0).T.tolist(),
            'delta_new_old_new_qaoa':dif.mean(axis=0).tolist(),'delta_ci95':np.quantile(b,[.025,.975],axis=0).T.tolist()}
