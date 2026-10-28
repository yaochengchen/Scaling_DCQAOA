"""Reconstruct all fitted amplitudes/exponents from archived training angles."""
from pathlib import Path
import sys,json,argparse
import numpy as np
ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT/'code'))
import study
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--bootstrap',action='store_true',help='Also repeat the historical 500-resample exponent analysis (slower)');a=ap.parse_args();d=study.load_training();m=json.loads((ROOT/'models/U1_power_law.json').read_text());D0=m['reference_degree'];out=ROOT/'reproduced/parameter_fit';out.mkdir(parents=True,exist_ok=True);rows=[]
 for p in sorted(d.p.unique()):
  for l in range(1,p+1):
   v=study.figtools.layer_values(d,p,l,D0);v=v[v.D>1]
   for q,col in [('gamma','g_raw'),('alpha','a_raw')]:
    x,y,_,_,n=study.figtools.bin_stats(v.D,v[col]);keep=(x>1)&(y!=0);x,y,n=x[keep],y[keep],n[keep];w=np.sqrt(n);X=np.column_stack([np.ones(len(x)),np.log(x/D0)]);c=np.linalg.lstsq(X*w[:,None],np.log(abs(y))*w,rcond=None)[0];A=float(np.sign(np.median(y))*np.exp(c[0]));b=float(c[1]);expected=m['templates'][str(p)]['layers'][l-1][q];np.testing.assert_allclose([A,b],[expected['A'],expected['b']],atol=1e-12,rtol=0);rows.append(dict(p=int(p),layer=l,angle=q,A=A,b=b))
 (out/'coefficients.json').write_text(json.dumps(rows,indent=2));print('All 64 fitted amplitudes/exponents match frozen model. The 16 mixer values and D0 are retained from the original pooled schedule.')
 if a.bootstrap:study.figtools.table_exponents(d,D0,out)
