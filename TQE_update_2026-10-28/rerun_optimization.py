"""Optional PennyLane reruns of energy optimizations, separate from frozen records."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import sys,json,argparse
ROOT=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT/'code'),str(ROOT/'code/reference')]
import numpy as np
import circuit_core as core
from cdtransfer.simulator import Angles
from cdtransfer.instances import IsingInstance
from cdtransfer.lightning import make_simulator
from cdtransfer.optimization import run_lbfgs
from portable_optimizer import run_one

def main():
 ap=argparse.ArgumentParser();ap.add_argument('experiment',choices=['benchmark','refinement']);ap.add_argument('--all',action='store_true');ap.add_argument('--limit',type=int,default=1);a=ap.parse_args()
 ginput=json.loads((ROOT/'inputs/figure5_refinement_inputs.json').read_text(encoding='utf-8'));graphs={g['id']:g for g in ginput['graphs']};model=json.loads((ROOT/'models/U1_power_law.json').read_text())
 out=ROOT/'reproduced'/('rerun_'+a.experiment);out.mkdir(parents=True,exist_ok=True)
 if a.experiment=='benchmark':
  records=[json.loads(f.read_text()) for f in sorted((ROOT/'inputs/optimizer_benchmark/runs').rglob('*.json'))];tasks=[(r['id'],r) for r in records]
 else:
  records=[json.loads(f.read_text()) for f in sorted((ROOT/'results/figure5_ajoint_c128_cpu/runs').glob('*.json'))];tasks=[(r['id'],(k,c)) for r in records for k,c in r['chains'].items()]
 if not a.all:tasks=tasks[:a.limit]
 for i,(gid,info) in enumerate(tasks,1):
  g=graphs[gid];angles=Angles(*core.predict(g,2,'u1',{'u1':model}));key=(f"{info['method']}-{info['start']}" if a.experiment=='benchmark' else info[0]);f=out/f'{gid}-{key}.json'
  signature=core.object_digest({'graph':g,'task':info,'model':model,'experiment':a.experiment})
  if f.exists():
   old=json.loads(f.read_text());assert old['input_sha256']==signature;print(i,'reused',flush=True);continue
  if a.experiment=='benchmark':
   result=run_one(g,angles,info['start'],info['method']);np.testing.assert_allclose(result['initial_angles'],info['initial_angles'],atol=1e-14,rtol=0)
   if result['error']:raise RuntimeError(result['error'])
   result['archived_endpoint']=info['evaluations'][-1]['best_ratio'];result['endpoint_drift']=result['evaluations'][-1]['best_ratio']-result['archived_endpoint']
  else:
   k,c=info;inst=IsingInstance.from_dict({k:g[k] for k in ('n','edges','weights','family','weight_kind','seed','split')});sim=make_simulator(inst,{'backend':'lightning.qubit','dtype':'complex128','engine':'tape'})
   init=Angles(**{k:np.array(v) for k,v in c['initial_angles'].items()});trace=[]
   if k=='formula':np.testing.assert_allclose(init.pack(True),angles.pack(True),atol=1e-13,rtol=0)
   r=run_lbfgs(inst,2,True,init.pack(True),sim,maxiter=200,trace_callback=trace.append,ftol=1e-10,gtol=1e-5)
   curve=[(g['max_energy']-e['best_so_far_energy'])/(g['max_energy']-g['ground_energy']) for e in trace]
   result={'trace':trace,'best_so_far_ratio':curve,'returned_angles':r.angles.to_dict(),'archived_endpoint':c['final_ratio'],'endpoint_drift':curve[-1]-c['final_ratio']}
  result['input_sha256']=signature;core.atomic_json(f,result);print(i,len(tasks),key,'endpoint drift',result['endpoint_drift'],flush=True)
 print('Optimizer traces can differ between SciPy/platform versions; archived trajectories define the published plots.')
if __name__=='__main__':main()
