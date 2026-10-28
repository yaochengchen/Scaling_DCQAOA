"""Re-evaluate frozen probability pairs with PennyLane, independent of original paths."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import sys,json,argparse
from concurrent.futures import ProcessPoolExecutor
ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT/'code'))
import circuit_core as core
import numpy as np

def inputs():
 main=json.loads((ROOT/'results/pennylane_probability_1050/manifest.json').read_text(encoding='utf-8'));extra=json.loads((ROOT/'results/pennylane_probability_low_degree_p6_600/manifest.json').read_text(encoding='utf-8'))
 graphs={g['id']:g for g in main['graphs']};graphs.update(extra['graphs']);jobs={(g['id'],p) for g in main['graphs'] for p in [1,2,3,4]};jobs.update((g,6) for g in extra['selected_ids']);jobs.update((g,p) for g in extra['low_ids'] for p in [1,2,3,4,6])
 import csv
 refs={}
 for file,cols in [('pennylane_probability_1050/paired_probabilities_1050.csv',('probability_u1','probability_qaoa')),('pennylane_probability_low_degree_p6_600/paired_probability_600_five_depths.csv',('probability_powerlaw','probability_qaoa')),('pennylane_probability_low_degree_p6_600/low_degree_probability_55.csv',('probability_powerlaw','probability_qaoa'))]:
  for r in csv.DictReader((ROOT/'results'/file).open(encoding='utf-8')):refs[r['id'],int(r['p'])]=[float(r[c]) for c in cols]
 return graphs,main['models'],sorted(jobs,key=lambda t:(graphs[t[0]]['n'],t[1],t[0])),refs

def work(arg):
 g,p,models,ref,out=arg;key=core.object_digest({'graph':g,'p':p,'models':models,'protocol':core.PROTOCOL,'methods':['u1','qaoa']})
 path=Path(out)/f"{g['id']}-p{p}.json"
 if path.exists():
  r=json.loads(path.read_text());assert r['input_sha256']==key and r['checksum']==core.object_digest({k:v for k,v in r.items() if k!='checksum'})
  for k,v in zip(['u1','qaoa'],ref):assert abs(r['methods'][k]['probability']-v)<1e-8
  return 'reused'
 v=core.spectrum(g);mask=core.optimal_mask(v,sum(g['weights']));res={}
 for kind,expected in zip(['u1','qaoa'],ref):
  angles=core.predict(g,p,kind,models);psi=core.make_state_function(g,p,kind=='u1')(*angles);res[kind]=core.measure(psi,v,mask);res[kind]['angles']=[a.tolist() for a in angles]
  if abs(res[kind]['probability']-expected)>1e-8:raise ValueError(f"Probability mismatch {g['id']} p={p} {kind}: {res[kind]['probability']} vs {expected}")
 r={'id':g['id'],'p':p,'input_sha256':key,'methods':res,'runtime':core.runtime()};r['runtime']['threads']=1;r['checksum']=core.object_digest(r);core.atomic_json(path,r);return 'computed'
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--all',action='store_true');ap.add_argument('--depth',type=int);ap.add_argument('--low-degree',action='store_true');ap.add_argument('--limit',type=int,default=1);ap.add_argument('--workers',type=int,default=1);ap.add_argument('--out',type=Path,default=ROOT/'reproduced/probability_records');a=ap.parse_args()
 graphs,models,jobs,refs=inputs();jobs=[(g,p) for g,p in jobs if (a.depth is None or p==a.depth) and (not a.low_degree or core.scales(graphs[g])[0]<=1)];jobs=jobs if a.all else jobs[:a.limit];a.out.mkdir(parents=True,exist_ok=True)
 tasks=[(graphs[g],p,models,refs[g,p],str(a.out)) for g,p in jobs]
 with ProcessPoolExecutor(max_workers=a.workers) as pool:
  for i,status in enumerate(pool.map(work,tasks),1):print(f'{i}/{len(tasks)} {status}',flush=True)
 print('Verified against archived probabilities. Outputs are separate; frozen data were not overwritten.')
