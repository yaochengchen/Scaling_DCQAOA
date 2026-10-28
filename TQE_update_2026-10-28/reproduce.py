"""Portable reproduction from frozen inputs. Never changes manuscript sources."""
from pathlib import Path
import argparse,subprocess,sys,json,shutil,hashlib,importlib.util
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'code'))
def run(*args,cwd=None):
 subprocess.run(list(args),cwd=cwd or ROOT,check=True)
def load(p):return json.loads((ROOT/p).read_text(encoding='utf-8'))
def figures():
 for name in ["figures","editing_audit","reports"]:(ROOT/name).mkdir(exist_ok=True)
 import numpy as np,pandas as pd
 from types import SimpleNamespace
 import render_probability_supp as sp
 from probability_statistics import stats
 data=pd.read_csv(ROOT/'results/pennylane_probability_1050/paired_probabilities_1050.csv')
 rows=[dict(id=r.id,n=int(r.n),p=int(r.p),weight_kind=r.weight_kind,methods={k:{'probability':float(getattr(r,'probability_'+k))} for k in ['u1','old','qaoa']}) for r in data.itertuples()]
 order={g['id']:i for i,g in enumerate(load('results/pennylane_probability_1050/manifest.json')['graphs'])}
 rows.sort(key=lambda r:(order[r['id']],r['p']))
 original=load('results/pennylane_probability_1050/summary.json')
 groups=[(original['depth'][str(p)],[r for r in rows if r['p']==p]) for p in [1,2,3,4]]
 groups += [(original['weights'][f'{w}_p{p}'],[r for r in rows if r['weight_kind']==w and r['p']==p]) for w in ['bimodal','heavy_tail','lognormal','uniform','unit'] for p in [1,4]]
 for expected,group in groups:
  got=stats(group)
  for k in ['mean','ci95','delta_new_old_new_qaoa','delta_ci95']:np.testing.assert_allclose(got[k],expected[k],rtol=0,atol=1e-13)
 # Recompute Figure 4 curves and attainment from all archived evaluations.
 refs={x['id']:x for x in load('results/benchmark_12graphs.json')['rows']}
 traces=[json.loads(p.read_text(encoding='utf-8')) for p in sorted((ROOT/'inputs/optimizer_benchmark/runs').rglob('*.json'))]
 assert len(traces)==720
 old=load('results/benchmark_new_summary.json')
 for method in ['L-BFGS-B','COBYLA','Adam']:
  ts=[t for t in traces if t['method']==method];assert len(ts)==240
  curves=np.array([[t['evaluations'][min(i,len(t['evaluations']))-1]['best_ratio'] for i in range(1,501)] for t in ts])
  hits=[next((e['evaluation'] for e in t['evaluations'] if e['best_ratio']>=refs[t['id']]['ratio_new']-.001),None) for t in ts]
  np.testing.assert_allclose(curves.mean(0),old[method]['mean_best_curve'],rtol=0,atol=1e-13)
  np.testing.assert_allclose(np.quantile(curves,[.1,.9],axis=0),[old[method]['p10_best_curve'],old[method]['p90_best_curve']],rtol=0,atol=1e-13)
  assert hits==old[method]['run_hits'] or sorted(x for x in hits if x is not None)==sorted(x for x in old[method]['run_hits'] if x is not None)
 run(sys.executable,'code/render_shared_column_titles.py')
 run(sys.executable,'code/render_exponents.py')
 run(sys.executable,'code/figure3_depth_average_review.py')
 run(sys.executable,'code/figure6_depth_average_review.py')
 run(sys.executable,'code/render_single_column_trial.py')
 b=load('results/pennylane_probability_low_degree_p6_600/manifest.json')
 df=pd.read_csv(ROOT/'results/pennylane_probability_low_degree_p6_600/paired_probability_600_five_depths.csv').set_index(['id','p'])
 values=np.array([[[df.loc[(g,p),'probability_powerlaw'],df.loc[(g,p),'probability_qaoa']] for p in sp.campaign.DEPTHS] for g in b['selected_ids']])
 ns=np.array([b['graphs'][g]['n'] for g in b['selected_ids']]);ss=sp.render_main(ns,values)
 frozen=load('results/pennylane_probability_low_degree_p6_600/summary.json')['main']
 for p in sp.campaign.DEPTHS:
  for k in ['mean','ci95','mean_paired_difference','paired_difference_ci95']:np.testing.assert_allclose(ss['depths'][str(p)][k],frozen['depths'][str(p)][k],rtol=0,atol=2e-14)
 low=pd.read_csv(ROOT/'results/pennylane_probability_low_degree_p6_600/low_degree_probability_55.csv').set_index(['id','p'])
 def method(b,g,p,k):return {'result':{'probability':float(low.loc[(g,p),{'u1':'probability_powerlaw','qaoa':'probability_qaoa','old':'probability_previous_rule'}[k]])}}
 sp.campaign.check_method=method;sp.render_low(b)
 run('pdflatex','-interaction=nonstopmode','-halt-on-error','main.tex',cwd=ROOT/'code/figure1')
 mapping={'fig01_powerlaw_overview':'code/figure1/main','fig02_powerlaw_scaling_p1':'figures/u1_revision_1452s1/fig02_powerlaw_scaling_p1','figS1_powerlaw_scaling_p4':'figures/u1_revision_1452s1/figS1_powerlaw_scaling_p4','figS2_powerlaw_exponents':'editing_audit/side_S2_S3_2026-10-27/FigureS2_U1','fig06_powerlaw_probability_family_depth_averaged_size_1050':'figures/figure6_depth_average_probability/fig06_family_probability_depth_averaged_size','figS3_low_degree_probability':'figures/pennylane_probability_low_degree_p6_600/supplement_low_degree_11_probability_diagnostic','figS4_p6_probability_600':'figures/pennylane_probability_low_degree_p6_600/supplement_p6_600_powerlaw_vs_qaoa_probability','figS5_probability_size_by_depth':'figures/figure3_depth_average_probability/figS09_probability_size_by_depth'}
 for n in [3,4,5,7]:mapping[f'fig{n:02d}_single_column']=f'figures/vertical_single_column_trial/fig{n:02d}_single_column'
 out=ROOT/'reproduced/figures';out.mkdir(parents=True,exist_ok=True)
 for dst,src in mapping.items():shutil.copy2(ROOT/(src+'.pdf'),out/(dst+'.pdf'))
 (ROOT/'reproduced/figure_map.json').write_text(json.dumps(mapping,indent=2))
 run(sys.executable,'code/render_hardware.py')
 run(sys.executable,'code/render_main_hardware.py')
 for src,dst in [('H02_paired_probability_differences','figS6_hardware_paired'),('H03_resources_and_retention','figS7_hardware_resources')]:shutil.copy2(ROOT/'reproduced/hardware/figures'/(src+'.pdf'),out/(dst+'.pdf'))
 print('All 15 publication figures regenerated.')
def build():
 out=ROOT/'reproduced/paper';out.mkdir(parents=True,exist_ok=True)
 for p in (ROOT/'manuscript/paper/latex_preview_v4_merged').iterdir():
  if p.is_file() and p.suffix!='.pdf':shutil.copy2(p,out/p.name)
 figs=ROOT/'reproduced/figures'
 if not figs.exists():shutil.copytree(ROOT/'manuscript/paper/latex_preview_v4_merged/figures',figs)
 shutil.copytree(figs,out/'figures',dirs_exist_ok=True)
 for doc in ['main','main','supplementary','supplementary']:run('pdflatex','-interaction=nonstopmode','-halt-on-error',doc+'.tex',cwd=out)

def verify():
 import numpy as np,pandas as pd
 df=pd.read_csv(ROOT/'results/pennylane_probability_1050/paired_probabilities_1050.csv');assert len(df)==4200 and df.id.nunique()==1050 and not df.duplicated(['id','p']).any()
 assert set(df.backend)=={'pennylane.lightning.qubit'};assert set(df.groupby(['n','family','weight_kind','p']).size())=={3}
 summary=load('results/pennylane_probability_1050/summary.json')
 for p in range(1,5):np.testing.assert_allclose(df[df.p==p][['probability_u1','probability_old','probability_qaoa']].mean(),summary['depth'][str(p)]['mean'],rtol=0,atol=1e-14)
 p6=pd.read_csv(ROOT/'results/pennylane_probability_low_degree_p6_600/paired_probability_600_five_depths.csv');low=pd.read_csv(ROOT/'results/pennylane_probability_low_degree_p6_600/low_degree_probability_55.csv');assert len(set(zip(df.id,df.p))|set(zip(p6.id,p6.p))|set(zip(low.id,low.p)))==4850
 if (ROOT/'SHA256.json').exists():
  for f,h in load('SHA256.json').items():
   assert hashlib.sha256((ROOT/f).read_bytes()).hexdigest()==h,f
 print('Frozen file integrity and graph counts/means verified.')
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('command',choices=['verify','figures','build','all']);cmd=a.parse_args().command
 if cmd in ['verify','all']:verify()
 if cmd in ['figures','all']:figures()
 if cmd in ['build','all']:build()
