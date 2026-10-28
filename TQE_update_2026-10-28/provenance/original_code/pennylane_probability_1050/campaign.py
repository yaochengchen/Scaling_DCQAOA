"""Freeze, run, resume and verify the PennyLane 1050-graph campaign."""
import argparse,csv,gzip,json,os,time,traceback
from collections import Counter
from pathlib import Path
import core

ROOT=core.ROOT
OUT=ROOT/'results/pennylane_probability_1050'
DEPTHS=(1,2,3,4)

class CampaignLock:
    def __enter__(self):
        import msvcrt
        OUT.mkdir(parents=True,exist_ok=True);self.f=(OUT/'run.lock').open('a+b')
        if self.f.seek(0,2)==0:self.f.write(b'0');self.f.flush()
        self.f.seek(0)
        try:msvcrt.locking(self.f.fileno(),msvcrt.LK_NBLCK,1)
        except OSError:
            self.f.close();raise RuntimeError('Campaign already locked by another process')
        return self
    def __exit__(self,*args):
        import msvcrt
        self.f.seek(0);msvcrt.locking(self.f.fileno(),msvcrt.LK_UNLCK,1);self.f.close()

def status(stage,**extra):core.atomic_json(OUT/'status.json',{'stage':stage,'updated_unix':time.time(),'pid':os.getpid(),**extra})
def taskkey(g,p):return f"{g['id']}-p{p}"
def path_for(g,p):return OUT/'records'/f'{taskkey(g,p)}.json'

def source_paths():
    return {'selection':ROOT/'results/u1_balanced_1050_367/frozen_manifest.json',
            'u1':ROOT/'models/U1_power_law.json','old':ROOT/'inputs/original_model.json',
            'small':core.PROJECT/'Clean/Clean/results/gpu_eqdepth/eqdepth_small_N20/FROZEN_MODEL.json',
            'large':core.PROJECT/'Clean/Clean/results/gpu_eqdepth/eqdepth_large_N20/FROZEN_MODEL.json',
            'csv':core.PROJECT/'Clean/Clean/code/CDQAOA_EqualDepth_LightningGPU_8x/assets/original_transfer_data.csv',
            'small_graphs':ROOT/'inputs/eqdepth_small_N20_manifest.json.gz',
            'large_graphs':ROOT/'inputs/eqdepth_large_N20_manifest.json.gz',
            'energy_reference':ROOT/'results/u1_p6_small_4000_v2/paired_records_32000.csv',
            'core_code':Path(core.__file__),'campaign_code':Path(__file__)}

def prepare():
    if (OUT/'manifest.json').exists():return load_manifest()
    paths=source_paths();selection=json.loads(paths['selection'].read_text())['selected']
    assert len(selection)==1050 and len({g['id'] for g in selection})==1050
    counts=Counter((g['n'],g['family'],g['weight_kind']) for g in selection)
    assert len(counts)==350 and set(counts.values())=={3}
    graphs={}
    for k in ('small_graphs','large_graphs'):
        with gzip.open(paths[k],'rt',encoding='utf-8') as f:graphs.update({g['id']:g for g in json.load(f)['splits']['test']})
    selected=[]
    for s in selection:
        g=graphs[s['id']]
        if core.fingerprint(g)!=s['fingerprint'] or g['fingerprint']!=s['fingerprint']:raise ValueError('Graph fingerprint mismatch')
        if g['edges']!=sorted(g['edges']):raise ValueError('Unexpected edge order')
        selected.append(g)
    models={k:json.loads(paths[k].read_text()) for k in ('u1','old','small','large')}
    models['csv']={}
    with paths['csv'].open(newline='',encoding='utf-8') as f:
        for row in list(csv.reader(f))[2:]:
            if len(row)>3 and row[0].strip() and row[1].strip() and int(float(row[0]))==2:
                p=int(float(row[1]));models['csv'][str(p)]=[float(x) for x in row[3:3+2*p]]
    ids={g['id'] for g in selected}
    with paths['energy_reference'].open(newline='',encoding='utf-8-sig') as f:
        refs={f"{r['id']}-p{r['p']}":r for r in csv.DictReader(f) if r['id'] in ids and int(r['p']) in DEPTHS}
    assert len(refs)==4200
    legacy={}
    for g in selected:
        for p in DEPTHS:
            key=taskkey(g,p)
            for source in (ROOT/f"results/optimal_probability_u1_old_qaoa/records/{g['n']}/{key}.json",ROOT/f'results/optimal_probability_quick_preview/records/{key}.json'):
                if source.exists():
                    r=json.loads(source.read_text())
                    if r['fingerprint']!=g['fingerprint'] or r['model_sha256']!=core.digest(paths['u1']):raise ValueError('Legacy identity mismatch')
                    legacy[key]={'probabilities':r['prob'],'source':str(source.relative_to(ROOT)),'sha256':core.digest(source)};break
    bundle={'models':models,'graphs':selected,'references':refs,'legacy_audit_only':legacy,'depths':list(DEPTHS),
            'protocol':core.PROTOCOL,'runtime':core.runtime(),'source_hashes':{k:{'path':os.path.relpath(p,ROOT),'sha256':core.digest(p)} for k,p in paths.items()}}
    bundle['campaign']=core.object_digest(bundle)
    core.atomic_json(OUT/'manifest.json',bundle)
    status('prepared',total=4200,legacy_audit_records=len(legacy),campaign=bundle['campaign'])
    return bundle

def load_manifest():
    b=json.loads((OUT/'manifest.json').read_text(encoding='utf-8'));copy=dict(b);copy.pop('campaign')
    if core.object_digest(copy)!=b['campaign']:raise ValueError('Manifest checksum mismatch')
    if b['runtime']!=core.runtime():raise ValueError('Runtime changed; do not mix environments in this campaign')
    for item in b['source_hashes'].values():
        if core.digest(ROOT/item['path'])!=item['sha256']:raise ValueError(f"Frozen source changed: {item['path']}")
    return b

def identity(b,g,p):
    angles={k:[x.tolist() for x in core.predict(g,p,k,b['models'])] for k in core.KINDS}
    return {'campaign':b['campaign'],'id':g['id'],'p':p,'fingerprint':g['fingerprint'],'backend':core.BACKEND,'angles_sha256':core.object_digest(angles)},angles

def check_record(b,g,p):
    ident,_=identity(b,g,p);r=core.read_record(path_for(g,p),ident);payload=dict(r);checksum=payload.pop('content_sha256',None)
    if checksum!=core.object_digest(payload):raise ValueError(f'Record checksum mismatch {taskkey(g,p)}')
    W=sum(g['weights']);ref=b['references'][taskkey(g,p)];legacy=b['legacy_audit_only'].get(taskkey(g,p))
    for k,m in r['methods'].items():
        if abs(m['energy']-float(ref['energy_'+k]))>1e-7*max(1,W):raise ValueError('Saved energy mismatch')
        if abs(m['ratio']-float(ref['ratio_'+k]))>1e-8:raise ValueError('Saved ratio mismatch')
        if legacy and abs(m['probability']-legacy['probabilities'][k])>1e-8:raise ValueError('Saved probability mismatch with legacy reference')
    return r

def inventory(b):
    found=[];missing=[];expected={}
    for g in b['graphs']:
        for p in DEPTHS:
            expected[path_for(g,p).name]=1
            if path_for(g,p).exists():found.append(check_record(b,g,p))
            else:missing.append(taskkey(g,p))
    extras=[p.name for p in (OUT/'records').glob('*.json') if p.name not in expected]
    if extras:raise ValueError(f'Unexpected production records: {extras[:3]}')
    return found,missing

def run(b,args):
    if (OUT/'STOP').exists():status('paused',reason='STOP exists; remove it explicitly before resume');return
    if args.graph_ids and not set(args.graph_ids)<={g['id'] for g in b['graphs']}:raise ValueError('Unknown graph id')
    graphs=[g for g in b['graphs'] if (not args.sizes or g['n'] in args.sizes) and (not args.graph_ids or g['id'] in args.graph_ids)]
    new=0;skipped=0;tic=time.perf_counter()
    status('running',new_records=0,campaign=b['campaign'])
    for g in graphs:
        v=mask=None
        for p in args.depths:
            if (OUT/'STOP').exists():status('paused',new_records=new,skipped=skipped,reason='STOP requested');return
            if path_for(g,p).exists():check_record(b,g,p);skipped+=1;continue
            if args.max_records is not None and new>=args.max_records:
                status('partial',new_records=new,skipped=skipped,reason='requested short-run limit',seconds=time.perf_counter()-tic);return
            start=time.perf_counter();spectrum_seconds=0.;W=sum(g['weights'])
            if v is None:
                t=time.perf_counter();v=core.spectrum(g);mask=core.optimal_mask(v,W);spectrum_seconds=time.perf_counter()-t
            ident,angles=identity(b,g,p);ref=b['references'][taskkey(g,p)];legacy=b['legacy_audit_only'].get(taskkey(g,p))
            record={**ident,'n':g['n'],'family':g['family'],'weight_kind':g['weight_kind'],'angles':angles,
                    'optimal_states':int(mask.sum()),'energy_min':float(v.min()),'energy_max':float(v.max()),
                    'dtype':'complex128','shots':None,'threads':4,'methods':{},'spectrum_seconds':spectrum_seconds,
                    'legacy_audit_source':legacy['source'] if legacy else None}
            for k in core.KINDS:
                t=time.perf_counter();node=core.make_state_function(g,p,k!='qaoa');psi=node(*angles[k]);m=core.measure(psi,v,mask)
                m['seconds']=time.perf_counter()-t;del psi,node
                m['energy_reference_error']=abs(m['energy']-float(ref['energy_'+k]))
                if m['energy_reference_error']>1e-7*max(1,W) or abs(m['ratio']-float(ref['ratio_'+k]))>1e-8:raise ValueError(f'Energy/ratio mismatch {taskkey(g,p)} {k}')
                if legacy:
                    m['legacy_probability_error']=abs(m['probability']-legacy['probabilities'][k])
                    if m['legacy_probability_error']>1e-8:raise ValueError(f'Probability mismatch {taskkey(g,p)} {k}')
                record['methods'][k]=m
            record['elapsed_seconds']=time.perf_counter()-start;record['content_sha256']=core.object_digest(record)
            core.atomic_json(path_for(g,p),record);check_record(b,g,p);new+=1
            status('running',new_records=new,skipped=skipped,last=taskkey(g,p),seconds=time.perf_counter()-tic,campaign=b['campaign'])
            print(json.dumps({'new':new,'skipped':skipped,'last':taskkey(g,p),'task_seconds':record['elapsed_seconds'],'seconds':time.perf_counter()-tic}),flush=True)
    found,missing=inventory(b)
    status('simulation_complete' if not missing else 'partial',valid_records=len(found),missing=len(missing),new_records=new,skipped=skipped,seconds=time.perf_counter()-tic)

def verify(b):
    records,missing=inventory(b)
    if missing:raise ValueError(f'Incomplete: {len(records)}/4200; plotting is blocked')
    report={'records':len(records),'graphs':1050,'methods':list(core.KINDS),'backend':core.BACKEND,'campaign':b['campaign'],
            'max_energy_reference_error':max(m['energy_reference_error'] for r in records for m in r['methods'].values()),
            'max_legacy_probability_error':max((m.get('legacy_probability_error',0.) for r in records for m in r['methods'].values()),default=0.)}
    core.atomic_json(OUT/'verification.json',report);return records

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=['prepare','run','status','verify','plot','pipeline'])
    parser.add_argument('--sizes',nargs='+',type=int,choices=[10,12,14,16,18,20,22]);parser.add_argument('--depths',nargs='+',type=int,choices=DEPTHS,default=list(DEPTHS))
    parser.add_argument('--graph-ids',nargs='+');parser.add_argument('--max-records',type=int)
    args=parser.parse_args()
    if args.max_records is not None and args.max_records<0:parser.error('max-records must be nonnegative')
    if args.command=='status':
        b=load_manifest();found,missing=inventory(b);print(json.dumps({'valid':len(found),'missing':len(missing),'status':json.loads((OUT/'status.json').read_text())},indent=2));return
    with CampaignLock():
        try:
            b=prepare() if args.command in ('prepare','pipeline') else load_manifest()
            if args.command=='prepare':print('PREPARED',b['campaign'],len(b['legacy_audit_only']));return
            if args.command in ('run','pipeline'):run(b,args)
            if args.command in ('verify','plot','pipeline'):
                if args.command=='pipeline' and (OUT/'STOP').exists():return
                records=verify(b)
                if args.command in ('plot','pipeline'):
                    import plots
                    plots.render(records,b);status('complete',valid_records=4200,campaign=b['campaign'])
                else:print('VERIFIED',len(records))
        except KeyboardInterrupt:
            status('interrupted',reason='KeyboardInterrupt; completed records retained');raise
        except Exception as e:
            status('blocked_error',error=str(e));raise

if __name__=='__main__':main()
