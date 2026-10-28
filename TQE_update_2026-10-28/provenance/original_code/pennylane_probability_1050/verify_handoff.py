"""Short resume/STOP checks; never computes a new circuit record."""
import json,subprocess,sys
from pathlib import Path
import core,campaign

if __name__=='__main__':
    b=campaign.load_manifest();records,missing=campaign.inventory(b)
    before={p.name:core.digest(p) for p in (campaign.OUT/'records').glob('*.json')}
    command=[sys.executable,'-X','utf8',str(Path(campaign.__file__))]
    subprocess.run(command+['run','--sizes','10','--depths','1','4','--max-records','0'],check=True,capture_output=True)
    resumed=json.loads((campaign.OUT/'status.json').read_text())
    assert resumed['new_records']==0 and resumed['skipped']>=2
    stop=campaign.OUT/'STOP'
    if stop.exists():raise RuntimeError('Existing user STOP request; do not overwrite')
    try:
        stop.write_text('short handoff verification',encoding='utf-8')
        subprocess.run(command+['run','--max-records','0'],check=True,capture_output=True)
        assert json.loads((campaign.OUT/'status.json').read_text())['stage']=='paused'
    finally:stop.unlink()
    after={p.name:core.digest(p) for p in (campaign.OUT/'records').glob('*.json')}
    assert before==after
    try:campaign.verify(b)
    except ValueError as e:assert 'Incomplete' in str(e)
    else:raise AssertionError('Expected partial production set at handoff')
    result={'campaign':b['campaign'],'valid_pennylane_records':len(records),'remaining_records':len(missing),
            'resume_skipped_completed':resumed['skipped'],'resume_new_records':0,'record_hashes_unchanged':True,
            'stop_gate_passed':True,'incomplete_plot_gate_passed':True,'no_long_run_started':True,
            'max_energy_reference_error':max(m['energy_reference_error'] for r in records for m in r['methods'].values()),
            'max_legacy_probability_error':max(m.get('legacy_probability_error',0.) for r in records for m in r['methods'].values()),
            'timing_records':[{'id':r['id'],'n':r['n'],'p':r['p'],'seconds':r['elapsed_seconds']} for r in records]}
    core.atomic_json(campaign.OUT/'handoff_verification.json',result)
    campaign.status('ready_for_luna',valid_records=len(records),missing=len(missing),campaign=b['campaign'])
    print(json.dumps(result,indent=2))
