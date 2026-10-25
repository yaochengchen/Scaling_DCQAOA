"""Equal-depth pipeline end to end on Lightning CPU (same PennyLane device API as lightning.gpu)."""
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np
import pytest
pytest.importorskip('pennylane')
from cdtransfer.instances import make_instance, exact_extrema, graph_features
from cdtransfer.simulator import StatevectorSimulator
from cdtransfer.optimization import start_plan, interp_start, ramp_start
from cdtransfer.eqdepth import instance_plan, load_records

ROOT = Path(__file__).resolve().parents[1]


def _cfg(samples, depths=(1, 2), restarts=3):
    fam, w = ['regular3', 'er3'], ['uniform']
    return {'comparison': 'same_depth', 'seed': 5, 'depths': list(depths), 'shots': 0,
            'exact_max_n': 24,
            'execution': {'backend': 'lightning.qubit', 'dtype': 'complex128', 'gpus': '0,1',
                          'slots_per_gpu': 3, 'task_weight_by_n': {}},
            'optimization': {'restarts': restarts, 'maxiter': 25},
            'datasets': {'train': {'n': [6], 'families': fam, 'weights': w, 'samples_per_cell': samples},
                         'val': {'n': [6], 'families': fam, 'weights': w, 'samples_per_cell': 1},
                         'test': {'n': [6, 8], 'families': fam, 'weights': w, 'samples_per_cell': 1}}}


def _run(stage, cfg, out, tmp, *extra):
    path = tmp / f'{out.name}.json'
    path.write_text(json.dumps(cfg))
    env = {**os.environ, 'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1'}
    return subprocess.run([sys.executable, '-m', 'cdtransfer.cli', stage, '--config', str(path),
                           '--out', str(out), *extra], cwd=ROOT, env=env, check=True,
                          capture_output=True, text=True, timeout=600)


def test_exact_extrema_and_features():
    for fam in ('er05', 'bipartite', 'community', 'tree'):
        inst = make_instance(11, fam, 'heavy_tail', 3)
        e = StatevectorSimulator(inst).energy_vector
        lo, hi = exact_extrema(inst, chunk_bits=4)
        assert np.isclose(lo, e.min()) and np.isclose(hi, e.max())
    g = graph_features(make_instance(10, 'bipartite', 'unit', 1))
    assert g['is_bipartite'] and g['triangles'] == 0


def test_start_plan_and_interp():
    assert start_plan(3, False, 8, True)[:2] == ['ramp', 'interp']
    assert start_plan(1, False, 8, False)[:2] == ['ramp', 'random']
    assert start_plan(3, True, 8, True)[:2] == ['from_qaoa', 'interp']
    assert len(start_plan(4, True, 8, True)) == 8
    x = interp_start(np.r_[np.linspace(-1, -2, 2), [.5, .2], [.1, .3]], 2, 4, True)
    assert x.shape == (12,)
    assert ramp_start(make_instance(6, 'ring', 'unit', 1), 3, True).shape == (9,)


def test_pipeline_resume_and_reuse(tmp_path):
    out = tmp_path / 'run20'
    cfg = _cfg(2)
    _run('pipeline', cfg, out, tmp_path)
    plan = instance_plan(cfg)
    recs = load_records(out)
    assert len(recs) == 4 * len(plan)
    assert {r['init_style'] for r in recs} >= {'ramp', 'interp', 'from_qaoa', 'random'}
    # CD warm-started from the QAOA optimum: never meaningfully worse at the same depth
    best = {}
    for r in recs:
        k = (r['id'], r['p'], r['cd'])
        best[k] = min(best.get(k, np.inf), r['energy'])
    for (i, p, cd), e in best.items():
        if cd:
            assert e <= best[(i, p, False)] + 1e-7
    assert all(r['ratio'] is not None and 0 <= r['ratio'] <= 1 + 1e-9 for r in recs)
    test = json.loads((out / 'heldout_results.json').read_text())
    assert len(test) == 4 * 2 and all('energy_cd_v1' in r for r in test)
    assert all(r['ratio_cd_proposed'] is not None for r in test)
    rep = (out / 'EXPERIMENT_REPORT.md').read_text()
    assert 'same number of layers' in rep
    assert (out / 'figures' / 'fig01_graph_structures_same_layer.png').exists()
    assert (out / 'FROZEN_MODEL.sha256').exists()
    # resume: nothing recomputed
    before = {p: p.read_bytes() for p in (out / 'checkpoints').rglob('p*.json')}
    log = _run('train', cfg, out, tmp_path).stdout
    assert 'pending 0' in log
    assert before == {p: p.read_bytes() for p in (out / 'checkpoints').rglob('p*.json')}
    # grow the train set 2 -> 3 per cell and add depth 3: only new work is computed
    out2 = tmp_path / 'run30'
    cfg2 = _cfg(3, depths=(1, 2, 3))
    log = _run('train', cfg2, out2, tmp_path, '--reuse-from', str(out)).stdout
    imported = int(log.split('imported ')[1].split()[0])
    assert imported == 4 * len(plan)
    recs2 = load_records(out2)
    assert len(recs2) == 6 * len(instance_plan(cfg2))
    old = {(r['id'], r['p'], r['cd'], r['restart']): r['energy'] for r in recs}
    for r in recs2:
        k = (r['id'], r['p'], r['cd'], r['restart'])
        if k in old:
            assert r['energy'] == old[k]


def test_slot_weights_and_scheduler_packing():
    from cdtransfer.gpu_pool import resolve_execution, task_weight
    ex = resolve_execution({'execution': {'gpus': '0,1', 'slots_per_gpu': 4}}, {'task_weights': '22:3'})
    assert [task_weight(n, ex) for n in (18, 20, 22, 23, 24, 28)] == [1, 1, 3, 4, 4, 4]
    ex = resolve_execution({'execution': {'gpus': '0', 'slots_per_gpu': 1}}, {'slots_per_gpu': 2})
    assert ex['slots_per_gpu'] == 2 and task_weight(24, ex) == 2


def test_scheduler_never_oversubscribes_a_gpu():
    from cdtransfer.gpu_pool import resolve_execution, run_tasks, task_weight
    ex = resolve_execution({'execution': {'gpus': 'a,b', 'slots_per_gpu': 4,
                                          'task_weight_by_n': {'24': 4, '22': 2}}}, None)
    sizes = [24, 24, 22, 22, 22, 18, 18, 18, 18, 18, 18, 24]
    tasks = [{'n': n, 'seconds': 0.4} for n in sizes]
    done = [p for s, i, p in run_tasks(tasks, sizes, ex, 'cdtransfer.gpu_pool:_sleep_probe', {})]
    assert len(done) == len(tasks)
    for g in ('a', 'b'):
        mine = [d for d in done if d['gpu'] == g]
        for d in mine:
            t = (d['start'] + d['end']) / 2
            load = sum(task_weight(e['n'], ex) for e in mine if e['start'] <= t <= e['end'])
            assert load <= 4
    assert any(sum(1 for e in done if e['gpu'] == d['gpu'] and e['start'] < d['end'] and d['start'] < e['end']) >= 3
               for d in done if d['n'] == 18)   # small tasks really run concurrently on one GPU


def test_lock_reports_owner_pid(tmp_path):
    import fcntl
    from cdtransfer.eqdepth import train
    (tmp_path / 'checkpoints').mkdir()
    holder = (tmp_path / '.training.lock').open('a+')
    fcntl.flock(holder, fcntl.LOCK_EX | fcntl.LOCK_NB)
    holder.write('424242'); holder.flush()
    with pytest.raises(RuntimeError, match='424242'):
        train({'config': {}}, tmp_path)
    holder.close()


def test_reuse_from_several_sources_hardlinks_traces(tmp_path):
    a, b, c = tmp_path / 'roundA', tmp_path / 'roundB', tmp_path / 'roundC'
    _run('train', _cfg(1), a, tmp_path)
    _run('train', _cfg(2), b, tmp_path, '--reuse-from', str(a))
    log = _run('train', _cfg(3), c, tmp_path, '--reuse-from', f'{a},{b},{c}').stdout
    plan = len(instance_plan(_cfg(3)))
    assert f'imported {2 * plan} finished' in log      # from a
    assert f'imported {2 * plan} finished' in log.split(str(a))[1]   # the rest from b
    assert 'cost-weighted ETA' in log
    recs = load_records(c)
    assert len(recs) == 6 * plan
    imported = [r for r in recs if 'imported_from' in r]
    assert len(imported) == 4 * plan
    t = c / imported[0]['trace_path']
    assert t.exists() and t.stat().st_nlink >= 2


def test_round_configs_are_nested():
    import importlib.util
    spec = importlib.util.spec_from_file_location('rc', ROOT / 'scripts' / '23_round_config.py')
    rc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rc)
    from cdtransfer.instances import make_splits_stable
    for group, depths in (('small', [1, 2, 3, 4, 6]), ('large', [1, 2, 3, 4])):
        c20, c40 = rc.make_config(group, 20), rc.make_config(group, 40)
        assert c20['depths'] == depths and c20['execution']['dtype'] == 'complex64'
        assert c20['optimization']['ftol'] == 1e-6 and c20['optimization']['gtol'] == 1e-3
        small = {k: v for k, v in c20['datasets'].items()}
        small = {s: dict(v, n=v['n'][:1], families=['regular3'], weights=['unit']) for s, v in small.items()}
        big = {s: dict(v, n=v['n'][:1], families=['regular3'], weights=['unit'])
               for s, v in c40['datasets'].items()}
        s20 = make_splits_stable(dict(c20, datasets=small))
        s40 = make_splits_stable(dict(c40, datasets=big))
        f20 = {r['id']: r['fingerprint'] for r in s20['train']}
        f40 = {r['id']: r['fingerprint'] for r in s40['train']}
        assert len(f20) == 20 and len(f40) == 40 and all(f40[k] == v for k, v in f20.items())
        assert [r['fingerprint'] for r in s20['test']] == [r['fingerprint'] for r in s40['test']]


def test_complex64_run_rechecks_energies_in_complex128(tmp_path):
    from cdtransfer.lightning import LightningTapeSimulator
    from cdtransfer.simulator import Angles
    cfg = _cfg(1)
    cfg['execution']['dtype'] = 'complex64'
    cfg['optimization'].update(ftol=1e-6, gtol=1e-3)
    out = tmp_path / 'c64'
    _run('pipeline', cfg, out, tmp_path)
    recs = load_records(out)
    assert recs and all(r['dtype'] == 'complex64' and r['energy_dtype'] == 'complex128' for r in recs)
    assert all(r['ftol'] == 1e-6 and r['gtol'] == 1e-3 for r in recs)
    man = json.loads((out / 'manifest.json').read_text())
    rows = {r['id']: r for r in man['splits']['train']}
    from cdtransfer.eqdepth import _row_instance
    r = recs[-1]
    ref = LightningTapeSimulator(_row_instance(rows[r['id']]), device='lightning.qubit', dtype='complex128')
    e = ref.evaluate(Angles(**{k: np.asarray(v) for k, v in r['angles'].items()}))
    assert abs(e - r['energy']) < 1e-9
    assert abs(r['energy'] - r['energy_optimizer_dtype']) < 1e-3 * abs(r['energy']) + 1e-6
    assert (out / 'EXPERIMENT_REPORT.md').exists()
    # different tolerances -> nothing imported
    cfg2 = json.loads(json.dumps(cfg))
    cfg2['optimization']['ftol'] = 1e-7
    log = _run('train', cfg2, tmp_path / 'c64b', tmp_path, '--reuse-from', str(out)).stdout
    assert 'imported 0 finished' in log
