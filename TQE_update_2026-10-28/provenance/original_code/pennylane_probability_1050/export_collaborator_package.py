"""Create a compact, relocatable 1050-graph collaborator ZIP.

The live campaign is read only. Its per-record atomic files permit a coherent
snapshot while computation continues. No intermediate or unrelated project
products are copied.
"""
import hashlib
import json
import re
import shutil
import tempfile
import zipfile
from pathlib import Path

import core


STUDY = core.ROOT
PROJECT = core.PROJECT
OUTPUT = PROJECT / 'deliverables' / 'PennyLane_1050_collaborator_2026-10-27.zip'
LIVE = STUDY / 'results/pennylane_probability_1050'
PAPER = STUDY / 'manuscript/paper'


def copy_file(src, dst):
    if not src.is_file():
        raise FileNotFoundError(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def portable_lock(source):
    old = '''        import msvcrt
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
'''
    new = '''        OUT.mkdir(parents=True,exist_ok=True);self.f=(OUT/'run.lock').open('a+b')
        if self.f.seek(0,2)==0:self.f.write(b'0');self.f.flush()
        self.f.seek(0)
        try:
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(self.f.fileno(),msvcrt.LK_NBLCK,1)
            else:
                import fcntl
                fcntl.flock(self.f.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        except OSError:
            self.f.close();raise RuntimeError('Campaign already locked by another process')
        return self

    def __exit__(self,*args):
        if os.name == 'nt':
            import msvcrt
            self.f.seek(0);msvcrt.locking(self.f.fileno(),msvcrt.LK_UNLCK,1)
        else:
            import fcntl
            fcntl.flock(self.f.fileno(),fcntl.LOCK_UN)
        self.f.close()
'''
    if source.count(old) != 1:
        raise ValueError('Campaign lock source changed: review portability patch')
    return source.replace(old, new)


def main():
    source_manifest = json.loads((LIVE / 'manifest.json').read_text(encoding='utf-8'))
    without_id = dict(source_manifest)
    without_id.pop('campaign')
    if core.object_digest(without_id) != source_manifest['campaign']:
        raise ValueError('Live manifest checksum mismatch')
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='pennylane1050_export_') as temp:
        top = Path(temp)
        dst_study = top / 'study'
        dst_code = dst_study / 'code/pennylane_probability_1050'

        # Keep the frozen input layout used by campaign.prepare().
        for name, item in source_manifest['source_hashes'].items():
            src = (STUDY / item['path'].replace('\\', '/')).resolve()
            if core.digest(src) != item['sha256']:
                raise ValueError(f'Frozen input changed: {name}')
            if name == 'campaign_code':
                continue
            relative = Path(item['path'].replace('\\', '/'))
            copy_file(src, dst_study / relative)
        live_code = STUDY / 'code/pennylane_probability_1050'
        patched = portable_lock((live_code / 'campaign.py').read_text(encoding='utf-8'))
        (dst_code / 'campaign.py').write_text(patched, encoding='utf-8')
        for name in ('plots.py', 'import_snapshot.py', 'test_core.py', 'test_plot_smoke.py'):
            copy_file(live_code / name, dst_code / name)
        # The Windows launcher is optional; Python CLI works on both systems.
        copy_file(live_code / 'launch.ps1', dst_code / 'launch.ps1')
        copy_file(STUDY / 'reports/PennyLane_1050图成功概率_执行与交接规划.md',
                  dst_study / 'docs/PennyLane_1050图成功概率_执行与交接规划.md')

        copy_file(LIVE / 'manifest.json', dst_study / 'snapshot/source_manifest.json')
        count = 0
        for file in sorted((LIVE / 'records').glob('*.json')):
            record = json.loads(file.read_text(encoding='utf-8'))
            payload = dict(record)
            checksum = payload.pop('content_sha256', None)
            if core.object_digest(payload) != checksum:
                raise ValueError(f'Live record checksum mismatch: {file.name}')
            if record['campaign'] != source_manifest['campaign']:
                raise ValueError(f'Live record campaign mismatch: {file.name}')
            copy_file(file, dst_study / 'snapshot/records' / file.name)
            count += 1

        paper_src = PAPER / 'latex_preview_v4_merged'
        paper_dst = dst_study / 'manuscript/paper/latex_preview_v4_merged'
        for name in ('main.tex', 'abstract.tex', 'body.tex', 'supplementary.tex',
                     'main.pdf', 'supplementary.pdf', 'main.aux',
                     'ieeeaccess.cls', 'IEEEtran.cls', 'logo.png', 'notaglinelogo.png', 'bullet.png'):
            copy_file(paper_src / name, paper_dst / name)
        figures = set()
        for name in ('main.tex', 'body.tex', 'supplementary.tex'):
            text = (paper_src / name).read_text(encoding='utf-8')
            figures.update(re.findall(r'\\includegraphics(?:\[[^]]*\])?\{(\.\./figures/[^}]+)\}', text))
        for relative in figures:
            copy_file((paper_src / relative).resolve(), (paper_dst / relative).resolve())

        readme = f'''# PennyLane 1,050-graph handoff

This bundle contains the **current review manuscript**, frozen graph/model inputs,
the PennyLane simulation and plotting code, and a checksum-verified snapshot of
**{count:,} / 4,200 graph-depth records** captured from the author's running campaign.
The snapshot is a point-in-time copy; its original campaign and runtime are
recorded in `study/snapshot/source_manifest.json` and on each imported record.

**Scope:** 1,050 graphs, seven sizes (10–22 qubits), ten graph families, five
weight distributions, three graphs per cell; depths 1–4; U1 power law, preceding
rule, and frozen QAOA comparator. All new quantum circuits use PennyLane
`lightning.qubit`, complex128, shots=None, and four computation threads.
Do not refit or reselect after observing test success probabilities.

## Run on the collaborator's machine

From the extracted `study` directory, with Python 3.13 and the pinned packages
in `requirements.txt` installed:

```
python code/pennylane_probability_1050/campaign.py prepare
python code/pennylane_probability_1050/import_snapshot.py
python code/pennylane_probability_1050/campaign.py status
python code/pennylane_probability_1050/campaign.py pipeline
```

`prepare` makes a new manifest for the collaborator's Python/platform; import
verifies source hashes, graphs, models, reference energies, angles and every
snapshot record before adapting its campaign ID. The only code change from the
author's running copy is a Windows/Linux file-lock adaptation in `campaign.py`.
The simulator, parameter rules and plotting code are unchanged. `pipeline`
skips imported valid records, computes missing records, verifies all 4,200 and
draws review figures. Do not run two processes against one results folder.

On Linux, use a persistent session such as tmux to keep the command running.
On Windows, `launch.ps1 -Python <python-path>` can run the pipeline in the
background after prepare/import. The first run may take hours; completed JSON
records remain usable after interruption. Use `campaign.py status` for counts.
To pause between records, create `results/pennylane_probability_1050/STOP`,
wait for the process to exit, then remove only this STOP file before resuming.

If any checksum, energy or probability check fails, stop and send the exact
error and record name to the author; do not loosen validation thresholds.
Snapshot probabilities produced by Qulacs are used only as audit references
inside the frozen source manifest. Formal results are PennyLane records.

## Return to the author after the 1,050-graph run

Please send the author `results/pennylane_probability_1050/` (including all
records, manifest, verification, summary and paired CSV),
`figures/pennylane_probability_1050/` and
`reports/PennyLane_1050图成功概率_运行结果.md`, along with Python/package versions,
wall time and any errors. **Pause here for the author's review. Do not expand
to 7,000 graphs or change the manuscript based on these figures yet.**

The manuscript in `study/manuscript/paper/latex_preview_v4_merged/` includes
the latest main and supplementary PDFs, TeX sources, class files and referenced
figure PDFs. It is a review draft; current text and figures are not replaced
by this probability experiment. Build the main TeX before supplementary if
cross-references need regeneration.

The original Chinese experiment and plotting plan is in `study/docs/`. Its
machine-specific Python paths and initial record counts describe the author's
environment; use the commands and snapshot count in this README for this ZIP.
'''
        (top / 'README.md').write_text(readme, encoding='utf-8')
        (dst_study / 'requirements.txt').write_text(
            'numpy==2.3.5\npennylane==0.45.0\npennylane-lightning==0.45.0\nmatplotlib==3.10.6\n',
            encoding='utf-8')
        inventory = {}
        for file in sorted(top.rglob('*')):
            if file.is_file():
                inventory[file.relative_to(top).as_posix()] = hashlib.sha256(file.read_bytes()).hexdigest()
        (top / 'SHA256.json').write_text(json.dumps(inventory, indent=2, ensure_ascii=False), encoding='utf-8')
        with zipfile.ZipFile(OUTPUT, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
            for file in sorted(top.rglob('*')):
                if file.is_file():
                    archive.write(file, file.relative_to(top).as_posix())
    with zipfile.ZipFile(OUTPUT) as archive:
        if archive.testzip() is not None:
            raise ValueError('ZIP CRC validation failed')
        names = archive.namelist()
        if sum(name.startswith('study/snapshot/records/') and name.endswith('.json') for name in names) != count:
            raise ValueError('Snapshot count changed in archive')
    print(json.dumps({'zip':str(OUTPUT), 'snapshot_records':count,
                      'remaining_at_snapshot':4200-count, 'files':len(names),
                      'bytes':OUTPUT.stat().st_size}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
