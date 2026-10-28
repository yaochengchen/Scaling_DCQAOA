"""Fill in the files omitted from this folder because Clean already has identical copies.

This folder holds only the files of TQE_reproducibility_2026-10-28 that were new relative
to Clean (manuscript excluded). Run this script once to copy the 50 identical files from their Clean locations
and extract the 720 optimizer trajectories from runs.zip, so that reproduce.py /
refit_parameters.py / rerun_*.py find everything at their package-relative paths.

Usage (from anywhere):
    python restore_from_clean.py                 # Clean root = parent of this folder
    python restore_from_clean.py --clean D:/software/Jiapei/CD_QAOA_FIRST/Clean
    python restore_from_clean.py --check         # only report, copy nothing

Every copied file is checked against SHA256.json. Existing files are never overwritten.
"""
from pathlib import Path
import argparse, hashlib, json, shutil, sys, zipfile

HERE = Path(__file__).resolve().parent


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--clean', type=Path, default=HERE.parent)
    ap.add_argument('--check', action='store_true')
    a = ap.parse_args()
    clean = a.clean.resolve()
    expected = json.loads((HERE / 'SHA256.json').read_text(encoding='utf-8'))
    m = json.loads((HERE / 'IDENTICAL_IN_CLEAN.json').read_text(encoding='utf-8'))

    copied = skipped = 0
    errors = []
    for rel, info in m['files'].items():
        dst = HERE / rel
        if dst.exists():
            if sha(dst.read_bytes()) != expected[rel]:
                errors.append(f'existing file differs, left untouched: {rel}')
            skipped += 1
            continue
        src = clean / info['clean_path']
        if not src.exists():
            errors.append(f'missing in Clean: {info["clean_path"]}')
            continue
        data = src.read_bytes()
        if sha(data) != expected[rel]:
            errors.append(f'hash mismatch in Clean copy: {info["clean_path"]}')
            continue
        if not a.check:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
        copied += 1

    for prefix, info in m['zip_members'].items():
        zpath = clean / info['zip']
        if not zpath.exists():
            errors.append(f'missing zip: {info["zip"]}')
            continue
        with zipfile.ZipFile(zpath) as z:
            for name in z.namelist():
                if name.endswith('/'):
                    continue
                rel = prefix + name
                dst = HERE / rel
                if rel not in expected:
                    continue
                if dst.exists():
                    skipped += 1
                    continue
                data = z.read(name)
                if sha(data) != expected[rel]:
                    errors.append(f'hash mismatch in zip member: {name}')
                    continue
                if not a.check:
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    dst.write_bytes(data)
                copied += 1

    # manuscript/ is intentionally not part of this folder
    missing = [f for f in expected if not (HERE / f).exists() and not f.startswith('manuscript/')]
    verb = 'would copy' if a.check else 'copied'
    print(f'{verb}: {copied}, already present: {skipped}')
    if a.check:
        print('(check mode: nothing written)')
    else:
        print(f'files listed in SHA256.json still missing: {len(missing)}')
        for f in missing[:20]:
            print('  ', f)
    for e in errors:
        print('ERROR:', e)
    sys.exit(1 if errors else 0)


if __name__ == '__main__':
    main()
