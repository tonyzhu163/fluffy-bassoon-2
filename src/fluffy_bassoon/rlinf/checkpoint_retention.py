"""Bound local recovery copies without weakening complete-boundary recovery."""
import hashlib
import json
from pathlib import Path
import re
import shutil


def _sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def retain_boundaries(root, *, current_step, keep_latest, protected_steps=()):
    """Prune only complete, inventoried checkpoints after checking recovery copies.

    Step zero and explicit evaluation/teacher checkpoints always survive. Unknown
    or incomplete directories are left alone. This is opt-in and run-local;
    callers must not point at shared input checkpoints.
    """
    root = Path(root)
    if keep_latest < 2:
        raise ValueError('Keep at least two complete recovery boundaries')
    if root.is_symlink() or root.resolve() != root.absolute():
        raise ValueError('Checkpoint root must be a canonical local directory')
    complete = {}
    for path in root.iterdir():
        match = re.fullmatch(r'global_step_(\d+)', path.name)
        if not match or not path.is_dir() or path.is_symlink():
            continue
        receipt = path / 'pilot-boundary.json'
        if not receipt.is_file() or receipt.is_symlink():
            continue
        data = json.loads(receipt.read_text())
        step = int(match[1])
        if data.get('completed') is True and data.get('global_step') == step:
            complete[step] = (path, data)
    if current_step not in complete or max(complete) != current_step:
        raise ValueError('Current boundary must be the newest complete checkpoint')
    retained = set(sorted(complete)[-keep_latest:]) | {0} | set(protected_steps)
    candidates = sorted(set(complete) - retained)
    if not candidates:
        return []

    def verify(step):
        path, data = complete[step]
        rows = data['actor_checkpoint_files'] + data['workers']
        if not data['actor_checkpoint_files'] or not data['workers']:
            raise ValueError('Recovery requires actor and worker state')
        expected = set()
        for row in rows:
            file = Path(row['path'])
            if (not file.is_absolute() or not file.is_relative_to(path)
                    or file.resolve() != file or file.is_symlink()):
                raise ValueError('Checkpoint inventory escapes its own boundary')
            if file in expected:
                raise ValueError('Duplicate checkpoint inventory member')
            expected.add(file)
            if (('bytes' in row and file.stat().st_size != row['bytes'])
                    or _sha(file) != row['sha256']):
                raise ValueError('Checkpoint recovery hash mismatch')
        actual = {p for p in path.rglob('*') if p.is_file()}
        if any(p.is_symlink() for p in path.rglob('*')) or actual != expected | {path / 'pilot-boundary.json'}:
            raise ValueError('Uninventoried checkpoint contents must be preserved')

    # A damaged newest checkpoint must never cause removal of a valid fallback.
    for step in sorted(set(candidates) | set(sorted(complete)[-keep_latest:])):
        verify(step)
    receipts = root / 'retired-boundaries'
    receipts.mkdir(exist_ok=True)
    removed = []
    for step in candidates:
        path, data = complete[step]
        record = dict(boundary=data, current_step=current_step,
                      retained_steps=sorted(set(complete) & retained),
                      reason='Superseded complete recovery copy; raw histories untouched',
                      removed=False)
        receipt = receipts / f'global_step_{step}.json'
        if receipt.exists():
            raise ValueError('Existing retirement receipt requires reconciliation')
        receipt.write_text(json.dumps(record, indent=2) + '\n')
        shutil.rmtree(path)
        record['removed'] = True
        temporary = receipt.with_suffix('.tmp')
        temporary.write_text(json.dumps(record, indent=2) + '\n')
        temporary.replace(receipt)
        removed.append(step)
    return removed
