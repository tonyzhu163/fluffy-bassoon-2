"""Run-local Linux qualification of deletion boundaries and retained records."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import threading

BASE = Path('/mowenjie/fluffy-bassoon-2/outputs/robocasa-retention-audit-20261105-r1')


def load(name):
    spec = importlib.util.spec_from_file_location(name, BASE / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def row(path, root, category, reason):
    s = path.lstat()
    return [str(path.relative_to(root)), s.st_mode, s.st_size, s.st_mtime_ns,
            s.st_blocks, s.st_dev, s.st_ino, category, reason]


def main():
    audit, prune = load('audit'), load('prune')
    with tempfile.TemporaryDirectory(prefix='retention-prune-qualification-') as temp:
        work = Path(temp)
        source = work / 'source'
        outside = work / 'outside'
        source.mkdir(); outside.mkdir()
        (source / 'corpus').mkdir()
        (source / 'RESULT.md').write_text('keep this result\n')
        (source / 'summary.json').write_text('{"success": 7}\n')
        (source / 'metrics.npz').write_bytes(b'keep numerical result')
        (source / 'unknown.py').write_text('def reusable(): return 7\n')
        (outside / 'shared.pt').write_bytes(b'shared immutable payload')
        (source / 'escape').symlink_to(outside, target_is_directory=True)
        os.link(outside / 'shared.pt', source / 'corpus/shared.pt')
        (source / 'corpus/chunk.npz').write_bytes(b'obsolete raw chunk')
        (source / 'old.sh').write_text('#!/bin/sh\nexit 0\n')
        keep = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in source.iterdir() if p.is_file() and p.name != 'old.sh'}
        directories = {'.': row(source, source, 'keep', None), 'corpus': row(source / 'corpus', source, 'keep', None)}
        rootfd = os.open(source, os.O_RDONLY | os.O_DIRECTORY)
        bulk = [row(p, source, 'bulk', audit.bulk_reason(p.relative_to(source), p.stat().st_size))
                for p in (source / 'corpus/shared.pt', source / 'corpus/chunk.npz')]
        result = prune.delete_group(str(source), rootfd, 'corpus', bulk, directories,
                                    threading.Event(), set(), audit.bulk_reason)
        assert result['files'] == 2 and not result['preserved']
        assert (outside / 'shared.pt').read_bytes() == b'shared immutable payload'
        assert audit.bulk_reason('metrics.npz', 20) is None
        assert audit.bulk_reason('site-packages/project.pth', 20) is None
        assert audit.bulk_reason('RESULT.md', 20) is None
        assert audit.bulk_reason('experiment/raw/request.pkl.gz', 5000)
        assert audit.bulk_reason('experiment/traces/episode-001.pkl.gz', 5000)
        assert audit.bulk_reason('thin-evidence.tar.gz', 5000) is None
        assert audit.bulk_reason('console.log.gz', 5000) is None
        assert audit.bulk_reason('allocation/cache/vllm/torch_compile_cache/abc/artifact', 100) == 'regenerable run-local compiler cache'
        assert audit.bulk_reason('banks/bank-1/environment/rank-00/test-chunk01.pt.gz', 100) == 'retired compressed evaluation environment chunk'
        for name, digest in keep.items():
            assert hashlib.sha256((source / name).read_bytes()).hexdigest() == digest
        old = source / 'old.sh'
        data = old.read_bytes(); blob = hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest()
        result = prune.delete_group(str(source), rootfd, '.', [row(old, source, 'one_off_code', 'pinned Git copy')],
                                    directories, threading.Event(), {blob}, audit.bulk_reason)
        assert result['files'] == 1 and not old.exists()
        changed = source / 'changed.pt'; changed.write_bytes(b'before')
        record = row(changed, source, 'bulk', audit.bulk_reason('changed.pt', changed.stat().st_size))
        changed.write_bytes(b'after modification')
        result = prune.delete_group(str(source), rootfd, '.', [record], directories, threading.Event(), set(), audit.bulk_reason)
        assert not result['files'] and result['preserved'] and changed.exists()
        fake = row(source / 'RESULT.md', source, 'bulk', 'fabricated class')
        result = prune.delete_group(str(source), rootfd, '.', [fake], directories, threading.Event(), set(), audit.bulk_reason)
        assert not result['files'] and result['preserved'] and (source / 'RESULT.md').exists()
        try:
            prune.open_parent(rootfd, 'escape', directories)
        except (OSError, KeyError, AssertionError):
            pass
        else:
            raise AssertionError('Symlink traversal was accepted')
        os.close(rootfd)
    result = {'passed': True, 'bulk_only_deletion': True, 'results_configs_unknown_code_unchanged': True,
              'small_numeric_results_and_pth_config_preserved': True, 'external_hardlink_payload_preserved': True,
              'symlink_escape_rejected': True, 'modified_file_preserved': True,
              'unreviewed_payload_class_rejected': True, 'code_requires_pinned_git_blob': True,
              'GPU_jobs_launched': 0}
    (BASE / 'PRUNE-QUALIFICATION.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
