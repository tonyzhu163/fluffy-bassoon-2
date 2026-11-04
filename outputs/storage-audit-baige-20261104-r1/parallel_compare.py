"""Partition a GNU tar comparison into exact, nonrecursive member lists."""
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import tempfile
import time


def run_comparison(archive, root, names, audit, workers=8):
    audit.mkdir(exist_ok=False)
    assert len(names) == len(set(names)) and names
    for name in names:
        assert not name.startswith('/') and '..' not in Path(name).parts
    parts = [names[i::workers] for i in range(workers)]
    assert sorted(x for part in parts for x in part) == sorted(names)
    def compare(i):
        members = audit / f'part-{i}.names'
        members.write_bytes(b''.join(os.fsencode(n) + b'\0' for n in parts[i]))
        cmd = ['tar', '-I', 'zstd', '-df', str(archive), '-C', str(root),
               '--no-recursion', '--no-wildcards', '--null',
               '--verbatim-files-from', '-T', str(members)]
        with (audit / f'part-{i}.log').open('wb') as out:
            code = subprocess.call(cmd, stdout=out, stderr=out)
        return {'part': i, 'members': len(parts[i]), 'returncode': code}
    begin = time.monotonic()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(compare, range(workers)))
    receipt = {'members': len(names), 'workers': workers, 'results': results,
               'seconds': time.monotonic()-begin,
               'all_passed': all(x['returncode'] == 0 for x in results)}
    (audit/'RESULT.json').write_text(json.dumps(receipt, indent=2)+'\n')
    return receipt


def archive_names(archive):
    # Read the whole local archive. GNU tar still performs all source comparisons.
    proc = subprocess.Popen(['zstd', '-dc', str(archive)], stdout=subprocess.PIPE)
    names = []
    try:
        with tarfile.open(fileobj=proc.stdout, mode='r|') as stream:
            for member in stream:
                names.append(member.name.rstrip('/'))
        # Drain any trailing tar padding before checking decoder completion.
        while proc.stdout.read(1024*1024):
            pass
        assert proc.wait() == 0
    finally:
        proc.stdout.close()
        if proc.poll() is None:
            proc.terminate()
            proc.wait()
    assert len(names) == len(set(names)), 'Duplicate archive member'
    return names


def qualify(audit):
    audit.mkdir(exist_ok=False)
    work = Path(tempfile.mkdtemp(prefix='baige-compare-qualification-'))
    try:
        source = work/'fixture'
        (source/'nested').mkdir(parents=True)
        for i in range(24):
            (source/'nested'/f'file {i}').write_bytes(bytes([i])*123)
        (source/'nested'/'line\nbreak[*]').write_bytes(b'literal odd name\n')
        (source/'link').symlink_to('nested/file 0')
        os.link(source/'nested'/'file 1', source/'hardlink')
        archive = work/'fixture.tar.zst'
        subprocess.run(['tar', '--sort=name', '-I', 'zstd', '-cf', str(archive),
                        '-C', str(work), 'fixture'], check=True)
        names = archive_names(archive)
        expected = ['fixture'] + [str(p.relative_to(work)) for p in source.rglob('*')]
        assert sorted(names) == sorted(expected)
        cases=[]
        for case in ('unchanged', 'content_same_size_mtime', 'missing_file',
                     'symlink_target', 'mode', 'mtime'):
            if case != 'unchanged':
                shutil.rmtree(source)
                subprocess.run(['tar', '-I', 'zstd', '-xf', str(archive),
                                '-C', str(work)], check=True)
            target=source/'nested'/'file 7'
            s=target.stat()
            if case=='content_same_size_mtime':
                target.write_bytes(b'x'*s.st_size)
                os.utime(target, ns=(s.st_atime_ns,s.st_mtime_ns))
            elif case=='missing_file': target.unlink()
            elif case=='symlink_target':
                (source/'link').unlink(); (source/'link').symlink_to('nested/file 2')
            elif case=='mode': target.chmod(0o600)
            elif case=='mtime': os.utime(target, ns=(s.st_atime_ns,s.st_mtime_ns+10**9))
            with (audit/(case+'-reference.log')).open('wb') as out:
                reference=subprocess.call(['tar','-I','zstd','-df',str(archive),
                                           '-C',str(work)],stdout=out,stderr=out)
            parallel=run_comparison(archive,work,names,audit/case,4)
            assert (reference==0)==parallel['all_passed']==(case=='unchanged')
            cases.append({'case':case,'reference_returncode':reference,
                          'parallel_all_passed':parallel['all_passed']})
        result={'cases':cases,'qualified':True,'exact_member_partition':True,
                'literal_names_with_spaces_newline_and_globs':True,
                'hardlink_and_symlink_fixture':True,
                'finished_at':time.strftime('%Y-%m-%dT%H:%M:%S%z')}
        (audit/'QUALIFICATION.json').write_text(json.dumps(result,indent=2)+'\n')
        return result
    finally:
        shutil.rmtree(work)


if __name__=='__main__':
    import sys
    print(json.dumps(qualify(Path(sys.argv[1])),indent=2),flush=True)
