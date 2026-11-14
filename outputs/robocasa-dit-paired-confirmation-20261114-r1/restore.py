"""Restore only historical inference inputs from verified private archives."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tarfile
import time


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def extract(archive, targets, root):
    found = set()
    process = subprocess.Popen(['zstd', '-dc', str(archive)], stdout=subprocess.PIPE)
    try:
        with tarfile.open(fileobj=process.stdout, mode='r|') as bank:
            for member in bank:
                name = member.name.removeprefix('./')
                if name not in targets:
                    continue
                if not member.isfile():
                    raise ValueError(f'Required inference member is not a regular file: {name}')
                destination = root / targets[name]
                destination.parent.mkdir(parents=True, exist_ok=True)
                with bank.extractfile(member) as source, destination.open('xb') as output:
                    shutil.copyfileobj(source, output, 16 * 2**20)
                if destination.stat().st_size != member.size:
                    raise ValueError('Incomplete inference input')
                found.add(name)
        # Consume the remainder so decompression verifies its frame checksum.
        while process.stdout.read(16 * 2**20):
            pass
        if process.wait() != 0 or found != set(targets):
            raise ValueError(f'Archive restore incomplete: {set(targets) - found}')
    finally:
        process.stdout.close()
        if process.poll() is None:
            process.terminate()
            process.wait()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--staging', type=Path, required=True)
    p.add_argument('--cold', type=Path, required=True)
    args = p.parse_args()
    spec = json.loads(Path(__file__).with_name('INFERENCE-INPUTS.json').read_text())
    receipts = json.loads(Path(__file__).parents[1].joinpath(
        'robocasa-six-task-scale-20261113-r1/evidence/HISTORICAL-ARCHIVE-20261114.json').read_text())
    args.staging.mkdir(parents=True, exist_ok=False)
    args.cold.mkdir(parents=True, exist_ok=False)
    assert shutil.disk_usage(args.staging).free > 32 * 2**30
    started = time.time()
    wanted = {}
    for name, row in spec['DIT'].items():
        member = row['path'].removeprefix('/mowenjie/fluffy-bassoon-2/')
        wanted[member] = 'DIT/' + name
    for task, row in spec['PPO'].items():
        prefix = row['path'].removeprefix('/mowenjie/fluffy-bassoon-2/')
        wanted[prefix + '/pilot-boundary.json'] = f'PPO/{task}/pilot-boundary.json'
        wanted[prefix + '/actor/model_state_dict/full_weights.pt'] = f'PPO/{task}/full_weights.pt'
    archives = []
    for receipt in receipts['completed']:
        selected = {key: value for key, value in wanted.items()
                    if key.startswith(receipt['source'].removeprefix('/mowenjie/fluffy-bassoon-2/') + '/')}
        if not selected:
            continue
        archive = Path(receipt['archive'])
        if archive.stat().st_size != receipt['archive_bytes'] or sha(archive) != receipt['sha256']:
            raise ValueError('Cold archive differs from its verified receipt')
        extract(archive, selected, args.staging)
        archives.append(dict(path=str(archive), sha256=receipt['sha256']))
        print(json.dumps(dict(restored_archive=str(archive), members=len(selected))), flush=True)
    identities = {}
    for name, row in spec['DIT'].items():
        identities['DIT/' + name] = row['sha256']
    for task, row in spec['PPO'].items():
        boundary = args.staging / f'PPO/{task}/pilot-boundary.json'
        if sha(boundary) != row['boundary_sha256']:
            raise ValueError('Historical fixed PPO boundary changed')
        record = json.loads(boundary.read_text())
        if not record['completed'] or record['global_step'] != 130:
            raise ValueError('Require complete historical PPO130')
        weights = next(x for x in record['actor_checkpoint_files']
                       if x['path'] == row['path'] + '/actor/model_state_dict/full_weights.pt')
        identities[f'PPO/{task}/pilot-boundary.json'] = row['boundary_sha256']
        identities[f'PPO/{task}/full_weights.pt'] = weights['sha256']
    records = {}
    for name, digest in identities.items():
        source, destination = args.staging / name, args.cold / name
        if sha(source) != digest:
            raise ValueError('Restored input differs from historical inference identity')
        destination.parent.mkdir(parents=True, exist_ok=True)
        with source.open('rb') as reader, destination.open('xb') as writer:
            shutil.copyfileobj(reader, writer, 16 * 2**20)
        if sha(destination) != digest:
            raise ValueError('Inference copy failed complete cold readback')
        records[name] = dict(path=str(destination), sha256=digest, bytes=destination.stat().st_size)
        source.unlink()
    result = dict(completed=True, source_commit=subprocess.check_output(
        ['git', '-C', str(Path(__file__).parents[2]), 'rev-parse', 'HEAD'], text=True).strip(),
        inputs=records, archives=archives, old_corpora_restored=False,
        local_staging_payload_reclaimed=True, elapsed_seconds=time.time()-started)
    write(args.cold / 'RESTORED.json', result)
    write(args.staging / 'RESTORED.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
