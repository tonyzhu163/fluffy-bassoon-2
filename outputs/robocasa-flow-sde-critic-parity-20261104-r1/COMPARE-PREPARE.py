"""Restore and hash-qualify a private full policy-0 comparison checkpoint."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tarfile

import zstandard


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(16 << 20), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    source = Path('/mowenjie/fluffy-bassoon-2/outputs/robocasa-flow-sde-10m-per-task-20261103-r1/PickPlaceDrawerToCounter')
    original = source / 'q1-r1/PickPlaceDrawerToCounter/checkpoints/global_step_0'
    cold = json.loads((original / 'COLD-STORAGE.json').read_text())
    assert cold['full_BOS_readback'] and cold['tar_compare']
    assert cold['sha256'] == '559edaf5354212798dc64202fca5f846dd9dbfbcea30ea5c975631d2366ac94d'
    assert shutil.disk_usage(args.root).free >= 240 * (1 << 30)
    output = args.root / 'comparison-input-r1'
    output.mkdir(exist_ok=False)
    with Path(cold['archive']).open('rb') as compressed:
        with zstandard.ZstdDecompressor().stream_reader(compressed) as stream:
            with tarfile.open(fileobj=stream, mode='r|') as archive:
                for member in archive:
                    name = Path(member.name)
                    assert not name.is_absolute() and '..' not in name.parts
                    assert name.parts[0] == 'global_step_0'
                    target = output / name
                    if member.isdir():
                        target.mkdir(parents=True, exist_ok=True)
                    else:
                        assert member.isfile()
                        target.parent.mkdir(parents=True, exist_ok=True)
                        with target.open('xb') as destination:
                            shutil.copyfileobj(archive.extractfile(member), destination, 16 << 20)
    checkpoint = output / 'global_step_0'
    marker = checkpoint / 'pilot-boundary.json'
    assert sha(marker) == sha(original / 'pilot-boundary.json')
    boundary = json.loads(marker.read_text())
    assert boundary['completed'] and boundary['global_step'] == 0 and len(boundary['workers']) == 18
    verified = []
    for row in boundary['actor_checkpoint_files'] + boundary['workers']:
        relative = Path(row['path']).relative_to(original)
        target = checkpoint / relative
        assert sha(target) == row['sha256']
        verified.append(dict(path=str(relative), sha256=row['sha256'], bytes=target.stat().st_size))
    reference_path = source / 'BEFORE-RESUME.json'
    reference = json.loads(reference_path.read_text())
    assert reference['completed'] and reference['cases'] == 64 and reference['metrics']['success_once'] == .5
    for row in reference['episode_refs']:
        assert sha(row['path']) == row['sha256']
    reference.update(checkpoint=str(checkpoint), checkpoint_boundary_sha256=sha(marker),
        comparison_reuse_scope='Same policy-0 before panel; original reference named the later step-1 resume boundary.')
    own_reference = output / 'BEFORE-REFERENCE.json'
    own_reference.write_text(json.dumps(reference, indent=2) + '\n')
    receipt = dict(completed=True, checkpoint=str(checkpoint), files=verified,
        marker_sha256=sha(marker), original_cold_receipt_sha256=sha(original / 'COLD-STORAGE.json'),
        archive=str(cold['archive']), source_archive_previously_verified_sha256=cold['sha256'],
        verification='Every full model/DCP/optimizer and worker-sidecar member rehashed against original boundary.',
        before_reference=str(own_reference), before_reference_sha256=sha(own_reference),
        original_before_reference_sha256=sha(reference_path), new_controls=0, SGD=0,
        script_sha256=sha(__file__))
    (output / 'PREPARED.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(dict(completed=True, checkpoint=str(checkpoint), verified_files=len(verified))))


if __name__ == '__main__':
    main()
