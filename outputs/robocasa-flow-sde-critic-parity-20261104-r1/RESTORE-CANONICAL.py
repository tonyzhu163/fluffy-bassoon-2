"""Restore exact archived canonical bytes into a new private recovery input."""
import argparse
import hashlib
import json
from pathlib import Path

from PREPARE import ClosedArchive, file_sha


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    root = args.root
    source = Path('/mowenjie/fluffy-bassoon-2/outputs/robocasa-flow-sde-10m-per-task-20261103-r1')
    archive = ClosedArchive(source, source / 'CORPUS-ARCHIVE-PickPlaceDrawerToCounter-R1-RECLAIMED.json')
    output = root / 'comparison-input-r2'
    output.mkdir(exist_ok=False)
    receipts = []
    for rank in range(8):
        directory = output / 'canonical-eval' / f'rank-{rank:02d}'
        directory.mkdir(parents=True)
        for filename in ['initial.pt', 'complete.json', 'owner.json']:
            member = f'PickPlaceDrawerToCounter/q1-r1/canonical-eval/rank-{rank:02d}/{filename}'
            item = archive.files[member]
            archive.stream.seek(archive.offsets[member])
            data = archive.stream.read(item['bytes'])
            assert len(data) == item['bytes'] and hashlib.sha256(data).hexdigest() == item['sha256']
            target = directory / filename
            target.write_bytes(data)
            receipts.append(dict(member=member, sha256=file_sha(target), bytes=len(data), target=str(target)))
        marker = json.loads((directory / 'complete.json').read_text())
        assert marker['snapshot_sha256'] == file_sha(directory / 'initial.pt')
        assert marker['case']['seeds'] == list(range(630100 + 8 * rank, 630108 + 8 * rank))
    original_reference = root / 'comparison-input-r1/BEFORE-REFERENCE.json'
    reference = json.loads(original_reference.read_text())
    reference['canonical_directory'] = str(output / 'canonical-eval')
    reference['comparison_recovery'] = 'All24 canonical members restored byte-exact from the closed corpus; completed baseline training is retained.'
    path = output / 'BEFORE-REFERENCE.json'
    path.write_text(json.dumps(reference, indent=2) + '\n')
    result = dict(completed=True, files=receipts, canonical_ranks=8, canonical_snapshot_bytes=sum(
        row['bytes'] for row in receipts if row['member'].endswith('/initial.pt')),
        archive_manifest_sha256=file_sha(archive.manifest_path), before_reference=str(path),
        before_reference_sha256=file_sha(path), checkpoint=reference['checkpoint'],
        original_reference_sha256=file_sha(original_reference), script_sha256=file_sha(__file__),
        shared_source_and_previous_attempt_unchanged=True, new_controls=0, SGD=0)
    (output / 'RESTORED.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(completed=True, members=len(receipts), canonical_ranks=8)))


if __name__ == '__main__':
    main()
