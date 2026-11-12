"""Build separate analysis inputs from complete, independently audited banks.

Interrupted workers can leave every bank durable without a group-level receipt.
No source artifact is changed. The original COMPARE.py performs the full panel
validation after this mechanical reconstruction; no outcomes select inclusion.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
TASKS = dict(ID=('OpenDrawer', 'CloseToasterOvenDoor', 'CoffeeSetupMug'),
             OOD=('CloseFridge', 'TurnOnMicrowave'))


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def paths(group, split):
    return [group / mode / task / f'bank-{first:03d}'
            for mode in ('full', 'episode') for task in TASKS[split]
            for first in range(0, 50, 8)]


def recover_group(source, fit, target, method, split):
    assert not target.exists()
    banks = paths(source, split)
    assert all((p/'RESULT.json').exists() for p in banks)
    binding_path = source/'BINDING.json'
    binding = read(binding_path if binding_path.exists() else source.parent/'ID/BINDING.json')
    binding['split'] = split
    target.mkdir(parents=True)
    write(target/'BINDING.json', binding)
    binding_sha = sha(target/'BINDING.json')
    candidate = fit/'kv_bridge'/method/'candidate.pt'
    assert sha(candidate) == binding['selected_candidate_sha256']
    shutil.copyfile(candidate, target/'selected-candidate.pt')
    summaries, evidence = [], []
    for bank in banks:
        raw, audit, marker = (read(bank/name) for name in ('RESULT.json','AUDIT.json','PACKED-ARCHIVE.json'))
        digest = sha(bank/'RESULT.json')
        assert raw['binding_sha256'] == binding_sha
        assert audit['passed'] and marker['verified']
        assert audit['result_sha256'] == marker['result_sha256'] == digest
        assert raw['method'] == method and raw['optimizer_updates'] == 0
        dest = target/bank.relative_to(source)
        dest.mkdir(parents=True)
        for name in ('RESULT.json','AUDIT.json','PACKED-ARCHIVE.json'):
            shutil.copyfile(bank/name, dest/name)
            assert sha(bank/name) == sha(dest/name)
        summaries.append({k:v for k,v in raw.items()
                          if k not in ('episodes','references','cases','actual_batch_sizes')})
        evidence.append(dict(source=str(bank), result_sha256=digest, audit_sha256=sha(bank/'AUDIT.json')))
    if (source/'RESULT.json').exists():
        original = read(source/'RESULT.json')
        assert original['completed'] and original['binding'] == binding
        assert original['results'] == summaries
    write(target/'RESULT.json', dict(completed=True, binding=binding, results=summaries,
        deployment_optimizer_updates=0, elapsed_seconds=sum(x['elapsed_seconds'] for x in summaries),
        elapsed_seconds_scope='Sum of original bank execution durations, not allocation wall time',
        group_receipt_reconstructed_from_audited_banks=True))
    write(target/'RECOVERY.json', dict(source=str(source), source_artifacts_unchanged=True,
        binding_sha256=binding_sha, original_group_receipt_present=(source/'RESULT.json').exists(),
        original_group_receipt_sha256=sha(source/'RESULT.json') if (source/'RESULT.json').exists() else None,
        banks=evidence, selection='All predeclared banks, independent of observed outcomes'))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('old','new','output'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--qualification', action='store_true')
    p.add_argument('--wait-hours', type=float, default=0)
    args = p.parse_args()
    assert sys.platform == 'linux'
    methods = ('ad',) if args.qualification else ('dpt','dit','ic_iql')
    source = args.old if args.qualification else args.new
    deadline = time.time() + args.wait_hours*3600
    while not all((b/'RESULT.json').exists() for method in methods for split in TASKS
                  for b in paths(source/'readouts/kv_bridge'/method/split, split)):
        if time.time() >= deadline:
            raise TimeoutError('Complete predeclared bank coverage is still unavailable')
        time.sleep(30)
    args.output.mkdir(parents=True, exist_ok=False)
    inputs = args.output/'inputs'
    spec = importlib.util.spec_from_file_location('original_reduce',
        HERE.parent/'robocasa-three-task-kv-velocity-meta-20261105-r1/REDUCE.py')
    reducer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reducer)
    for method in methods:
        for split in TASKS:
            original = source/'readouts/kv_bridge'/method/split
            target = inputs/'readouts/kv_bridge'/method/split
            recover_group(original, args.old/'finalized-fits-20261106-r1', target, method, split)
            summary = reducer.reduce_group(target)
            assert summary and summary['completed']
            if args.qualification:
                expected = read(original/'SUMMARY.json')
                assert summary['curves'] == expected['curves']
                assert summary['episodes'] == expected['episodes']
                assert summary['selected_candidate_sha256'] == expected['selected_candidate_sha256']
    if args.qualification:
        write(args.output/'QUALIFICATION.json', dict(passed=True, reference='Completed AD panel',
            episodes=3000, banks=70, all_curves_exact=True, source_artifacts_unchanged=True))
    else:
        subprocess.run([sys.executable,'-B',str(HERE/'COMPARE.py'),'--old',str(args.old),
            '--new',str(inputs),'--output',str(args.output/'comparison')],check=True)
        report = read(args.output/'comparison/RESULT.json')
        assert report['completed'] and report['new_episodes'] == 9000
        write(args.output/'COMPLETE.json',dict(completed=True, new_episodes=9000,
            original_bank_bytes_unchanged=True, comparison_sha256=sha(args.output/'comparison/RESULT.json')))


if __name__ == '__main__':
    main()
