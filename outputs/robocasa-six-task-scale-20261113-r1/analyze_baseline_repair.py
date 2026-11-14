"""Read the entire fixed64 paired baseline repair only after it completes."""
import argparse
import json
from pathlib import Path
import sys
import tarfile
import time

from audit_saved_baselines import paired, sha


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--spec', type=Path, required=True)
    p.add_argument('--spec-sha256', required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    assert sys.platform == 'linux' and sha(args.spec) == args.spec_sha256
    spec = json.loads(args.spec.read_text())
    complete_path = args.input / 'COMPLETE.json'
    complete = json.loads(complete_path.read_text())
    assert complete['completed'] and complete['cases'] == 64 and complete['physical_episodes'] == 128
    assert complete['optimizer_updates'] == 0
    assert complete['binding']['spec_sha256'] == args.spec_sha256
    assert complete['binding']['teacher'] == spec['teacher'] and spec['teacher']['step'] == 30
    assert sorted(row['first'] for row in complete['results']) == list(range(0, 64, 8))
    assert len(complete['results']) == 8
    outcomes = dict(SFT={}, PPO30={})
    records, cases = [], {}
    for first in range(0, 64, 8):
        root = args.input / f'bank-{first:04d}'
        receipt_path = root / 'ARCHIVE.json'
        receipt = json.loads(receipt_path.read_text())
        assert receipt['every_member_verified'] and receipt['cold_full_readback']
        archive = Path(receipt['archive'])
        assert archive.stat().st_size == receipt['bytes'] and sha(archive) == receipt['sha256']
        complete_row = next(row for row in complete['results'] if row['first'] == first)
        assert complete_row['archive'] == {k: v for k, v in receipt.items() if k != 'inventory'}
        data_path = root / 'PAIRED.json'
        assert sha(data_path) == receipt['inventory']['PAIRED.json']['sha256']
        with tarfile.open(archive, 'r:') as bank:
            packed = bank.extractfile('PAIRED.json').read()
        assert packed == data_path.read_bytes()
        data = json.loads(packed)
        assert data['first'] == first and len(data['cases']) == 8
        assert [row['case']['native_seed'] for row in data['cases']] == list(range(spec['seed_base'] + first, spec['seed_base'] + first + 8))
        assert all(row['case']['bank'] == spec['case_bank'] and row['case']['task'] == 'OpenDrawer'
                   and row['horizon'] == 600 for row in data['cases'])
        assert data['arms']['SFT']['endpoint'].keys() == data['arms']['PPO30']['endpoint'].keys()
        for arm, summary in data['arms'].items():
            assert arm in outcomes and summary['all_native_evidence_checked']
            assert summary['physical_episodes'] == 8 and summary['physical_controls'] == 4800
            assert len(summary['endpoint']) == 8
            for key, row in summary['endpoint'].items():
                assert key not in outcomes[arm] and row['attempt'] == 1 and row['reset_from_empty_replay_exact']
                assert isinstance(row['success_once'], bool)
                outcomes[arm][key] = row['success_once']
                if key in cases:
                    assert cases[key] == row['case']
                cases[key] = row['case']
        records.append(dict(first=first, receipt=dict(path=str(receipt_path), sha256=sha(receipt_path)),
                            archive_sha256=receipt['sha256'], summary_sha256=sha(data_path)))
    assert len(cases) == len(outcomes['SFT']) == len(outcomes['PPO30']) == 64
    assert {row['native_seed'] for row in cases.values()} == set(range(spec['seed_base'], spec['seed_base'] + 64))
    result = dict(completed=True, task='OpenDrawer', case_bank=spec['case_bank'], checked_unix=time.time(),
        paired_result=paired(outcomes['SFT'], outcomes['PPO30'], 20261120),
        cases=[dict(case=cases[key], SFT=outcomes['SFT'][key], PPO30=outcomes['PPO30'][key]) for key in sorted(cases)],
        archives=records, complete=dict(path=str(complete_path), sha256=sha(complete_path)),
        spec_sha256=args.spec_sha256, analysis_script_sha256=sha(__file__),
        limitations=['Fresh64-case panel, separate from original source SFT33/64; never pool or pair across banks.',
            'One fixed PPO30 teacher and one training seed; marginal paired bootstrap interval is descriptive.',
            'This is a source-teacher baseline, not a held-out meta-learning or history-effect estimate.'])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(result['paired_result']), flush=True)


if __name__ == '__main__':
    main()
