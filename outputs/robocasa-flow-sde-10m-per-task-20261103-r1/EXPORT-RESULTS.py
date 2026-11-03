"""Export completed native source and fixed-batch reporting tables; no model calls."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import time

TASKS = ['OpenDrawer', 'CloseToasterOvenDoor', 'PickPlaceDrawerToCounter',
         'CoffeeSetupMug', 'TurnOffStove']


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--wait-seconds', type=int, default=0)
    args = p.parse_args()
    assert args.output.resolve().is_relative_to(args.root.resolve())
    report_root = args.root / 'offline-initial-ratio-gpu-r3'
    deadline = time.monotonic() + args.wait_seconds
    while True:
        clock = report_root / 'clock.json'
        if clock.exists():
            clock_value = json.loads(clock.read_text())
            assert clock_value['exit_code'] == 0 and clock_value['allocated_gpus'] == 1
            break
        if time.monotonic() >= deadline:
            raise TimeoutError('Reporting GPU clock is not complete')
        time.sleep(5)
    references = {str(clock): digest(clock)}
    training, ratios, summary = [], [], []
    for task in TASKS:
        paths = {name: args.root / task / name for name in
                 ['FINAL-READOUT-R1.json', 'FINAL-ACTION-ACCOUNTING-R1.json']}
        paths['ratio'] = report_root / task / 'RESULT.json'
        values = {name: json.loads(path.read_text()) for name, path in paths.items()}
        references.update({str(path): digest(path) for path in paths.values()})
        native, ledger, report = (values[k] for k in
                                  ['FINAL-READOUT-R1.json', 'FINAL-ACTION-ACCOUNTING-R1.json', 'ratio'])
        assert native['completed'] and ledger['completed'] and report['completed']
        assert native['task'] == ledger['task'] == report['task'] == task
        assert native['training']['controls'] == ledger['valid_TRAIN_controls'] == 9984000
        assert ledger['unresolved_intents'] == 0 and not native['admitted_teacher']
        assert report['initial_parity']['passed']
        assert report['initial_parity']['max_abs_chunk_log_ratio'] == 0
        assert report['initial_parity']['max_abs_coordinate_log_ratio'] == 0
        assert report['rows'][-1]['checkpoint']['marker_sha256'] == native['checkpoint']['boundary_sha256']
        assert [r['collection'] for r in native['iteration_rates']] == list(range(1, 131))
        assert [r['checkpoint']['step'] for r in report['rows']] == [0, 1, 2] + list(range(10, 131, 10))
        for row in native['iteration_rates']:
            assert row['first_success']['episodes'] == row['success_at_end']['episodes'] == 128
            training.append(dict(task=task, iteration=row['collection'],
                cumulative_TRAIN_controls=row['collection'] * 76800,
                episodes=128, first_successes=row['first_success']['successes'],
                first_success_rate=row['first_success']['rate'],
                success_at_end_count=row['success_at_end']['successes'],
                success_at_end_rate=row['success_at_end']['rate']))
        for row in report['rows']:
            assert math.isfinite(row['mean_chunk_log_ratio_to_initial'])
            assert len(row['per_sample_chunk_log_ratio_to_initial']) == 16
            ratios.append(dict(task=task, checkpoint=row['checkpoint']['step'], batch_size=16,
                mean_chunk_log_ratio_to_initial=row['mean_chunk_log_ratio_to_initial'],
                mean_coordinate_log_ratio_to_initial=row['mean_coordinate_log_ratio_to_initial'],
                model_sha256=row['checkpoint']['sha256'],
                **{f'sample_{i:02}_chunk_log_ratio': v for i, v in enumerate(
                    row['per_sample_chunk_log_ratio_to_initial'])}))
        tr, ev = native['training'], native['paired_evaluation']
        summary.append(dict(task=task, training_controls=9984000,
            training_early_successes=tr['first10_successes'],
            training_late_successes=tr['last10_successes'], window_episodes=1280,
            training_improvement_pp=tr['improvement_percentage_points'],
            paired_before_successes=ev['before']['successes'],
            paired_after_successes=ev['after']['successes'], paired_cases=64,
            paired_wins=ev['improved_cases'], paired_losses=ev['regressed_cases'],
            exact_two_sided_mcnemar_p=ev['exact_two_sided_mcnemar_p'],
            admitted_teacher=False, corpus_role='reference_source',
            final_mean_chunk_log_ratio_to_initial=ratios[-1]['mean_chunk_log_ratio_to_initial']))
    args.output.mkdir(parents=True, exist_ok=False)
    for name, rows in [('training.csv', training), ('initial-policy-log-ratios.csv', ratios),
                       ('source-summary.csv', summary)]:
        with (args.output / name).open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    result = dict(completed=True, checked_unix=time.time(), source_sha256=digest(Path(__file__)),
        input_sha256=references, output_sha256={p.name: digest(p) for p in args.output.glob('*.csv')},
        training_rows=len(training), likelihood_rows=len(ratios), tasks=5,
        valid_TRAIN_controls=49920000, admitted_teachers=0,
        initial_parity_zero_all_tasks=True, reporting_clock=clock_value,
        metric_scope='One original fixed B16 saved-chain cohort/task; signed conditional-transition log ratios. '
                     'This is not full-action marginal KL, a performance gate or OOD adaptation evidence.',
        new_controls=0, SGD=0, model_calls=0,
        remaining='Campaign archive/cost closeout and monitor completion are separate requirements.')
    (args.output / 'RESULT.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
