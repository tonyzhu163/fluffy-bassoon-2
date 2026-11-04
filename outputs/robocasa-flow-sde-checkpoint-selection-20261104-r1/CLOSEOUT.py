"""Read-only Linux CPU verification and tables after all five native jobs exit."""
import argparse
import csv
import importlib.util
import json
import math
from pathlib import Path


def main(root, execution, output):
    state = json.loads(execution.read_text())
    helper = (Path(state['metadata_worktree']) / 'outputs' / root.name / 'sweep.py')
    spec = importlib.util.spec_from_file_location('qualified_sweep', helper)
    sweep = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sweep)
    matrix_path = root / state['matrix']
    matrix = sweep.read(matrix_path)
    output.mkdir(parents=True, exist_ok=False)
    summaries, curves, evidence = [], [], {}
    for task_state in state['tasks']:
        task = task_state['task']
        task_row = next(r for r in matrix['tasks'] if r['task'] == task)
        directory = Path(task_state['output'])
        assert directory == root / task / 'evaluation-r1'
        clock, result = sweep.read(directory/'clock.json'), sweep.read(directory/'RESULT.json')
        assert clock['exit_code'] == 0 and result['completed'] and result['SGD'] == 0
        parity = sweep.read(directory/'initial-parity.json')
        assert len(parity)==1 and parity[0]['passed'] and parity[0]['max_abs_chunk_log_ratio']==0
        assert sweep.read(directory/'gpu-probe.json')['status']=='passed'
        assert result['matrix_sha256'] == sweep.sha(matrix_path)
        lock_path = directory / 'SELECTION-LOCK.json'
        lock = sweep.read(lock_path)
        assert result['selection_sha256'] == sweep.sha(lock_path)
        assert lock['matrix_sha256'] == sweep.sha(matrix_path)
        panels = []
        for name, expected in result['raw_receipt_sha256'].items():
            path = directory / 'panels' / name
            assert sweep.sha(path) == expected
            panel = sweep.read(path)
            assert sweep.sha(panel['episode_path']) == panel['episode_sha256']
            raw = sweep.read(panel['episode_path'])
            assert raw['rows'] == panel['episodes']
            assert [r['seed'] for r in raw['rows']] == task_row[panel['panel']]['seeds'][panel['bank']*8:(panel['bank']+1)*8]
            assert len(raw['rows']) == 8
            initial = Path(panel['episode_path']).with_name(Path(panel['episode_path']).name.replace('-episodes.json','-initial.json'))
            assert sweep.sha(initial) == panel['initial_report_sha256']
            report = sweep.read(initial)['report']
            assert report['matched'] and report['physical_state_exact'] and report['public_state_exact']
            for file_name, digest in panel['sampler_sha256'].items():
                assert sweep.sha(directory/'sampler'/file_name) == digest
            assert len(panel['sampler_sha256']) == 12
            panels.append(panel)
        scores = {step: sum(p['first_success'] for p in panels if p['panel']=='DEV' and p['step']==step)
                  for step in sweep.STEPS}
        for step in sweep.STEPS:
            selected = [p for p in panels if p['panel']=='DEV' and p['step']==step]
            assert sorted(p['bank'] for p in selected) == list(range(8))
            successes = scores[step]
            z, n, phat = 1.959963984540054, 64, successes/64
            centre = (phat+z*z/(2*n))/(1+z*z/n)
            radius = z*math.sqrt(phat*(1-phat)/n+z*z/(4*n*n))/(1+z*z/n)
            curves.append(dict(task=task, checkpoint=step, source_TRAIN_controls=76800*step,
                DEV_first_successes=successes, DEV_cases=64, DEV_percent=100*phat,
                Wilson95_low_percent=100*(centre-radius), Wilson95_high_percent=100*(centre+radius)))
        winner = min(sweep.STEPS, key=lambda s:(-scores[s],s))
        assert lock['selected_step'] == result['selected_step'] == winner
        assert {str(k):v for k,v in scores.items()} == lock['DEV_first_successes']
        assert sorted(lock['test_steps']) == sorted(set([0,winner,130]))
        model = next(cp for cp in task_row['checkpoints'] if cp['step']==winner)
        assert model['sha256'] == lock['selected_model_sha256']
        outcomes = {}
        for step in lock['test_steps']:
            selected = [p for p in panels if p['panel']=='TEST' and p['step']==step]
            assert sorted(p['bank'] for p in selected) == list(range(16))
            outcomes[step] = {r['episode_id']:bool(r['success_once']) for p in selected for r in p['episodes']}
            assert len(outcomes[step]) == 128
        best = sweep.paired(outcomes[0],outcomes[winner],15000000+task_row['ordinal'])
        final = sweep.paired(outcomes[0],outcomes[130],16000000+task_row['ordinal'])
        assert best == result['selected_vs_initial'] and final == result['final_vs_initial']
        journal_path = directory.parent/'accounting/rank-00'/f'{directory.name}.json'
        calls = sweep.read(journal_path)['calls']
        assert all(c['resolved'] for c in calls)
        controls = sum(c['executed_or_conservative'] for c in calls)
        assert controls == result['all_controls'] == result['policy_controls'] == 4800*len(panels)
        assert all(c['started_unix'] >= lock['locked_unix'] for c in calls if c['phase'].startswith('test-'))
        assert all(c['executed_or_conservative']==0 for c in calls if c['phase'].endswith('/native_constructor_reset'))
        assert len(lock['DEV_receipt_sha256']) == 128
        for name, digest in lock['DEV_receipt_sha256'].items():
            assert name.startswith('dev-') and sweep.sha(directory/'panels'/name)==digest
        # Hash every retained raw native observation/action record after exit.
        raw_hashes = {str(p.relative_to(directory)):sweep.sha(p) for p in sorted(directory.rglob('*'))
                      if p.is_file() and p.suffix in {'.json','.pt','.gz'}}
        sweep.write(output/f'{task}-RAW-SHA256.json',raw_hashes)
        evidence[task] = dict(result_sha256=sweep.sha(directory/'RESULT.json'),
            selection_sha256=sweep.sha(lock_path), clock_sha256=sweep.sha(directory/'clock.json'),
            raw_manifest_sha256=sweep.sha(output/f'{task}-RAW-SHA256.json'),journal_sha256=sweep.sha(journal_path))
        summaries.append(dict(task=task,job=task_state['job'],selected_checkpoint=winner,
            selected_model_sha256=model['sha256'],DEV_selected_successes=scores[winner],
            TEST_cases=128,TEST_initial_successes=sum(outcomes[0].values()),
            TEST_selected_successes=sum(outcomes[winner].values()),TEST_final_successes=sum(outcomes[130].values()),
            selected_vs_initial=best,final_vs_initial=final,policy_controls=controls,
            process_GPU_seconds=clock['observed_gpu_seconds']))
    previous = 0.
    for rank, row in enumerate(sorted(summaries,key=lambda r:r['selected_vs_initial']['exact_McNemar_p'])):
        adjusted = max(previous,min(1.,(5-rank)*row['selected_vs_initial']['exact_McNemar_p']))
        row['selected_vs_initial']['Holm_p_five_tasks'] = adjusted
        previous = adjusted
    with (output/'DEV-curves.csv').open('w') as stream:
        writer = csv.DictWriter(stream,fieldnames=list(curves[0]));writer.writeheader();writer.writerows(curves)
    sweep.write(output/'RESULT.json',dict(completed=True,
        scope='Native data/statistical audit; separate scheduler-release and cumulative-cost closeout required',
        source_runtime=matrix['source_runtime'],tasks=summaries,
        evidence=evidence,original_teacher_gates_unchanged=True,SGD=0,
        qualification_charged_GPU_seconds=349,
        process_GPU_seconds_with_qualification=304+sum(r['process_GPU_seconds'] for r in summaries),
        final_scheduler_release_and_cost_audit_required=True))
    print(json.dumps(summaries,indent=2))


if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--execution',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();main(args.root,args.execution,args.output)
