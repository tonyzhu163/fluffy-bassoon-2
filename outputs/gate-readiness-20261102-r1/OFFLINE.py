"""Existing-data reward thresholds and gate0 noise decomposition; Linux CPU only."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import platform
import statistics
import subprocess


def ref(path):
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def wilson(k, n):
    if not n:
        return None
    z = 1.959963984540054
    den = 1 + z*z/n
    center = (k/n + z*z/(2*n))/den
    half = z*math.sqrt(k/n*(1-k/n)/n + z*z/(4*n*n))/den
    return [max(0., center-half), min(1., center+half)]


def metrics(rows, predict):
    tp = fp = tn = fn = 0
    for row in rows:
        yes, truth = predict(row), row['truth']['success']
        tp += yes and truth; fp += yes and not truth
        tn += not yes and not truth; fn += not yes and truth
    return dict(n=len(rows), tp=tp, fp=fp, tn=tn, fn=fn,
        precision=tp/(tp+fp) if tp+fp else None, precision_ci95=wilson(tp,tp+fp),
        recall=tp/(tp+fn) if tp+fn else None, false_positive_rate=fp/(fp+tn) if fp+tn else None,
        false_positive_rate_ci95=wilson(fp,fp+tn))


def threshold_analysis(pairs):
    train = [r for r in pairs if r['dataset'] == 'TRAIN']
    assert len(pairs) == 634 and len(train) == 384
    group = lambda r: hashlib.sha256(str(r['truth']['case']['native_seed']).encode()).digest()[0] % 3
    fit = [r for r in train if group(r) != 0]
    validation = [r for r in train if group(r) == 0]
    assert {r['truth']['case']['native_seed'] for r in fit}.isdisjoint(
        r['truth']['case']['native_seed'] for r in validation)
    thresholds = (.5, .6, .7, .75, .8, .85, .9, .925, .95, .975, .99)
    result = {}
    for prompt in ('saved_original', 'rerun_original', 'candidate'):
        rows = []
        for field in ('reward', 'p5'):
            for threshold in thresholds:
                def predict(r):
                    score = r[prompt]
                    value = (score['expected_score'] - 1)/4 if field == 'reward' else score['probabilities']['5']
                    return value >= threshold
                rows.append(dict(field=field, threshold=threshold, fit=metrics(fit,predict),
                    validation=metrics(validation,predict), all_train=metrics(train,predict),
                    calibration_by_task={task:metrics([r for r in pairs if r['dataset'] != 'TRAIN'
                        and r['task_key'] == task],predict)
                        for task in sorted({r['task_key'] for r in pairs if r['dataset'] != 'TRAIN'})}))
        # Pick using fit data only. This is internal development, not an unseen qualification bank.
        eligible = [r for r in rows if r['fit']['precision_ci95'] is not None
            and r['fit']['precision_ci95'][0] >= .95 and r['fit']['recall'] >= .5
            and r['fit']['false_positive_rate_ci95'][1] <= .05]
        chosen = max(eligible, key=lambda r:r['fit']['recall']) if eligible else None
        result[prompt] = dict(hard_grade5=metrics(train,lambda r:r[prompt]['score']==5),
                              sweeps=rows, fit_selected_candidate=chosen)
    return dict(input_episodes=len(pairs), train_episodes=len(train), fit=len(fit), validation=len(validation),
        candidate_selection_rule=dict(precision_lower95=.95, recall_min=.5, false_positive_upper95=.05),
        prompts=result, production_admitted=False,
        first_success_timing='NOT_MEASURABLE: saved scores are terminal-only; causal prefix/chunk scores absent',
        limitations=['All videos already used for development; internal held-out split is not independent qualification.',
                     'Terminal native success differs from success_once; H750 calibration does not qualify H400/H600.',
                     'Thresholding changes the reward contract and requires separate calibration and cache identity.'])


def noise_analysis(directory):
    tasks = {}
    for path in sorted(directory.glob('*-native8.json')):
        audit = json.loads(path.read_text())
        assert audit['completed'] and audit['physical_and_public_pairing_exact']
        cases = defaultdict(dict)
        for identity, outcome in audit['combined_outcomes'].items():
            seed, draw = map(int,identity.split(':'))
            assert type(outcome) is bool and draw not in cases[seed]
            cases[seed][draw] = outcome
        assert len(cases) == 20 and all(set(x)==set(range(8)) for x in cases.values())
        counts = [sum(x.values()) for x in cases.values()]
        histogram = Counter(counts)
        probabilities = [k/8 for k in counts]
        within = statistics.mean(p*(1-p)*8/7 for p in probabilities)
        between = statistics.variance(probabilities) - within/8
        tasks[audit['task']] = dict(cases=20, draws_per_case=8,
            histogram={str(k):histogram[k] for k in range(9)},
            observed_all_fail=histogram[0], observed_all_success=histogram[8],
            observed_mixed=20-histogram[0]-histogram[8],
            coin_flip_3_to_5_of8=sum(histogram[k] for k in (3,4,5)),
            successes=sum(counts), single_draw_rate=statistics.mean(probabilities),
            oracle_best_of8_cases=20-histogram[0],
            within_case_bernoulli_variance_unbiased=within,
            between_case_probability_variance_unbiased=between,
            mean_pair_disagreement=2*within,
            per_case=[dict(native_seed=seed, successes=sum(outcomes.values()),
                           probability=sum(outcomes.values())/8, wilson95=wilson(sum(outcomes.values()),8),
                           outcomes=[outcomes[i] for i in range(8)])
                      for seed,outcomes in sorted(cases.items())], source=ref(path))
    assert len(tasks) == 4
    return dict(tasks=tasks, episodes=640, cases=80, new_episodes=0,
        selection='20 selected anchor failures/task, not a random task population; H600 final-state outcomes',
        interpretation='Mixed outcomes show motor-noise headroom. Best-of8 is an oracle ceiling, not a realizable selector or evidence of adaptation. All-fail/all-success are observed 8-draw classes, not known probabilities 0/1. A negative between-case variance estimate is retained, not silently clipped.')


def main():
    assert platform.system() == 'Linux', 'Run offline analysis on allocated Linux'
    p=argparse.ArgumentParser();p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    pairs_path=args.data_root/'robocasa-flora-rubric-replay-20261102-r1/PAIRED-EPISODES.json'
    result=dict(complete=True, platform=platform.platform(), new_GPU_seconds=0,
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        input=ref(pairs_path), reward=threshold_analysis(json.loads(pairs_path.read_text())),
        noise=noise_analysis(args.data_root/'robocasa-interface-screen-20261102-r3/audits'))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(result,stream,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps(dict(complete=True,output=str(args.output),noise=result['noise']['tasks'],
        thresholds={p:r['fit_selected_candidate'] for p,r in result['reward']['prompts'].items()}),indent=2))


if __name__=='__main__':
    main()
