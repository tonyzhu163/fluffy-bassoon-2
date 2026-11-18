"""Collect real chronological SFT histories on the frozen eight-task fitting subset."""
import argparse
import json
from pathlib import Path
import time

from native import Motor, write
from qualify import rollout


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--deadline', required=True, type=float)
    parser.add_argument('--checkpoint', required=True, choices=('openpi', 'rlinf'))
    parser.add_argument('--reference-cadence', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    splits = json.loads(Path(__file__).with_name('SPLITS.json').read_text())
    subset = splits['development_scaling']['task_subsets']['8']
    trials = []
    for repeat in range(2):
        for suite_index, (suite, tasks) in enumerate(sorted(subset.items())):
            for task_id in tasks:
                # Match resets and exogenous draws across BASE and S20. Each
                # repeat uses three fresh reset IDs and independent noise.
                seed_base = 18110000 + suite_index * 10000 + task_id * 100 + repeat * 3
                for variant in ('BASE', 'S20'):
                    trials.append([dict(suite=suite, task_id=task_id, variant=variant,
                                        init_state_id=3 * repeat + attempt,
                                        seed=seed_base + attempt) for attempt in range(3)])
    write(args.output/'design.json', dict(trials=trials, split='fitting',
          source='frozen SFT', attempts_per_trial=3, paired_conditions=['BASE', 'S20']))
    motor = Motor(args.checkpoint, reference_cadence=args.reference_cadence)
    write(args.output/'motor-config.json', dict(motor.config, executed_before_replan=motor.replan_steps))
    summary = []
    for trial_id, cases in enumerate(trials):
        outcomes = []
        for attempt, case in enumerate(cases):
            if time.time() >= args.deadline:
                raise TimeoutError('Pilot reached its allocation limit')
            episode = args.output/f'trial-{trial_id:03d}'/f'attempt-{attempt+1}'
            result = rollout(motor, case, episode, role='fitting corpus pilot, frozen SFT behavior')
            outcomes.append({k:result[k] for k in ('case','success','controls','seconds','npz_bytes')})
        summary.append(dict(trial_id=trial_id, episodes=outcomes))
        write(args.output/'completed-trials.json', summary)


if __name__ == '__main__':
    main()
