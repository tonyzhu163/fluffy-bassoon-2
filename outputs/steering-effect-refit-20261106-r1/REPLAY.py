"""Reconstruct the corrected stop from original scheduled validation checks."""
import argparse
import json
from pathlib import Path


def replay(rows, final_update, schedule):
    from fluffy_bassoon.meta_rl.vla_history_train import advance_plateau
    scheduled = {}
    for row in rows:
        update = row['update']
        if update > final_update or (update != 1 and update % 100):
            continue
        loss = row['validation_flow_loss']
        if update in scheduled and scheduled[update] != loss:
            raise ValueError('Contradictory original scheduled validation records')
        scheduled[update] = loss
    if set(scheduled) != {1, *range(100, final_update + 1, 100)}:
        raise ValueError('Original scheduled validation history is incomplete')
    state, first_convergence = {}, None
    for update, loss in sorted(scheduled.items()):
        state = advance_plateau(state, loss, update, minimum_updates=schedule['minimum_updates'],
            patience=schedule['validation_patience'], relative_delta=schedule['validation_relative_delta'])
        if state['converged'] and first_convergence is None:
            first_convergence = update
    return dict(plateau=state, scheduled_checks=len(scheduled), first_convergence_update=first_convergence,
                last_update=final_update, stopping_rule='relative_abs_reference_v2')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parent', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    contract = json.loads(Path(__file__).with_name('CONTRACT.json').read_text())
    result = {}
    for interface in contract['diagnostic']['interfaces']:
        for objective in contract['diagnostic']['objectives']:
            fit = args.parent / 'fits' / interface / objective
            status = json.loads((fit / 'training-status.json').read_text())
            rows = [json.loads(line) for line in (fit / 'metrics.jsonl').read_text().splitlines()]
            result[interface + '/' + objective] = replay(rows, status['update'], contract['refit'])
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
