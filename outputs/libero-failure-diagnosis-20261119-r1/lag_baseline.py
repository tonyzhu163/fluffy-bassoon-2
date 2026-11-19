"""Apply the prior bowl-fitted cadence threshold to the current query menu."""
import argparse
import json
from pathlib import Path
from panel import ACTIONS, TRANSFER
from design import use_pacing
from diagnose import bound


def score(panel, model):
    tapes = {(t['index'],t['timing'],t['target']):t for t in panel['tapes']}
    rows = []
    for r in panel['rows']:
        tape = tapes[r['index'], r['timing'], r['target']]
        target = r['target'] if r['reward'] else 1-r['target']
        pace = use_pacing(model,tape['lag'])
        success = r['native'][ACTIONS.index((target,pace))]
        rows.append(dict(r, success=success, retry_target=target, retry_paced=pace))
    groups = dict(all=rows, zero=[r for r in rows if not r['reward']],
                  wrong_target=[r for r in rows if not r['reward'] and r['completed']],
                  missed_both=[r for r in rows if not r['completed']])
    scores = {k:dict(cases=len(v), native=sum(r['success'] for r in v), tape_reward_bound=bound(v))
              for k,v in groups.items()}
    keys = ('index','timing','target','hidden','reward','retry_target','retry_paced','success')
    return dict(scores=scores, decisions=[{k:r[k] for k in keys} for r in rows])


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('replay',type=Path)
    p.add_argument('output',type=Path)
    arg = p.parse_args()
    policy_file = TRANSFER/'frozen-policy.json'
    model = json.loads(policy_file.read_text())['learned']['model']
    # Only the public-lag cadence component is reused. First targets stay balanced.
    data = json.loads(arg.replay.read_text())
    out = dict(previous_fit=str(policy_file), model=model,
               results={name:score(panel,model) for name,panel in data.items()})
    with arg.output.open('x') as f:
        print(json.dumps(out,indent=2),file=f)
    print(out['results']['bottle_transfer']['scores'])
