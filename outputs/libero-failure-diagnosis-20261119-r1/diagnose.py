"""Post hoc physical failure counts and ceilings that respect ambiguous rewards."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from panel import ROOT


def bound(rows):
    # A policy must make the same decision for two goals with the same public tape and feedback.
    groups = defaultdict(list)
    for row in rows:
        key = (row['index'], row['timing'], row['target'], row['reward'])
        groups[key].append(row['native'])
    return sum(max(map(sum, zip(*hits))) for hits in groups.values())


def illustrate(contacts, output):
    pairs = [(a,b) for a in contacts for b in contacts
             if a['reached'] and not b['reached'] and a['contacts'] and b['contacts']
             and (a['target'],a['timing']) == (b['target'],b['timing'])]
    if not pairs:
        return []
    good, missed = pairs[0]
    destinations = ['cabinet','rack']
    target = destinations[good['target']]
    hidden = destinations[1-good['target']]
    fig, axes = plt.subplots(2,3,figsize=(8.9,5.9),layout='constrained')
    fig.suptitle(f'Bottle placements under {good["timing"]} timing\nAttempted {target} · hidden goal {hidden} · reward zero in both trials', fontsize=14)
    for row,(record,label) in enumerate([(good,'Reached'),(missed,'Missed')]):
        with np.load(ROOT/record['path']/'public.npz') as tape:
            frames, steps = tape['frames'], tape['decision_steps']
        for col,control in enumerate((0,144,288)):
            ax = axes[row,col]
            index = np.flatnonzero(steps==control).item()
            ax.imshow(frames[index,0])
            ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if not row:
                ax.set_title(f'After {control} controls',fontsize=11)
            if not col:
                ax.set_ylabel(f'{label} {target}\nPose {record["pose"]}',rotation=0,ha='right',va='center',labelpad=8)
    fig.savefig(output,dpi=165)
    plt.close(fig)
    return [v['path'] for v in (good,missed)]


def main(directory):
    replay = json.loads((directory/'replay.json').read_text())['bottle_transfer']
    contacts = [json.loads(line) for line in (directory/'contacts.jsonl').read_text().splitlines()]
    assert len(contacts)==24 and {r['path'] for r in contacts} == {t['path'] for t in replay['tapes']}
    for r in contacts:
        t = next(t for t in replay['tapes'] if t['path']==r['path'])
        assert r['reached'] == t['hit'][t['target']]
    misses = [r for r in contacts if not r['reached']]
    rows = replay['rows']
    sets = dict(all=rows, zero=[r for r in rows if not r['reward']],
                completed_wrong_target=[r for r in rows if r['completed'] and not r['reward']],
                missed_both=[r for r in rows if not r['completed']])
    evidence = dict(tape_and_reward_ceiling={k:bound(v) for k,v in sets.items()},
                    missed_tapes=len(misses), missed_with_bilateral_contact=sum(bool(r['contacts']) for r in misses),
                    verified_tapes=len(contacts), misses=misses,
                    illustration=illustrate(contacts,directory/'same-reward.png'))
    (directory/'diagnosis.json').write_text(json.dumps(evidence,indent=2)+'\n')
    print(json.dumps({k:v for k,v in evidence.items() if k!='misses'}))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('directory',type=Path)
    main(p.parse_args().directory)
