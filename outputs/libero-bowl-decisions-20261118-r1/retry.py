"""Evaluate single-support decisions by looking up paired native query outcomes."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from statistics import median


def evaluate(supports, queries):
    def strategy(b):
        if b.get('paced',False):
            return 'paced'
        return {1.: 'full', .5: 'half'}[b['gain']]

    lookup = {(b['scene'], b['variant'], b['goal'], strategy(b)): b['first'] for b in queries}
    query_scenes = sorted({b['scene'] for b in queries})
    support_scenes = {b['scene'] for b in supports}
    assert support_scenes.isdisjoint(query_scenes)
    gains = ('full', 'paced' if any(b.get('paced') for b in queries) else 'half')
    variants = ('BASE', 'S20')
    assert len(lookup) == len(queries) == len(query_scenes)*len(variants)*2*len(gains)
    lags = {(b['scene'], b['variant'], b['goal']): b['public_lag'] for b in supports}
    by_condition = defaultdict(Counter)
    timing_correct = 0
    for tape in supports:
        scene, variant, option = tape['scene'], tape['variant'], tape['goal']
        anchors = [median(b['public_lag'] for b in supports if b['variant']==v and b['scene']!=scene)
                   for v in variants]

        def select_gain(lag):
            return min(zip(anchors,gains), key=lambda pair: abs(lag-pair[0]))[1]

        gain = select_gain(tape['public_lag'])
        timing_correct += variants[gains.index(gain)] == variant
        other = next(v for v in variants if v != variant)
        donor_gain = select_gain(lags[scene, other, option])
        counts = by_condition[variant]
        for assigned in (0,1):
            support_reward = tape['first'][assigned] is not None
            chosen = option if support_reward else 1-option
            for q in query_scenes:
                def success(dest, g):
                    return lookup[q, variant, dest, g][assigned] is not None
                counts['trials'] += 1
                counts['selected_assigned_destination'] += (chosen == assigned)
                counts['matched'] += success(chosen,gain)
                counts['other_timing_support'] += success(chosen, donor_gain)
                for g in gains:
                    counts['retry_'+g] += success(chosen, g)
                    counts['uniform_'+g] += sum(success(dest,g) for dest in (0,1)) / 2
                counts['uniform_matched_execution'] += sum(success(dest,gain) for dest in (0,1)) / 2
                counts['privileged_best_branch'] += any(success(dest,g) for dest in (0,1) for g in gains)
    totals = {name: sum(row[name] for row in by_condition.values())
              for name in by_condition[variants[0]]}
    return dict(supports=len(supports),query_scenes=query_scenes,timing_correct=timing_correct,
                by_condition=dict(by_condition),total=totals)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('supports',type=Path)
    p.add_argument('queries',type=Path)
    args=p.parse_args()
    result=evaluate(*(json.loads(x.read_text())['branches'] for x in (args.supports,args.queries)))
    print(json.dumps(result,indent=2))
