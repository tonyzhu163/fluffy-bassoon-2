"""Fit a two-attempt policy on exhaustive native rollouts; freeze it on held-out scenes."""
import argparse
import itertools
import json
from pathlib import Path

ACTIONS = ('left', 'right', 'either')


def load_pairs(roots, indices):
    pairs = []
    for i in indices:
        pair = []
        for seed in (805610 + 2*i, 805611 + 2*i):
            scene = {}
            anchor = None
            for action in ACTIONS:
                filename = f'PortionFruitBowl-{seed}-{action}.json'
                paths = [root / filename for root in roots if (root / filename).is_file()]
                if len(paths) != 1:
                    raise ValueError(f'Expected one completed rollout for {filename}, got {paths}')
                row = json.loads(paths[0].read_text())
                assert (row['seed'], row['arm'], row['task']) == (seed, action, 'PortionFruitBowl')
                identity = row['physical'], row['initial_public']
                if anchor is None:
                    anchor = identity
                assert identity == anchor
                assert len(row['placed']) == 2 and all(type(x) is bool for x in row['placed'])
                assert type(row['controls']) is int and 1 <= row['controls'] <= 350
                scene[action] = row
            pair.append(scene)
        pairs.append(pair)
    return pairs


def measure(pairs, policy, reverse=False):
    first, on_failure, on_success = policy
    support_success = query_success = controls = 0
    for support, query in pairs:
        for goal in (0, 1):
            reward = support[first]['placed'][goal]
            next_action = (on_failure, on_success)[int(reward) ^ int(reverse)]
            support_success += reward
            query_success += query[next_action]['placed'][goal]
            controls += support[first]['controls'] + query[next_action]['controls']
    return dict(trials=2*len(pairs), support=support_success, query=query_success,
                total=support_success + query_success, controls=controls)


def fit(pairs):
    options = list(itertools.product(ACTIONS, repeat=3))
    def best(candidates, key):
        return max(candidates, key=lambda p: key(measure(pairs, p)))
    # Frozen lexicographic order breaks exact ties without evaluating held-out scenes.
    return {
        'active': best(options, lambda m: (m['total'], m['support'])),
        'immediate': best(options, lambda m: (m['support'], m['query'])),
        'memoryless': best([p for p in options if p[1] == p[2]], lambda m: (m['total'], m['support']))
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument('roots', nargs='+', type=Path)
    p.add_argument('--holdout', action='store_true')
    args = p.parse_args()
    dev = load_pairs(args.roots, range(4))
    policies = fit(dev)
    panel = load_pairs(args.roots, range(4, 8)) if args.holdout else dev
    counts = {a: [sum(s[a]['placed'][j] for pair in panel for s in pair) for j in (0, 1)]
              for a in ACTIONS}
    metrics = {name: measure(panel, policy) for name, policy in policies.items()}
    metrics['reverse_feedback'] = measure(panel, policies['active'], reverse=True)
    # These ceilings diagnose motor controllability. Neither is a deployable memory policy.
    informed = [max(ACTIONS, key=lambda a: sum(q[a]['placed'][g] for _, q in dev))
                for g in (0, 1)]
    known_goal = sum(q[informed[g]]['placed'][g] for _, q in panel for g in (0, 1))
    hindsight = sum(any(q[a]['placed'][g] for a in ACTIONS) for _, q in panel for g in (0, 1))
    assert all(m['query'] <= hindsight for m in metrics.values())
    print(json.dumps(dict(split='held_out' if args.holdout else 'development', policies=policies,
                          metrics=metrics, placements_by_prompt=counts,
                          goal_informed_prompts=informed, goal_informed_query=known_goal,
                          hindsight_query_ceiling=hindsight), indent=2))


if __name__ == '__main__':
    main()
