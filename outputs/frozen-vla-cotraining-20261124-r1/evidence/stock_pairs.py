import collections
import itertools
import json
import math
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
panel = json.loads((root / 'evaluation/gpu02/verified-native.json').read_text())
by_robot = {}
for robot in ('Kinova3', 'Panda'):
    cases = [row for row in panel['rows'] if row['case']['arm'] == robot]
    groups = collections.defaultdict(list)
    for row in cases:
        c = row['case']
        groups[(c['suite'], c['task_id'])].append(row)
    contrasts = {}
    for comp in ('joint', 'adapter_only', 'policy_only'):
        effects, changes = [], []
        law = collections.Counter({0: 1})
        for rows in groups.values():
            ds = []
            for row in rows:
                modes = row['outcomes']
                v = modes[comp][0] - modes['original'][0]
                ds.append(v)
                if v:
                    c = row['case']
                    changes.append([c['suite'], c['task_id'], c['init_state_id'], v])
            effects.extend(ds)
            per = collections.Counter(sum(draw) for draw in itertools.product(ds, repeat=len(ds)))
            joined = collections.Counter()
            for a,na in law.items():
                for b,nb in per.items(): joined[a+b] += na*nb
            law = joined
        n = len(effects)
        def q(p):
            count, rank = 0, math.ceil(p*sum(law.values()))
            for x, freq in sorted(law.items()):
                count += freq
                if count >= rank: return x/n
        contrasts[comp] = dict(pairs=n, rescued=effects.count(1), lost=effects.count(-1),
                               difference=sum(effects)/n, interval95=[q(.025),q(.975)], discordant=changes)
    by_robot[robot] = contrasts
print(json.dumps(dict(reference='original', method='Exact paired layout bootstrap within the three fixed tasks and robot. One fixed noise bank per case. Adapter-only and policy-only vs stock are exploratory.', results=by_robot), indent=2))
