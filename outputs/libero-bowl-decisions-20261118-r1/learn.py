"""Learn a one-split execution policy from measured native query rewards."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def fit(nominal, gains, pacing):
    supports = nominal + [x for x in gains if x['gain'] == 1]
    query = {(x['scene'], x['variant'], x['goal'], x['paced']): x['first'] for x in pacing}
    scenes = sorted({key[0] for key in query})
    assert not {r['scene'] for r in supports}.intersection(scenes)
    examples = []
    for r in supports:
        for assigned in range(2):
            success = r['first'][assigned] is not None
            chosen = r['goal'] if success else 1-r['goal']
            for scene in scenes:
                rewards = [query[scene,r['variant'],chosen,mode][assigned] is not None for mode in (False, True)]
                examples.append((r['public_lag'], rewards))
    def candidate(cut, low, high, gap):
        score = sum(rewards[low if cut is None or lag <= cut else high] for lag,rewards in examples)
        return dict(threshold=cut,below_paced=low,above_paced=high,training_successes=score,gap=gap)
    constants = [candidate(None,mode,mode,0) for mode in (False,True)]
    models = constants.copy()
    lags = sorted({lag for lag,_ in examples})
    for left,right in zip(lags,lags[1:]):
        for low in (False,True):
            models.append(candidate((left+right)/2,low,not low,right-left))
    def rank(m):
        return m['training_successes'],m['threshold'] is None,m['gap'],not m['below_paced']
    return dict(support_tapes=len(supports),query_scenes=scenes,training_lookups=len(examples),
                observed_lags=lags,selected=max(models,key=rank),history_blind=max(constants,key=rank))


if __name__=='__main__':
    panels=[json.loads((ROOT/f'{name}-evidence.json').read_text())['branches']
            for name in ('nominal02','paired01','pacing01')]
    print(json.dumps(fit(*panels),indent=2))
