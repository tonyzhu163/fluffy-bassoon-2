"""Pair the frozen replication with the follow-up on its preassigned observations."""
from collections import defaultdict
from pathlib import Path
from readback import clustered, dump, error, read

HERE=Path(__file__).resolve().parent


def main():
    pred=HERE/'evidence/fit01/Kinova3-predictions.json.gz'
    prior=HERE.parent/'symmeta-camera-independent-body-20261124-r1/evidence/score01/predictions.json.gz'
    source=read(prior)
    old={(r['index'],r['seed'],r['model']):r for r in source if r['model'] in ('A','B') and r['calibration']=='true'}
    current=[r for r in read(pred) if r['selected'] and r['assigned'] and r['calibration']=='true' and r['model'] in ('A','B')]
    assert len(old)==len(current)==384
    changes=defaultdict(dict)
    groups=defaultdict(lambda:defaultdict(list))
    for r in current:
        idx,s,m=r['index'],r['seed'],r['model']
        p=old[idx,s,m]
        for k in ('index','task','seed','model','view','observed_sha256','calibration_sha256',
                  'sensor_sha256','native_view_sha256','source_sha256'):
            assert p[k]==r[k],k
        key=r['task'],s
        e=error(dict(p,true_yaw_deg=r['true_yaw_deg']))
        groups[m+'_original'][key].append(e)
        groups[m+'_new'][key].append(error(r))
        delta=error(r)-e
        groups[m+'_new_minus_original'][key].append(delta)
        changes[(idx,key)][m]=delta
    for (_,key),d in changes.items(): groups['gap_new_minus_original'][key].append(d['A']-d['B'])
    dump(HERE/'evidence/readback01/Kinova3-original-change.json',{k:clustered(v) for k,v in groups.items()})


if __name__=='__main__': main()
