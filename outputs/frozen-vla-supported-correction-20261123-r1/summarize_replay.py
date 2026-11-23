"""Aggregate the fixed six-start physical replay, preserving task/layout pair identities."""
from hashlib import sha256
import json
from pathlib import Path
import sys

import numpy as np
from scipy.stats import t

from adapter import issue


def main(replay, adapters, output):
    rows = json.loads(replay.read_bytes())['rows']
    assert len(rows)==6 and len({(r['case']['suite'],r['case']['task_id'],r['case']['init_state_id']) for r in rows})==6
    maps = json.loads(adapters.read_bytes())
    methods = ('identity','rank1','rank1_free','smooth_df')
    cells=[]
    for r in rows:
        case = r['case']
        m = r['methods']
        p = Path(r['input'])
        assert sha256(p.read_bytes()).hexdigest()==r['tape_file_sha256']
        raw = np.concatenate([c['cmd'] for c in json.loads(p.read_bytes())['chunks']]).astype(np.float32)
        ads=maps['Kinova3/'+case['suite']+'/'+str(case['task_id'])]['adapters']
        constrained = issue(raw,**ads['rank1'])
        smooth = issue(raw,**ads['smooth_df'])
        gap = (constrained-smooth)[:,:3]
        p95,maximum = np.quantile(np.linalg.norm(gap,axis=-1),[.95,1])
        ep = lambda k: np.sum(m[k]['displacements_m'],axis=0)
        cells.append(dict(case=case,controls=len(raw), replay_error_mm=r['identity_replay_max_abs_m']*1000,
            mean_chunk_mm={k:float(np.mean(m[k]['chunk_errors_mm'])) for k in methods},
            endpoint_mm={k:m[k]['endpoint_error_mm'] for k in methods},
            difference_in_endpoints_mm=float(1000*np.linalg.norm(ep('rank1')-ep('smooth_df'))),
            unrestricted_rank1_vs_smooth_endpoint_gap_mm=float(1000*np.linalg.norm(ep('rank1_free')-ep('smooth_df'))),
            p95_nominal_OSC_gap_mm=float(50*p95),max_nominal_OSC_gap_mm=float(50*maximum),
            gap_to_raw_rms=float(np.linalg.norm(gap)/np.linalg.norm(raw[:,:3]))))
    stats={}
    for k in methods:
        stat = {}
        for metric in ('mean_chunk_mm','endpoint_mm'):
            values = np.array([c[metric][k] for c in cells])
            base = np.array([c[metric]['identity'] for c in cells])
            stat[metric]=dict(mean=float(values.mean()), worse_than_identity=int(np.sum(values>base)),
                              better_than_identity=int(np.sum(values<base)))
        stat['new_boundary_controls']=sum(r['methods'][k]['command']['new_boundary_controls'] for r in rows)
        stat['reversed_controls']=sum(r['methods'][k]['command']['reversed_controls'] for r in rows)
        stat['mean_per_start_change']=float(np.mean([r['methods'][k]['command']['mean_norm'] for r in rows]))
        stat['mean_per_start_off_axis']=float(np.mean([r['methods'][k]['command']['off_axis_mean'] for r in rows]))
        stats[k]=stat
    contrasts = {}
    tasks = sorted({(c['case']['suite'],c['case']['task_id']) for c in cells})
    for metric in ('mean_chunk_mm','endpoint_mm'):
        contrasts[metric] = {}
        for reference in ('identity','rank1_free','smooth_df'):
            effects = []
            for suite, task in tasks:
                group = [c[metric]['rank1']-c[metric][reference] for c in cells
                         if (c['case']['suite'],c['case']['task_id']) == (suite, task)]
                assert len(group)==2
                effects.append(np.mean(group))
            mean = np.mean(effects)
            half = t.ppf(.975,len(tasks)-1) * np.std(effects,ddof=1)/np.sqrt(len(tasks))
            contrasts[metric]['rank1_minus_'+reference] = dict(mean=mean,approx_95=[mean-half,mean+half],
                        task_means={f'{s}/{i}':float(e) for (s,i),e in zip(tasks,effects)})
    with output.open('x') as f:
        json.dump(dict(cells=cells, stats=stats, contrasts=contrasts, tape_controls=sum(c['controls'] for c in cells)), f,
                  indent=2,allow_nan=False)
        f.write('\n')


if __name__=='__main__':
    main(*(Path(v) for v in sys.argv[1:]))
