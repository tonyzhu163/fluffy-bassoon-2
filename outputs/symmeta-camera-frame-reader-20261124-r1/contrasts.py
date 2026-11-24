"""Regime/control and pooled physical contrasts from the released A/B prediction rows."""
import gzip
import json
from collections import defaultdict
from pathlib import Path
import sys
from summarize import estimate,error


def collect(path):
    with gzip.open(path,'rt') as f:
        rows=json.load(f)
    lookup={(r['index'],r['seed'],r['view'],r['model'],r['calibration']):r for r in rows}
    groups=defaultdict(lambda:defaultdict(list))
    for a in rows:
        if (a['model'],a['calibration'])!=('A','true') or a['regime']=='familiar':
            continue
        idx,s,view=a['index'],a['seed'],a['view']
        b=lookup[idx,s,view,'B','true']
        assert all(a[k]==b[k] for k in ('observed_sha256','calibration_sha256','native_view_sha256','source_sha256','true_yaw_deg','true_gain','true_sign'))
        n={m:lookup[idx,s,'nominal',m,'true'] for m in ('A','B','shipped')}
        key=(a['task'],s)
        if view=='nominal':
            condition='nominal'
        elif view in ('roll-20','roll+20','roll-45','roll+45'):
            condition='primary_digital'
        elif view in ('orbit-','orbit+'):
            condition='orbit'
        elif view in ('roll-','roll+'):
            condition='optical_roll'
        else:
            continue
        metrics={'A':error(a),'B':error(b),'A_minus_B':error(a)-error(b),
                 'A_cost_vs_upright':error(a)-error(n['A']), 'B_cost_vs_upright':error(b)-error(n['B']),
                 'A_cost_vs_shipped_upright':error(a)-error(n['shipped']),
                 'B_cost_vs_shipped_upright':error(b)-error(n['shipped'])}
        if view!='nominal':
            for m in ('A','B'):
                wrong=lookup[idx,s,view,m,'opposite']
                assert wrong['observed_sha256']==a['observed_sha256']
                metrics[m+'_wrong_cost']=error(wrong)-metrics[m]
        strata=['all_ood',a['regime'],a['embodiment']]
        if a['panel']=='physical':
            strata.append('original8' if a['original_orbit'] else 'additional56')
        for split in strata:
            for k,value in metrics.items():
                groups[condition,split,k][key].append(value)
    out={}
    for (c,split,k),vals in groups.items():
        out.setdefault(c,{}).setdefault(split,{})[k]=estimate(vals)
    return out


if __name__=='__main__':
    a,b,output=map(Path,sys.argv[1:])
    with output.open('x') as f:
        json.dump(dict(digital=collect(a),physical=collect(b)),f,separators=(',',':'),allow_nan=False)
