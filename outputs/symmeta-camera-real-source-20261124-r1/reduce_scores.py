"""Paired task-cluster inference with all signs, cases and seeds attached."""
from collections import defaultdict
import numpy as np
from summarize import estimate,error


def reduce_panel(rows,old):
    idx={(r['budget'],r['index'],r['seed'],r['view'],r['model'],r['calibration']):r for r in rows}
    assert len(idx)==len(rows)
    prior={} if old is None else {(r['index'],r['seed'],r['view'],r['model']):r for r in old if r['calibration']=='true'}
    table=defaultdict(lambda:defaultdict(list))
    for r in rows:
        if (r['model'],r['calibration'])!=('B','true'): continue
        n,i,s,v=(r[k] for k in ('budget','index','seed','view'))
        def get(m,cal='true',view=v): return idx[n,i,s,view,m,cal]
        assert all(get(m)['observed_sha256']==r['observed_sha256'] for m in ('A','B','digital_A','digital_B'))
        groups=[v]+(['both_orbits'] if v!='nominal' else [])+(['native_assigned'] if r['assigned'] else [])
        for g in groups:
            def put(k,val):
                table[n,g,k][r['task'],s].append(val)
            errs={m:error(get(m)) for m in ('A','B','digital_A','digital_B')}
            for m,e in errs.items():
                a=get(m)
                put(m,e)
                put(m+'_gain_abs_log',abs(np.log(a['xy_scale']/a['true_gain'])))
                put(m+'_sign_accuracy',float(a['gripper_sign']==a['true_sign']))
                if v!='nominal':
                    put(m+'_minus_nominal',e-error(get(m,view='nominal')))
                    for cal in ('identity','opposite'):
                        z=get(m,cal)
                        assert z['observed_sha256']==a['observed_sha256'] and z['checkpoint_sha256']==a['checkpoint_sha256']
                        put(m+'_'+cal+'_minus_true',error(z)-e)
            real=errs['A']-errs['B']; digital=errs['digital_A']-errs['digital_B']
            put('real_gap',real); put('digital_gap',digital); put('gap_interaction',real-digital)
            for m in 'AB': put(m+'_real_minus_digital',errs[m]-errs['digital_'+m])
            if prior:
                for m in 'AB':
                    p=prior[i,s,v,m]
                    for key in ('observed_sha256','native_view_sha256','source_sha256','true_yaw_deg','calibration_sha256'):
                        assert p[key]==get(m)[key],key
                    put(m+'_vs_original',errs[m]-error(p))
                put('gap_vs_original',real-(error(prior[i,s,v,'A'])-error(prior[i,s,v,'B'])))
    results={}
    for (n,g,k),d in table.items(): results.setdefault(str(n),{}).setdefault(g,{})[k]=estimate(d)
    return results
