"""Run locally. SSH invokes only the installed SQLite shell and read-only utilities."""
import ast
import csv
import json
import shlex
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE.parent
REMOTE = '/mowenjie/fluffy-bassoon-2/outputs'
SCALE = 'libero-xemb-ur5e-scale-20261120-r1'
CAMP = 'xemb-icml-campaign-20261121-r1'
V2 = 'ours_equiv_v2_passive'
HS = 'analytic_visual_hs_passive'
PANELS = {
    'ur5e-pi05': ('gpu03s2a', 'gpu30msur5e'),
    'iiwa-pi05': ('gpu09iiwa', 'gpu31msiiwa'),
    'kinova3-pi05': ('gpu06kinova3', 'gpu32mskinova'),
    'ur5e-pi0': ('gpu08pi0fix', 'gpu33msur5epi0'),
}
SEEDS = ['s41', 's42', 's43']
SUITES = ['libero_10', 'libero_goal', 'libero_object', 'libero_spatial']
TASKS = list(zip([s for s in SUITES for _ in range(2)], [5,8,3,6,1,8,0,8]))
Q = shlex.quote

def ssh(command):
    return subprocess.check_output(['ssh', 'baige-dev', command], text=True).strip()

def sql(query):
    result = ssh('python3 -m sqlite3 :memory: ' + Q(query))
    return [json.loads(ast.literal_eval(line)[0]) for line in result.splitlines()]

def save(name, rows):
    with (HERE/name).open('w') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

def projection(campaign, run, shard, names):
    path = f'{REMOTE}/{campaign}/{run}/live/shard-{shard}/live-shard.json'
    cols = ['index','suite','task_id','embodiment','regime','true_yaw','true_scale','true_grip']
    paths = cols + ['arms.'+name+'.success' for name in names]
    fields = ','.join("json_extract(value,'$."+p+"')" for p in paths)
    types = ','.join("json_type(value,'$.arms."+n+".success')" for n in names)
    # Encode file bytes as hex so JSON text cannot become SQL syntax.
    cmd = ('python3 -m sqlite3 :memory: "WITH doc(j) AS (SELECT CAST(X\'' +
           '$(xxd -p '+Q(path)+' | tr -d \'\\n\')' +
           '\' AS TEXT)) SELECT json_array(json_extract(j,\'$.shard_idx\'),'+fields+','+types+
           ') FROM doc,json_each(j,\'$.rows\')"')
    rows = []
    for line in ssh(cmd).splitlines():
        idx,*rest = json.loads(ast.literal_eval(line)[0])
        assert idx == shard and all(t in ('true','false') for t in rest[-len(names):])
        rows.append((tuple(rest[:len(cols)]),rest[len(cols):-len(names)]))
    assert len(rows) == len(dict(rows)) == 12
    return dict(rows), path

def analyze(rows, panel, split, reader):
    # d is integer-valued. For the mean, three times the seed-average contrast
    # preserves exact tail comparisons and gives one sign per query.
    scale = 3 if reader == 'mean' else 1
    d = ('s41+s42+s43-3*hs' if scale == 3 else reader+'-hs')
    vals = ','.join('('+','.join(str(r[k]) for k in ('task_order','hs',*SEEDS))+')' for r in rows)
    ctes = [f'obs(t,hs,s41,s42,s43) AS (VALUES {vals})',
            f'q AS (SELECT t,{d} AS d FROM obs)',
            'tasks AS (SELECT t,sum(d) AS v,count(*) AS n FROM q GROUP BY t)',
            'masks(m) AS (VALUES(0) UNION ALL SELECT m+1 FROM masks WHERE m<255)',
            'flips AS (SELECT sum(iif((m >> t)&1,v,-v)) AS v FROM masks,tasks GROUP BY m)',
            'weights(w) AS (VALUES(1),(2),(3))',
            'counts AS (SELECT w,count(d) AS n FROM weights LEFT JOIN q ON abs(d)=w GROUP BY w)',
            '''bin(w,n,k,v,prob) AS (SELECT w,n,0,-w*n,pow(0.5,n) FROM counts
                UNION ALL SELECT w,n,k+1,v+2*w,prob*(n-k)/(k+1) FROM bin WHERE k<n)''',
            '''tail AS (SELECT sum(a.prob*b.prob*c.prob) AS p FROM bin a,bin b,bin c
                WHERE a.w=1 AND b.w=2 AND c.w=3 AND abs(a.v+b.v+c.v)>=abs((SELECT sum(d) FROM q)))''',
            'boot0(v,num) AS (VALUES(0,1))']
    # Exact empirical percentile task bootstrap. Convolve the 8 task totals
    # eight times (8^8 equiprobable ordered resamples), preserving whole tasks.
    for i in range(1,9):
        ctes.append(f'boot{i}(v,num) AS (SELECT b.v+t.v,sum(b.num) FROM boot{i-1} b,tasks t GROUP BY b.v+t.v)')
    ctes.append('cdf AS (SELECT v,sum(num) OVER (ORDER BY v) AS cum FROM boot8)')
    base = 'WITH RECURSIVE '+',\n'.join(ctes)
    expr = f"""SELECT json_array(count(*),sum(d>0),sum(d<0),sum(d=0),100.0*avg(d)/{scale},
        (SELECT p FROM tail),(SELECT count(*) FROM flips WHERE abs(v)>=abs((SELECT sum(d) FROM q))),
        (SELECT min(v) FROM cdf WHERE cum*40 >= 16777216)*100.0/(count(*)*{scale}),
        (SELECT min(v) FROM cdf WHERE cum*40 >= 39*16777216)*100.0/(count(*)*{scale})) FROM q"""
    n,win,loss,tie,delta,p,k,lo,hi = sql(base+' '+expr)[0]
    prefix = dict(panel=panel,split=split,reader=reader)
    result = dict(**prefix,n=n,wins=win,losses=loss,ties=tie,delta_pp=delta,p_query=p,
                  task_tail=k,p_task=k/256,task_boot_lo_pp=lo,task_boot_hi_pp=hi)
    task_rows = []
    for t,v,n in sql(base+' SELECT json_array(t,v,n) FROM tasks ORDER BY t'):
        task_rows.append(dict(**prefix,task_order=t,suite=TASKS[t][0],task_id=TASKS[t][1],
                              n=n,net_successes=v/scale,delta_pp=100*v/(scale*n)))
    return result,task_rows

def main():
    queries,checks,stats,tasks,sources = [],[],[],[],[]
    for panel,(old,new) in PANELS.items():
        panel_rows = []
        for task in range(8):
            source41 = 'gpu04s2agoal6' if panel=='ur5e-pi05' and task==3 else old
            a,pa = projection(CAMP,source41,task,[HS,V2])
            b,pb = projection(SCALE,new,task,[V2+'_s42',V2+'_s43'])
            hashes=ssh('sha256sum '+Q(pa)+' '+Q(pb)).splitlines()
            ha,hb=[line.split()[0] for line in hashes]
            if panel=='ur5e-pi05':
                manifest=OUT/'xemb-live-wave-20261121-r1/evidence/pi05-provenance.json'
                reference=json.loads(manifest.read_text())['shards'][str(task)]
                assert (pa,ha)==(reference['path'],reference['sha256'])
            assert a.keys()==b.keys()
            sources.append(dict(panel=panel,task_order=task,seed41_and_hs=pa,seeds42_43=pb,sha256_41_hs=ha,sha256_42_43=hb))
            for key in sorted(a):
                idx,suite,tid,shift,reg,*truth = key
                assert TASKS[task]==(suite,tid)
                hs,s41 = a[key]
                s42,s43 = b[key]
                panel_rows.append(dict(panel=panel,index=idx,task_order=task,suite=suite,task_id=tid,
                    embodiment=shift,regime=reg,variant='BASE',query_init_state_id=13,
                    query_reset_seed=41800013+10000*SUITES.index(suite)+100*tid,
                    hs=hs,s41=s41,s42=s42,s43=s43))
        assert [r['index'] for r in panel_rows]==list(range(96))
        assert len({(r['suite'],r['task_id'],r['embodiment']) for r in panel_rows})==96
        pubs = [OUT/CAMP/'evidence'/f'{panel}-summary.json',
                OUT/SCALE/'evidence'/f'multiseed-zeroshot-{panel}.json']
        # Inferential quantities and validation counts both come from Linux SQLite.
        for path in pubs:
            published=json.loads(path.read_text())['by_regime']
            for reg,data in published.items():
                subset=[r for r in panel_rows if reg=='overall' or reg==r['regime'] or
                        (reg=='shifted' and r['embodiment']!='fam_nom') or
                        (reg=='ood_combined' and r['regime'] in ('compositional_ood','continuous_angle_ood'))]
                mapping={HS:'hs',V2:'s41',**{V2+'_'+s:s for s in SEEDS}}
                for method,v in data['successes'].items():
                    if method not in mapping: continue
                    vals=','.join('('+str(r[mapping[method]])+')' for r in subset)
                    n,count=sql(f'WITH x(y) AS (VALUES {vals}) SELECT json_array(count(*),sum(y)) FROM x')[0]
                    assert (n,count)==(data['trials'],v)
                    checks.append(dict(panel=panel,regime=reg,reader=mapping[method],n=n,
                        observed=count,published=v,source=str(path.relative_to(OUT))))
        for split in ['all','held_out']:
            rows=[r for r in panel_rows if split=='all' or r['regime']!='familiar']
            assert len(rows)==(96 if split=='all' else 64)
            assert all(sum(r['task_order']==t for r in rows)==len(rows)//8 for t in range(8))
            for reader in [*SEEDS,'mean']:
                st,ts=analyze(rows,panel,split,reader)
                stats.append(st)
                tasks.extend(ts)
        queries.extend(panel_rows)
    save('outcomes.csv',queries)
    save('aggregates.csv',checks)
    save('statistics.csv',stats)
    save('tasks.csv',tasks)
    save('sources.csv',sources)
    pins = sorted({p.rsplit('/live/',1)[0]+'/launch.txt' for r in sources for p in
                   (r['seed41_and_hs'],r['seeds42_43'])})
    (HERE/'runtime.txt').write_text(ssh('uname -sm; hostname; python3 -m sqlite3 -v; '+
        'for f in '+' '.join(map(Q,pins))+'; do echo "$f"; cat "$f"; done')+'\n')

if __name__=='__main__':
    main()
