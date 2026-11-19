"""Score frozen preferences against physical query alternatives, retaining tied choices."""
from itertools import product
import json
from pathlib import Path
import sys
import numpy as np

from choices import choices,read_tape
from reader import MODES,prior,predict

ARCHIVE=prior.ROOT/'libero-interrupted-practice-20261119-r1'


def read_practices(folder,poses,alternate=None,mode='interrupt'):
    result={}
    for p,g,b in product(poses,(0,1),(False,True)):
        root=alternate if b and alternate is not None else folder
        result[p,g,b]=read_tape(root/f'p{p}-g{g}-{mode if b else "sham"}')
    return result


def queries(folder,poses):
    return {p:np.stack([read_tape(folder/f'p{p}-g{g}-query')['hits'] for g in (0,1)]) for p in poses}


def evaluate(model,library,pose_queries,rows):
    methods=list(rows[0]['ps'])
    physical=[]
    for (p,g,b),t in library.items():
        hits=t['hits'].any(axis=0)
        physical.append(dict(p=p,g=g,b=b,own=bool(hits[g]),other=bool(hits[1-g]),
                             pred=predict(model,t['issued'],t['states'])))
    def summary(rs):
        out=dict(n=len(rs),native={},correct={},final={})
        for m in methods:
            values=np.zeros(3)
            for r in rs:
                g,h=r['ps'][m],r['h']
                branch=pose_queries[r['q']][:,:,h]
                values += (g*branch[0].any()+(1-g)*branch[1].any(),
                           g if h==0 else 1-g,g*branch[0,-1]+(1-g)*branch[1,-1])
            for k,v in zip(('native','correct','final'),values):out[k][m]=float(v)
        out['ceiling']=sum(int(pose_queries[r['q']][:,:,r['h']].any()) for r in rs)
        return out
    stats={m:float(np.mean([(r['pred'][m]-r['own'])**2 for r in physical])) for m in (*MODES,'jaw')}
    return dict(all=summary(rows),zero=summary([r for r in rows if not any(r['r'])]),
                by_query={str(p):summary([r for r in rows if r['q']==p]) for p in pose_queries},brier=stats,
                own_practice={str(b):sum(r['own'] for r in physical if r['b']==b) for b in (False,True)},
                off_target=sum(r['other'] for r in physical),
                query_matrix={str(p):t.any(axis=1).astype(int).tolist() for p,t in pose_queries.items()})


def main(worker,out,stage):
    with np.load(worker/'model.npz') as z:model=dict(z)
    report=dict(fit={k:model[k].tolist() for k in ('weights','bands','losses')})
    if stage=='prior':
        poses=list(range(24,30))
        base=ARCHIVE/'closed01/episodes'
        q=queries(base,poses)
        for name,alternate,mode in [('custom_live',None,'interrupt'),('D18_live',ARCHIVE/'release02/episodes','D18')]:
            library=read_practices(base,poses,alternate,mode)
            rows=choices(model,library,poses)
            report[name]=evaluate(model,library,q,rows)
        poses=list(range(6,12))
        library={(p,g,b):read_tape(ARCHIVE/'pair01/branches'/f'bottle-p{p}-g{g}-d0-{int(b)}.npz')
                 for p,g,b in product(poses,(0,1),(False,True))}
        q={}
        for p in poses:
            records=[]
            for g in (0,1):
                paths=[path for path in (prior.BOTTLE/f's{p-6}-BASE').glob(f'support-g{g}-*')
                       if not json.loads((path/'result.json').read_text())['paced']]
                assert len(paths)==1
                records.append(np.load(paths[0]/'evaluator-predicates.npy'))
            q[p]=np.stack(records)
        report['archived_bottle']=evaluate(model,library,q,choices(model,library,poses))
        for name,file,panel in [('custom_live','closed.json',None),('D18_live','released-live.json',None),
                                 ('archived_bottle','replay.json','bottle')]:
            prev=json.loads((Path(prior.__file__).parent/file).read_text())
            if panel:prev=prev[panel]
            for key in ('all','zero'):
                assert report[name][key]['n']==prev[key]['n']
                assert report[name][key]['native']['table']==prev[key]['native']['table']
    else:
        poses=list(range(6));folder=worker/'episodes'
        library=read_practices(folder,poses)
        q=queries(folder,poses)
        saved=json.loads((folder/'choices.json').read_text())
        rows=choices(model,library,poses)
        assert rows==saved['rows'] and len(rows)==96
        cutoff=(folder/'choices.json').stat().st_mtime_ns
        for p,g in product(poses,(0,1)):
            assert (folder/f'p{p}-g{g}-interrupt/result.json').stat().st_mtime_ns<=cutoff
            assert (folder/f'p{p}-g{g}-query/result.json').stat().st_mtime_ns>=cutoff
        observed=[]
        for a in saved['actual']:
            p,g,h=a['q'],a['selected'],a['h']
            assert (folder/f'p{p}-g{g}-query/result.json').stat().st_mtime_ns < (folder/f'p{p}-g{1-g}-query/result.json').stat().st_mtime_ns
            observed.append(dict(pose=p,chosen=g,hidden=h,rewards=a['r'],success=bool(q[p][g,:,h].any())))
        events=[]
        for p,g in product(poses,(0,1)):
            path=folder/f'p{p}-g{g}-interrupt'
            record=json.loads((path/'result.json').read_text())
            events.append(dict(pose=p,goal=g,start=record['start'],obj=record['object'],own=record['ever'][g]))
        report['object']=evaluate(model,library,q,rows)
        report.update(executed=observed,events=events)
    (out/f'{stage}.json').write_text(json.dumps(report,indent=2)+'\n')
    for name,result in report.items():
        if isinstance(result,dict) and 'all' in result:print(name,result['all'],flush=True)


if __name__=='__main__':main(Path(sys.argv[1]),Path(sys.argv[2]),sys.argv[3])
