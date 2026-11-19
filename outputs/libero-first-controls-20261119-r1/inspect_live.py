"""Recompute native endpoints and the live decision from the verified public prefixes."""
from pathlib import Path
import json,sys
import numpy as np


def decision(file,cut):
    with np.load(file) as tape:
        delta=np.linalg.norm(tape['positions'][:7].astype(float)-tape['initial_position'],axis=1)
        ratio=float(delta[6]/max(1e-12,max(delta[:6])))
        return ('paced' if ratio<cut else 'default'),ratio


def main(root):
    load=lambda p:json.loads(p.read_text())
    cut=load(root/'frozen.json')['cut']
    rows=load(root/'rows.json')
    assert len(rows)==load(root/'complete.json')['rows']==16
    for row in rows:
        path=root/f"{row['suite']}-{row['task']}-{row['timing']}"
        prior,score=decision(path/'support/public.npz',cut)
        assert (prior,score)==(row['prior_mode'],row['prior_score'])
        files={k:path/k/'public.npz' for k in ('default','paced')}
        files['prior']=files[prior]
        for key,base in [('ten_first','paced'),('default_first','default')]:
            selected,score=decision(files[base],cut)
            assert row['choices'][key]==dict(score=score,mode=selected,alias=selected==base)
            files[key]=files[base] if selected==base else path/key/'executed.npz'
        for k,file in files.items():
            with np.load(file) as tape:
                rewards=tape['rewards']; assert len(rewards)==row['controls'][k]
                assert bool((rewards>0).any())==row['hits'][k]
                if file.name=='executed.npz':
                    base='paced' if k=='ten_first' else 'default'
                    with np.load(files[base]) as old:
                        np.testing.assert_array_equal(old['issued'][:10],tape['issued'][:10])
    keys=rows[0]['hits']
    scores={g:{k:sum(t['hits'][k] for t in rows if t['timing']==g) for k in keys} for g in ('BASE','S20')}
    paired={k:dict(wins=sum(r['hits']['prior'] and not r['hits'][k] for r in rows),
                   losses=sum(r['hits'][k] and not r['hits']['prior'] for r in rows)) for k in keys}
    return dict(scores=scores,paired_prior=paired,rows=rows,episodes=load(root/'complete.json')['episodes'],threshold=cut)


if __name__=='__main__':
    print(json.dumps(main(Path(sys.argv[1])),indent=2))
