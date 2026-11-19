"""Apply the frozen source decision to new poses, with matched native continuations."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import pilot

def main(openings,out,frozen):
    previous_tape=pilot.old.tape
    def tape(pose,timing,paced):
        if pose>=12:
            assert timing=='BASE'
            return openings/f'pose{pose}'
        return previous_tape(pose,timing,paced)
    pilot.old.tape=tape
    model=json.loads(frozen.read_text())
    out.mkdir()
    with ProcessPoolExecutor(max_workers=5) as pool:
        data=pilot.collect(pool,range(12,24),out/'native.jsonl')
    poses=list(range(12,24))
    pairs=list(zip(poses,poses[1:]+poses[:1]))
    tree=model['tree']
    groups={'history':pilot.score(data,pairs,tree),'opposite':pilot.score(data,pairs,tree,True),
            'fixed':pilot.score(data,pairs,model['constant'])}
    fixed={a:sum(data[q,g,a]['hit'] for q in poses for g in pilot.REGIMES) for a in pilot.OPTIONS}
    oracle=sum(any(data[q,g,a]['hit'] for a in pilot.OPTIONS) for q in poses for g in pilot.REGIMES)
    result=dict(scores={k:{m:sum(r[m] for r in rows) for m in ('hit','final')} for k,rows in groups.items()},
                fixed_options=fixed,oracle=oracle,rows=groups)
    (out/'result.json').write_text(json.dumps(result,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('openings','out','frozen'):
        p.add_argument(name,type=Path)
    main(**vars(p.parse_args()))
