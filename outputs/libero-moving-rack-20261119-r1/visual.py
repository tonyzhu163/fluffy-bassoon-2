"""Current-camera control with a common placement prefix and source-only pixel classifier."""
from concurrent.futures import ProcessPoolExecutor
from itertools import product
from pathlib import Path
import sys
import numpy as np
import json
import pilot


def image(sim,renderer):
    # Mean-pool the public RGB stream to limit spatial sensitivity to a single pose.
    renderer.update_scene(sim.data._data,camera='agentview')
    rgb=renderer.render()
    return rgb.reshape(32,4,32,4,3).mean(axis=(1,3)).astype(np.float32)


class Camera:
    def __init__(self, model):
        self.model=model
        self.delta=None
        self.choice=None

    def start(self,sim):
        import mujoco
        self.renderer=mujoco.Renderer(sim.model._model,height=128,width=128)
        self.initial=image(sim,self.renderer)

    def choose(self,sim):
        self.delta=image(sim,self.renderer)-self.initial
        self.choice='suffix'
        if self.model is not None:
            midpoint,direction=self.model
            if np.sum((self.delta-midpoint)*direction)>0:
                self.choice='shift'
        return self.choice


def trial(job):
    pose,g,model=job
    camera=Camera(model)
    row=pilot.simulate((pose,g,'suffix'),camera=camera)
    camera.renderer.close()
    row['camera_choice']=camera.choice
    return row,camera.initial,camera.delta


def train(out):
    out.mkdir()
    with ProcessPoolExecutor(max_workers=3) as pool:
        rows=list(pool.map(trial,((p,g,None) for p,g in product(range(12),pilot.REGIMES))))
    groups={g:[d for r,_,d in rows if r['regime']==g and d is not None] for g in pilot.REGIMES}
    base,moved=[np.mean(groups[g],axis=0) for g in pilot.REGIMES]
    model=((base+moved)/2,moved-base)
    np.savez(out/'model.npz',midpoint=model[0],direction=model[1])
    starts={ (r['pose'],r['regime']):im for r,im,_ in rows}
    summary=dict(identical_reset_images=sum(np.array_equal(starts[p,'BASE'],starts[p,'D02']) for p in range(12)),
                 samples=[len(groups[g]) for g in pilot.REGIMES], rows=[r for r,_,_ in rows])
    (out/'fit.json').write_text(json.dumps(summary,indent=2))


def evaluate(out,openings):
    orig=pilot.old.tape
    pilot.old.tape=lambda p,t,c: openings/f'pose{p}' if p>=12 else orig(p,t,c)
    with np.load(out/'model.npz') as a:
        model=(a['midpoint'],a['direction'])
    with ProcessPoolExecutor(max_workers=3) as pool:
        rows=list(pool.map(trial,((p,g,model) for p,g in product(range(12,24),pilot.REGIMES))))
    pairs={(r['pose'],r['regime']):im for r,im,_ in rows}
    summary=dict(rows=[r for r,_,_ in rows],hit=sum(r['hit'] for r,_,_ in rows),
                 final=sum(r['final'] for r,_,_ in rows),
                 identical_reset_images=sum(np.array_equal(pairs[p,'BASE'],pairs[p,'D02']) for p in range(12)))
    (out/'fresh.json').write_text(json.dumps(summary,indent=2))


if __name__=='__main__':
    {'fit':train,'test':evaluate}[sys.argv[1]](*map(Path,sys.argv[2:]))
