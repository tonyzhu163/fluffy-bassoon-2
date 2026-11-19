"""Three-neighbor, goal-conditioned reachability from pre-action public features."""
import json
from pathlib import Path

import numpy as np


def features(obs, proposals):
    images = [obs[k].reshape(28,8,28,8,3).mean(axis=(1,3)) for k in ('observation/image','observation/wrist_image')]
    visual = np.asarray(images,dtype=np.float32).reshape(-1)/255.
    plan = np.stack([proposals.mean(axis=0),proposals.std(axis=0)]).reshape(-1)
    return visual,plan


def distance(table, row):
    return np.sqrt(np.mean((table-row)**2,axis=-1))


class Reachability:
    def __init__(self, features_file):
        with np.load(features_file) as fit:
            self.images = fit['images']
            self.plans = fit['plans']
            self.goals = fit['goals']
            self.targets = fit['targets']
            self.scale = fit['scale']

    def score(self, goal, visual, plan):
        pool = np.flatnonzero(self.goals==goal)
        if not len(pool):
            pool = np.arange(len(self.goals))
        d = distance(self.images[pool],visual)/self.scale[0] + distance(self.plans[pool],plan)/self.scale[1]
        near = pool[np.argsort(d,kind='stable')[:3]]
        return self.targets[near].mean(axis=0).tolist()


def fit(source, output):
    images,plans,goals,targets = [],[],[],[]
    for f in sorted(source.glob('*/*/g*/result.json')):
        row=json.loads(f.read_text())
        goal=row['goal']
        t=row['first'][str(goal)]
        with np.load(f.parent/'initial.npz') as a:
            images.append(a['visual']);plans.append(a['plan'])
        goals.append(goal);targets.append([float(t is not None),300-t if t is not None else 0])
    assert len(goals)==64
    images,plans=np.array(images),np.array(plans)
    scales=[]
    for table in (images,plans):
        dist=np.concatenate([distance(table[i+1:],x) for i,x in enumerate(table[:-1])])
        nonzero=dist[dist>0]
        scales.append(float(np.median(nonzero)) if len(nonzero) else 1.)
    np.savez_compressed(output,images=images,plans=plans,goals=goals,targets=targets,scale=scales)
    return Reachability(output)
