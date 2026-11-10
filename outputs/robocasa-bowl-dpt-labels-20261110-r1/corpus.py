"""Read public support histories, public query views and masked physical branch targets."""
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

from readout import read

PROMPT = "Put one fruit from the plate into a bowl. Release it and move your hand away."
SIDES = ("left","right")


def given_side_prompt(goal):
    return f"Put one fruit from the plate into the {SIDES[goal]} bowl. Release it and move your hand away."

def transitions(archive):
    offsets = archive["offsets"]
    public = archive["public"]
    actions = archive["commands"]
    assert len(public) == len(offsets) and offsets[-1] == len(actions)
    sequence = []
    for index, (start,end) in enumerate(zip(offsets[:-1],offsets[1:])):
        valid = end-start
        assert 1 <= valid <= 5
        delivered = np.zeros((5,12),np.float32)
        delivered[:valid] = actions[start:end]
        delivered[:valid,[6,11]] = np.where(delivered[:valid,[6,11]] >= .5,1.,-1.)
        mask = np.arange(5) < valid
        sequence.append(np.r_[public[index],delivered.ravel(),mask,0.,0.,0.].astype(np.float32))
    return sequence


def public_history(scene, probe, goal):
    # Only the evaluator constructs this native scalar reward. The model receives its value.
    side = SIDES[probe]
    rewarded = bool(read(scene/f"{side}.json")["placed"][goal])
    with np.load(scene/"pickup.npz") as prefix, np.load(scene/f"{side}.npz") as branch:
        trajectory = transitions(prefix) + transitions(branch)
        terminal = np.r_[branch["public"][-1],np.zeros(65,np.float32),float(rewarded),0.,1.]
    trajectory.append(terminal.astype(np.float32))
    x = np.stack(trajectory)
    x[0,-2] = 1.
    assert x.shape[1] == 228 and x[0,-2] == x[-1,-1] == 1 and len(x) <= 102
    return x


def observation(frames, states):
    assert frames.shape == (3,256,256,3) and frames.dtype==np.uint8 and states.shape==(16,)
    return {"main_images":torch.from_numpy(frames[None,0].copy()),
            "wrist_images":torch.from_numpy(frames[None,1].copy()),
            "extra_view_images":torch.from_numpy(frames[None,2:3].copy()),
            "states":torch.from_numpy(states[None].astype(np.float32)), "task_descriptions":[PROMPT]}


def normalize_history(history, mean, scale):
    # The decoder consumes a contiguous prefix and reads its last real (terminal) token.
    out=np.zeros((102,len(mean)+1),np.float32)
    out[:len(history),:-1]=np.clip((history-mean)/scale,-12,12)
    out[:len(history),-1]=1.
    return out


@lru_cache(maxsize=72)
def query_archive(path):
    with np.load(path) as archive:
        return archive["frames"],archive["states"]


class Corpus:
    def __init__(self, source, labels):
        self.source = Path(source)
        self.labels = Path(labels)
        index = read(self.labels/"index.json")
        self.rows = [r for r in index["labels"] if r["label"]]
        self.scenes = sorted({r["seed"] for r in index["labels"]})
        self.supports = {(s,p,g):public_history(self.source/str(s),p,g)
                         for s in self.scenes for p in (0,1) for g in (0,1)}
        # Exclude unusable targets from training but keep their support trajectories.
        self.targets = {}
        for r in self.rows:
            with np.load(self.labels/r["label"]) as archive:
                self.targets[r["seed"],r["goal"]] = (archive["target"],archive["delivered"])

    def statistics(self):
        all_tokens = np.concatenate(list(self.supports.values()))
        return all_tokens.mean(0), all_tokens.std(0).clip(.1)

    def sample(self, rng):
        goal = int(rng.integers(2))
        possible = [r for r in self.rows if r["goal"]==goal]
        r = possible[int(rng.integers(len(possible)))]
        index = int(rng.integers(r["valid_rows"]))
        if r["matched_decision_index"] is not None and rng.random() < .5:
            index = r["matched_decision_index"]
        support = int(rng.choice([s for s in self.scenes if s!=r["seed"]]))
        return r, index, support, int(rng.integers(2))

    def get(self, row, index, support, probe, mean, scale, support_goal=None):
        assert support != row["seed"]
        goal=row["goal"]
        frames,states=query_archive(row["query_archive"])
        target,delivered=self.targets[row["seed"],goal]
        history=self.supports[support,probe,goal if support_goal is None else support_goal]
        return observation(frames[index],states[index]),target[index],delivered[index],normalize_history(history,mean,scale)
