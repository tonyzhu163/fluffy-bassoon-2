"""Read the hidden side from a fixed student's terminal support representation."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from bowl_students import terminal_features
from corpus import SIDES, normalize_history, public_history
from eval_bowl import plans
from fluffy_bassoon.meta_rl.causal_decoder import CausalDecoderCore
from history_check import predict, public_features
from readout import read


def examples(folder, training):
    seeds = read(folder.parent / (folder.name + "-geometry") / "admitted.json")["admitted"]
    if training:
        cases = [(None, seed, probe) for seed in seeds for probe in (0, 1)]
    else:
        cases = []
        for i, query in enumerate(seeds):
            plan = next(plans(seeds, i, ["own"]))
            cases.append((query, plan["support"], plan["probe"]))
    rows, tapes, movement = [], [], []
    for query, support, probe in cases:
        scene = folder / str(support)
        side = SIDES[probe]
        placed = read(scene / f"{side}.json")["placed"]
        paired = [public_history(scene, probe, goal) for goal in (0, 1)]
        assert np.count_nonzero(paired[0] != paired[1]) == int(any(placed))
        with np.load(scene / "pickup.npz") as prefix, np.load(scene / f"{side}.npz") as branch:
            for goal, tape in enumerate(paired):
                reward = int(placed[goal])
                assert tape[-1, -3] == reward
                movement.append(public_features(branch, prefix, reward))
                tapes.append(tape)
                rows.append(dict(query=query, support=support, probe=probe, goal=goal,
                                 reward=reward, informative=any(placed)))
    return rows, tapes, np.stack(movement)


@torch.inference_mode()
def embeddings(weights, histories):
    mean, scale = weights["mean"], weights["scale"]
    encoder = CausalDecoderCore(len(mean), width=128, memory_length=102, layers=2, heads=4).eval()
    encoder.load_state_dict({k.removeprefix("encoder."): v for k, v in weights["students"].items()
                             if k.startswith("encoder.")})
    blocks = []
    for start in range(0, len(histories), 8):
        group = [normalize_history(t, mean, scale) for t in histories[start:start + 8]]
        blocks.append(terminal_features(encoder, torch.from_numpy(np.stack(group))).numpy())
    return np.concatenate(blocks).astype(np.float64)


def measure(args):
    torch.set_num_threads(1)
    weights = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    assert weights["contract"]["history_encoder"] and not weights["contract"].get("given_side_in_prompt")
    tr, train_tapes, train_motion = examples(args.train, training=True)
    ev, eval_tapes, eval_motion = examples(args.test, training=False)
    assert {r["support"] for r in tr}.isdisjoint({r["query"] for r in ev} | {r["support"] for r in ev})
    latent = np.array([r["goal"] for r in tr])
    features = embeddings(weights, train_tapes + eval_tapes)
    for name, train, held in (("encoder", features[:len(tr)], features[len(tr):]),
                              ("public_motion", train_motion, eval_motion)):
        choices = (predict(train, latent, held) > .5).astype(int)
        for i, row in enumerate(ev):
            other = i ^ 1
            assert ev[other]["support"] == row["support"] and ev[other]["goal"] != row["goal"]
            zero = i if row["reward"] == 0 else other
            assert ev[zero]["reward"] == 0
            row[name] = int(choices[i])
            row[name + "_flipped"] = int(choices[other])
            row[name + "_masked"] = int(choices[zero])
    def summary(rows):
        return {"n": len(rows), **{key:sum(r[key] == r["goal"] for r in rows)
                for name in ("encoder", "public_motion") for key in (name, name+"_flipped", name+"_masked")}}
    return dict(step=weights["step"], fit_seed=weights["contract"]["seed"], training_rows=len(tr),
                training_scenes=len({r["support"] for r in tr}), split=args.test.name,
                all=summary(ev), placed=summary([r for r in ev if r["informative"]]),
                timeout=summary([r for r in ev if not r["informative"]]), cases=ev)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for field in ("checkpoint", "train", "test", "output"):
        parser.add_argument(field, type=Path)
    args = parser.parse_args()
    report = measure(args)
    with args.output.open("x") as f:
        json.dump(report, f, indent=2)
    print(json.dumps({k:v for k,v in report.items() if k != "cases"}))
