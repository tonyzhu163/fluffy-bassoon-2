"""Audit recorded endpoints and the native two-practice correction protocol on Baige CPU."""
import argparse
import hashlib
import json
import os
import platform
import subprocess
from itertools import combinations, permutations, product
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

BOWL = "libero-visual-witness-20261119-r1"
BOTTLE = "libero-witness-transfer-20261119-r1"
VARIANTS = ("BASE", "D18")
GOALS = (0, 1)
OUT = Path(__file__).resolve().parents[2]
MODES = ("agent", "wrist_camera", "rgb", "eef_xyz", "eef_quat_xyzw", "jaw", "state", "joint")
ARMS = ("terminal_rgb", "terminal_state")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(value):
    return hashlib.sha256(value).hexdigest()


def tag(key):
    return "p{}-g{}-{}".format(*key)


def terminal_rgb(frame):
    # Same camera order, preprocessing and 2x2 grid as the five-frame RGB arm.
    x = torch.from_numpy(frame).permute(0, 3, 1, 2).float()
    x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False, antialias=True)
    x = x.div_(127.5).sub_(1).to(torch.bfloat16).float()
    return F.adaptive_avg_pool2d(x, (2, 2)).numpy().astype(np.float64).ravel()


def read(directory, key, manifest):
    name = tag(key) + ".npz"
    path = directory / name
    require(digest(path.read_bytes()) == manifest[name], f"Tape hash mismatch: {path}")
    with np.load(path, allow_pickle=False) as z:
        frames, states, hits = (z[k] for k in ("frames", "states", "hits"))
        commands, initial = z["issued"], z["initial_state"]
        require(frames.shape == (13, 2, 96, 96, 3) and frames.dtype == np.uint8, name)
        require(states.shape == (300, 9) and np.isfinite(states).all(), name)
        require(commands.shape == (300, 7) and initial.shape == (9,), name)
        require(hits.shape == (300, 2) and np.isin(hits, (0, 1)).all(), name)
        ever = hits.any(axis=0).tolist()
        require(not ever[1-key[1]], f"Off-target hit: {name}")
        frame, state = frames[-1].copy(), states[-1].copy()
        start = (str(z["physics"].item()), digest(frames[0].tobytes()), digest(initial.tobytes()))
        return dict(frame=frame, state=state, start=start, commands=digest(commands.tobytes()),
                    ever=ever, terminal_rgb=terminal_rgb(frame), terminal_state=state.astype(float))


def compare(a, b):
    changed = np.any(a["frame"] != b["frame"], axis=-1).sum(axis=(1, 2)).tolist()
    delta = np.abs(a["state"].astype(float) - b["state"].astype(float))
    agent, wrist = [n == 0 for n in changed]
    xyz, quat, jaw = [bool((d == 0).all()) for d in (delta[:3], delta[3:7], delta[7:])]
    state = xyz and quat and jaw
    flags = dict(zip(MODES, (agent, wrist, agent and wrist, xyz, quat, jaw, state,
                             agent and wrist and state)))
    return dict(equal=flags, changed_pixels_agent_wrist=changed, max_abs_state_delta=float(delta.max()))


def inventory(tapes, poses, query):
    pairs = {}
    collisions = {mode: [] for mode in MODES}
    for left, right in combinations(tapes, 2):
        result = compare(tapes[left], tapes[right])
        for mode, same in result["equal"].items():
            if same:
                collisions[mode].append([tag(left), tag(right)])
    for p in poses:
        require(len({r["start"] for k, r in tapes.items() if k[0] == p}) == 1, f"Start {p}")
        for g in GOALS:
            base, drop = [tapes[p, g, v] for v in VARIANTS]
            require(base["commands"] == drop["commands"], f"Commands {p}/{g}")
            if query == "QUERY":
                require(base["commands"] != tapes[p, g, query]["commands"], f"Query {p}/{g}")
            pairs[p, g] = compare(base, drop)
    return pairs, dict(tapes=len(tapes), unordered_tape_pairs=len(tapes)*(len(tapes)-1)//2,
                       exact_counts={m: len(rs) for m, rs in collisions.items()}, exact_pairs=collisions,
                       same_command_pairs=len(pairs),
                       same_command_exact={m: sum(r["equal"][m] for r in pairs.values()) for m in MODES},
                       base_d18={f"p{p}-g{g}": r for (p, g), r in pairs.items()})


def fit_predict(bowl, tapes, arm):
    source = [(r[arm], r["ever"][g]) for (p, g, _), r in bowl.items() if p < 16]
    x = np.stack([v for v, y in source])
    labels = np.array([y for v, y in source])
    require(len(x) == 16 and labels.sum() == 8, "Source fit changed")
    scale = 1.
    if arm == "terminal_state":
        sd = x.std(axis=0)
        scale = np.where(sd > 1e-12, sd, 1.)
    miss, hit = [(x[labels == y] / scale).mean(axis=0) for y in (False, True)]
    scores = {}
    for k, row in tapes.items():
        if k[2] == "QUERY":
            continue
        d0, d1 = [float(np.linalg.norm(row[arm] / scale - c)) for c in (miss, hit)]
        scores[k] = d0 / (d0 + d1) if d0 + d1 else .5
    return scores


def prefer(a, b):
    return .5 if abs(a-b) <= 1e-12 else float(a < b)


def cyclic(poses):
    return [(a, poses[(i+1) % len(poses)], poses[(i+2) % len(poses)]) for i, a in enumerate(poses)]


def panel(tapes, pairs, triples, orders, query, scores):
    groups, rows = [], []
    totals = {arm: dict(strict=0., strict_swapped=0., all=0.) for arm in scores}
    for (a, b, q), first in product(triples, orders):
        order = (first, 1-first)
        slots = ((a, first), (b, 1-first))
        eligible = all(tapes[p, g, "BASE"]["ever"][g] and not tapes[p, g, "D18"]["ever"][g]
                       for p, g in slots) and all(tapes[q, g, query]["ever"][g] for g in GOALS)
        same = {m: all(pairs[k]["equal"][m] for k in slots) for m in MODES}
        groups.append(dict(triple=[a,b,q], order=order, eligible=eligible, same_endpoints=same))
        for va, vb, h in product(VARIANTS, VARIANTS, GOALS):
            keys = ((a, first, va), (b, 1-first, vb))
            rewards = [int(tapes[k]["ever"][h]) for k in keys]
            native = [int(tapes[q, g, query]["ever"][h]) for g in order]
            strict = eligible and va != vb and (va, vb)[order.index(h)] == "D18"
            if strict:
                require(rewards == [0, 0] and sum(native) == 1, "Invalid strict contrast")
            record = dict(triple=[a,b,q], order=order, variants=[va,vb], hidden=h, rewards=rewards,
                          query_hits=native, strict=strict, joint_match=same["joint"], scores={})
            for arm, s in scores.items():
                pa = rewards[0] if any(rewards) else prefer(s[keys[0]], s[keys[1]])
                credit = pa*native[0] + (1-pa)*native[1]
                record["scores"][arm] = dict(native=credit)
                totals[arm]["all"] += credit
                if strict:
                    ps = prefer(s[a, first, vb], s[b, 1-first, va])
                    swapped = ps*native[0] + (1-ps)*native[1]
                    totals[arm]["strict"] += credit
                    totals[arm]["strict_swapped"] += swapped
                    record["scores"][arm]["swapped"] = swapped
            rows.append(record)
    strict_groups = [g for g in groups if g["eligible"]]
    matched = {m: sum(g["same_endpoints"][m] for g in strict_groups) for m in MODES}
    return dict(candidate_groups=len(groups), eligible_groups=len(strict_groups),
                eligible_rows=2*len(strict_groups), all_rows=len(rows), matched_eligible_groups=matched,
                matched_eligible_rows={m: 2*n for m,n in matched.items()},
                native_correct=totals, groups=groups, cases=rows)


def load(root, run, attempt, poses, variants):
    lines = (OUT / run / "tapes.sha256").read_text().splitlines()
    manifest = {name: sha for sha, name in (line.split() for line in lines)}
    keys = list(product(poses, GOALS, variants))
    require(set(manifest) == {tag(k)+".npz" for k in keys}, "Unexpected manifest")
    return {k: read(root/run/attempt, k, manifest) for k in keys}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pin", required=True)
    ap.add_argument("--data", type=Path, default=Path("/mowenjie/fluffy-bassoon-2/outputs"))
    ap.add_argument("--output", required=True, type=Path)
    args = ap.parse_args()
    path = Path(__file__).resolve()
    git = ["git", "-C", str(OUT.parent)]
    head = subprocess.check_output(git + ["rev-parse", "HEAD"], text=True).strip()
    require(head == args.pin, "Wrong deployed Git commit")
    for p in (path, OUT/BOWL/"tapes.sha256", OUT/BOTTLE/"tapes.sha256", OUT/BOTTLE/"rgb.json"):
        saved = subprocess.check_output(git + ["show", f"{head}:{p.relative_to(OUT.parent)}"])
        require(saved == p.read_bytes(), f"Working bytes differ from pin: {p}")
    require(platform.system() == "Linux" and os.environ.get("CUDA_VISIBLE_DEVICES") == "", "CPU launch")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    bowl = load(args.data, BOWL, "cpu02", range(12,20), VARIANTS)
    bottle = load(args.data, BOTTLE, "cpu01", range(24,30), (*VARIANTS,"QUERY"))
    published = json.loads((OUT/BOTTLE/"rgb.json").read_text())
    result = dict(revision=head, host=platform.node(), torch=str(torch.__version__), numpy=np.__version__,
                  device="cpu", fit="16 bowl tapes, poses 12-15, both goals pooled, 8 hits/8 misses",
                  data=str(args.data), panels={})
    for name, tapes, poses, query in (("bowl",bowl,range(12,20),"BASE"),
                                     ("bottle",bottle,range(24,30),"QUERY")):
        pairs, result[name] = inventory(tapes, poses, query)
        scores = {arm: fit_predict(bowl, tapes, arm) for arm in ARMS}
        if name == "bottle":
            scores["published_five_frame_rgb"] = {(r["pose"],r["goal"],r["variant"]): r["score"]
                                                   for r in published["physical_practices"]}
        result[name]["practice_scores"] = {tag(k): {arm:s[k] for arm,s in scores.items()}
                                          for k in scores["terminal_rgb"]}
        target = [k for k in scores["terminal_rgb"] if k[0] >= 16]
        result[name]["completion"] = {arm: dict(n=len(target), correct=sum(
            prefer(.5,s[k]) if tapes[k]["ever"][k[1]] else 1-prefer(.5,s[k]) for k in target))
                                      for arm,s in scores.items()}
        panels = ({"bowl_fit_cyclic": cyclic(range(12,16)), "bowl_fit_ordered": permutations(range(12,16),3),
                   "bowl_dev_cyclic": cyclic(range(16,20)), "bowl_dev_ordered": permutations(range(16,20),3)}
                  if name == "bowl" else {"bottle_cyclic": cyclic(poses)})
        for label, triples in panels.items():
            result["panels"][label] = panel(tapes, pairs, triples, (0,) if name=="bowl" else GOALS, query, scores)
    target_panel = result["panels"]["bottle_cyclic"]
    def case_id(row, order_field):
        return (*row["triple"], *row[order_field], *row["variants"], row["hidden"])
    earlier = {case_id(r,"prompt_order"): r for r in published["cases"]}
    for row in target_panel["cases"]:
        old = earlier[case_id(row,"order")]
        require(all(row[k] == old[k] for k in ("rewards","query_hits","strict")), "Published case mismatch")
        score = row["scores"]["published_five_frame_rgb"]
        require(score["native"] == old["native"] and score.get("swapped") == old.get("swapped_native"),
                "Published native credit mismatch")
    for key, expected in (("bowl_dev_cyclic",4), ("bowl_dev_ordered",24), ("bottle_cyclic",12)):
        require(result["panels"][key]["eligible_rows"] == expected, f"Eligible row count: {key}")
    with args.output.open("x") as f:
        json.dump(result, f, indent=2, allow_nan=False)
        f.write("\n")
    for name, p in result["panels"].items():
        print(name, p["matched_eligible_rows"]["joint"], "/", p["eligible_rows"], p["native_correct"])


if __name__ == "__main__":
    main()
