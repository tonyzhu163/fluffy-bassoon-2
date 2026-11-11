"""Compile candidate history-learning cases from procedural LIBERO scenes.

This is a geometry-only authoring tool, not a simulator or a reward oracle.
It uses the existing four-axis rule family, separates scene and rule splits,
and keeps hidden labels out of the learner-facing schedule. Every physical
reset still needs its rules recomputed from the realized geometry. Attempt
feedback must come from native predicates on executed trajectories.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import itertools
import json
from collections import Counter
from pathlib import Path

import numpy as np

from scenes import FIXTURES

_spec = importlib.util.spec_from_file_location(
    "placement", Path(__file__).resolve().parents[1] / "learning-histories/placement.py"
)
placement = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(placement)

TARGETS = ("plate_1", "glazed_rim_porcelain_ramekin_1", "cookies_1",
           "flat_stove_1_cook_region", "wooden_cabinet_1_top_side")
AXES = {
    "pick": ("min_x", "max_x", "min_y", "max_y") + tuple(
        f"{relation}_{name}" for name in placement.REC for relation in ("nearest", "farthest")
    ),
    "pick_if": ("never", "plate_x_lt_ramekin_x", "cookies_y_lt_plate_y", "ramekin_x_lt_cookies_x"),
    "anchor": placement.REC + ("nearest_picked", "farthest_picked", "min_x", "max_x",
                               "min_y", "max_y", "nearest_other", "farthest_other"),
    "anchor_if": ("never", "plate_y_lt_ramekin_y", "cookies_x_lt_plate_x", "ramekin_y_lt_cookies_y"),
}
SPLITS = ("train", "validation", "test")
PROMPT = "Do the demonstrated task"


def rule_table() -> list[dict]:
    return [dict(id=i, **dict(zip(AXES, values)))
            for i, values in enumerate(itertools.product(*AXES.values()))]


def label_scenes(scenes: list[dict]) -> tuple[np.ndarray, list[list[dict]]]:
    """Nominal table-frame labels; never substitute these for physical rewards."""
    bowls, receptacles, catalog = [], [], []
    for scene in scenes:
        objects = {o["name"]: o for o in scene["objects"]}
        names = [o["name"] for o in scene["objects"] if o["role"] == "bowl"]
        if names != ["akita_black_bowl_1", "akita_black_bowl_2"] or set(scene["fixtures"]) != set(FIXTURES):
            raise ValueError("history compiler currently supports the positions preset only")
        bowls.append([objects[n]["xy"] for n in names])
        receptacles.append([objects[n]["xy"] for n in TARGETS[:3]] +
                           [FIXTURES[f]["xy"] for f in ("flat_stove", "wooden_cabinet")])
        by_pair = {(b["bowl"], b["target"]): b for b in scene["behaviours"]}
        catalog.append([by_pair[b, t] for b in names for t in TARGETS])
    encoded, _ = placement.behaviours(np.asarray(bowls), np.asarray(receptacles), ("anchor-if",), False)
    stride = placement.N_PLACE * placement.N_THEN
    if np.any(encoded % stride):
        raise ValueError("unexpected offset or second-step behaviour")
    labels = (encoded // stride).astype(np.uint8)
    if labels.shape[1] != len(rule_table()):
        raise ValueError("rule axes no longer match placement.behaviours")
    return labels, catalog


def partition(items: list, rng) -> dict[str, list]:
    order = rng.permutation(len(items))
    a, b = 3 * len(items) // 4, 7 * len(items) // 8
    return {name: [items[int(i)] for i in ids]
            for name, ids in zip(SPLITS, (order[:a], order[a:b], order[b:]))}


def compile_cases(source: Path, out: Path, counts: tuple[int, int, int], k: int, seed: int) -> dict:
    if k < 1 or any(n < 1 for n in counts):
        raise ValueError("use at least one attempt and one session per split")
    manifest = json.loads((source / "manifest.json").read_text())
    if manifest["preset"] != "positions":
        raise ValueError("use the positions preset; object-family extensions need new rule bindings")
    scenes = manifest["scenes"]
    hashes = [s["bddl_sha256"] for s in scenes]
    if len(set(hashes)) != len(hashes) or len({s["seed"] for s in scenes}) != len(scenes):
        raise ValueError("duplicate scenes or reset seeds would contaminate the layout split")
    for scene in scenes:
        if hashlib.sha256((source / scene["bddl"]).read_bytes()).hexdigest() != scene["bddl_sha256"]:
            raise ValueError(f"scene file changed: {scene['bddl']}")

    labels, catalog = label_scenes(scenes)
    rules = rule_table()
    # Group aliases empirically across the whole bank before splitting. Distinct
    # syntax alone does not establish distinct hidden functions.
    signatures: dict[bytes, list[int]] = {}
    for r in range(len(rules)):
        signatures.setdefault(labels[:, r].tobytes(), []).append(r)
    groups = list(signatures.values())
    family_reps = np.array([g[0] for g in groups])
    rng = np.random.default_rng(seed)
    rule_splits = partition(groups, rng)
    scene_splits = partition(list(range(len(scenes))), rng)
    if any(len(v) < k + 2 for v in scene_splits.values()):
        raise ValueError(f"each layout split needs at least {k + 2} distinct scenes; generate more scenes")
    if any(len(v) < 3 for v in rule_splits.values()):
        raise ValueError("each rule split needs at least three distinct functions")
    train_ids = [r for group in rule_splits["train"] for r in group]
    for axis, values in AXES.items():
        if {rules[r][axis] for r in train_ids} != set(values):
            raise ValueError(f"training split misses a primitive on axis {axis}; use another split seed")

    out.mkdir(parents=True, exist_ok=False)
    (out / "evaluator").mkdir()
    (out / "schedule").mkdir()
    (out / "scenes").mkdir()
    for scene in scenes:
        (out / "scenes" / scene["bddl"]).write_bytes((source / scene["bddl"]).read_bytes())

    summaries = {}
    for split, n in zip(SPLITS, counts):
        # Proposals are uniform over functions, not aliases. Acceptance below
        # conditions the resulting bank on ambiguity and identifiability.
        reps = np.array([g[0] for g in rule_splits[split]])
        pool = np.array(scene_splits[split])
        stats = []
        rejected = Counter()
        with (out / "schedule" / f"{split}.jsonl").open("w") as public, \
                (out / "evaluator" / f"{split}.jsonl").open("w") as private:
            for i in range(n):
                for _ in range(10000):
                    hidden = int(rng.choice(reps))
                    ids = rng.choice(pool, k + 2, replace=False)
                    demo, query = int(ids[0]), int(ids[-1])
                    # The split chooses the hidden rule, not what the observer
                    # knows. Use the full family when measuring ambiguity.
                    alive = family_reps[labels[demo, family_reps] == labels[demo, hidden]]
                    continuation = labels[np.ix_(ids[1:], alive)]
                    distinct = len({continuation[:, j].tobytes() for j in range(len(alive))})
                    query_choices = np.unique(labels[query, alive])
                    if distinct < 3:
                        rejected["fewer_than_three_continuations"] += 1
                        continue
                    if len(query_choices) < 2:
                        rejected["demo_already_determines_query"] += 1
                        continue
                    # Full correct behaviours are an information ceiling. A
                    # success bit after a failed motor action can reveal less.
                    support = labels[np.ix_(ids[1:-1], alive)]
                    consistent = alive[np.all(support == labels[ids[1:-1], hidden, None], axis=0)]
                    if len(np.unique(labels[query, consistent])) != 1:
                        rejected["support_cannot_identify_query_even_with_corrections"] += 1
                        continue
                    break
                else:
                    raise ValueError(f"cannot construct an informative {split} session {i}; see partial output")

                case_id = f"{split}-{i:05d}"
                # These are orchestration references, not student model tokens.
                scene_refs = [f"scenes/{scenes[int(j)]['bddl']}" for j in ids]
                schedule = dict(id=case_id, instruction=PROMPT, demonstration=scene_refs[0],
                                attempts=scene_refs[1:-1], query=scene_refs[-1])
                goals = [catalog[int(j)][int(labels[j, hidden])] for j in ids]
                answer = dict(id=case_id, hidden_rule=hidden, scene_indices=[int(j) for j in ids],
                              demonstration_goal=goals[0], attempt_goals=goals[1:-1], query_goal=goals[-1],
                              hypotheses_after_demo=len(alive), distinct_continuations=distinct,
                              query_choices_after_demo=len(query_choices),
                              query_choices_after_corrected_support=1)
                public.write(json.dumps(schedule) + "\n")
                private.write(json.dumps(answer) + "\n")
                stats.append((len(alive), distinct, len(query_choices)))

        summaries[split] = dict(
            sessions=n, scenes=len(pool), distinct_rule_functions=len(reps),
            syntactic_rules=sum(map(len, rule_splits[split])),
            after_demo={name: dict(min=int(min(v)), median=float(np.median(v)), max=int(max(v)))
                        for name, v in zip(("hypotheses", "distinct_continuations", "query_choices"), zip(*stats))},
            construction_rejections=dict(rejected),
        )

    private_manifest = dict(
        format="libero_history_candidates_v1", seed=seed, attempts=k,
        status="nominal_geometry_only_not_simulator_validated", coordinate_frame="main_table_xy_metres",
        rules=rules, rule_equivalence="same behaviour on every nominal scene in this bank only",
        rule_splits=rule_splits, scene_splits=scene_splits, scenes=scenes,
        anchor_partner=placement.PARTNER.tolist(),
        semantics={"pick_if": "switch to the other bowl when condition is true",
                   "anchor_if": "use the partner anchor selector when condition is true",
                   "relations": "Euclidean distance and min/max axes in table XY; not camera left/right",
                   "ties": "placement.py selector order, including complement for opposite two-bowl selectors; inspect realized resets"},
    )
    (out / "evaluator" / "manifest.json").write_text(json.dumps(private_manifest, indent=2) + "\n")
    summary = dict(status=private_manifest["status"], seed=seed, attempts=k,
                   scenes=len(scenes), syntactic_rules=len(rules), distinct_rule_functions=len(groups),
                   splits=summaries, ambiguity_reference="full rule family, without split membership",
                   native_episodes=0, model_updates=0,
                   source_manifest_sha256=hashlib.sha256((source / "manifest.json").read_bytes()).hexdigest())
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--scenes", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--train", type=int, default=1024)
    ap.add_argument("--validation", type=int, default=256)
    ap.add_argument("--test", type=int, default=256)
    ap.add_argument("--attempts", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    print(json.dumps(compile_cases(args.scenes, args.out, (args.train, args.validation, args.test),
                                   args.attempts, args.seed), indent=2))


if __name__ == "__main__":
    main()
