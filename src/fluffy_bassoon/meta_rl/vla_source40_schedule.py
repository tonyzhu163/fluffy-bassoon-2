"""Deterministic case plans for source-only official LIBERO40 collection.

This module reads static metadata only. It never constructs a simulator or VLA.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess


SOURCE_SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
LIBERO_COMMIT = "87edbd1b823e464bc66557d213cc45e9f88aa543"
FORMAT = "openpi_pi05_libero40_case_schedule_v1"


def content_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def case_plan(task_key, split, count, episodes, seed):
    if (split not in ("train", "validation") or type(count) is not int or count < 1
            or type(episodes) is not int or episodes < 2 or not 0 <= seed < 2**31):
        raise ValueError("Require complete multi-episode cases and a valid partition/seed")
    suite, number = task_key.split(":")
    if suite not in SOURCE_SUITES or not number.isdecimal() or not 0 <= int(number) < 10:
        raise ValueError("Only the four official ten-task source suites are permitted")
    bank = list(range(40)) if split == "train" else list(range(40, 50))
    result = []
    for ordinal in range(count):
        state = bank[ordinal % len(bank)]
        scene_seed = int(content_hash(dict(task=task_key, split=split, ordinal=ordinal, seed=seed))[:16], 16) % (2**31 - 1)
        result.append(dict(case_id=f"{task_key}/{split}/case-{ordinal:05d}", split=split,
            initial_state=state, scene_seed=scene_seed, seed_repetition=ordinal // len(bank),
            episodes=episodes, learner_reset=False, simulator_reset="same fixed state/seed at every episode"))
    return result


def build_schedule(root, *, train_cases, validation_cases, episodes_per_case, target_chunks,
                   seed=0, horizon=400, chunk=10, task_keys=None):
    root = Path(root)
    if (horizon != 400 or chunk != 10 or type(target_chunks) is not int or target_chunks <= 500
            or any(type(v) is not int or v < 1 for v in (train_cases, validation_cases))):
        raise ValueError("Require native H10 chunks, horizon400 and >500 source chunks per partition")
    actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    if actual != LIBERO_COMMIT:
        raise ValueError("LIBERO source commit differs")
    path = root / "libero/libero/benchmark/libero_suite_task_map.py"
    mapping = ast.literal_eval(ast.parse(path.read_text()).body[0].value)
    requested = set(task_keys) if task_keys else {f"{suite}:{i:02d}" for suite in SOURCE_SUITES for i in range(10)}
    valid = {f"{suite}:{i:02d}" for suite in SOURCE_SUITES for i in range(10)}
    if not requested or not requested.issubset(valid) or (task_keys and len(requested) != len(task_keys)):
        raise ValueError("Source schedule rejects LIBERO90/Plus/Pro and duplicate task keys")
    tasks = []
    for suite in SOURCE_SUITES:
        if len(mapping[suite]) != 10:
            raise ValueError("Each source suite must contain exactly ten tasks")
        for i, name in enumerate(mapping[suite]):
            key = f"{suite}:{i:02d}"
            if key not in requested:
                continue
            bddl = root / f"libero/libero/bddl_files/{suite}/{name}.bddl"
            init = root / f"libero/libero/init_files/{suite}/{name}.pruned_init"
            tasks.append(dict(task_key=key, suite=suite, task_id=i, semantic_task=name,
                bddl_sha256=hashlib.sha256(bddl.read_bytes()).hexdigest(),
                initial_states_sha256=hashlib.sha256(init.read_bytes()).hexdigest(),
                partitions={split: case_plan(key, split, count, episodes_per_case, seed)
                            for split, count in (("train", train_cases), ("validation", validation_cases))}))
    # Collection stops only at a complete case boundary after reaching target.
    maximum_chunks = target_chunks + episodes_per_case * (horizon // chunk)
    result = dict(format=FORMAT, libero_commit=actual, task_order_index=0, seed=seed,
        task_map_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), tasks=tasks,
        source_suites=list(SOURCE_SUITES), target_only_suites=["libero_90", "LIBERO-Plus", "LIBERO-Pro"],
        source_only=True, horizon=horizon, executed_chunk=chunk, native_horizon=10,
        episodes_per_case=episodes_per_case, target_chunks_per_task_partition=target_chunks,
        maximum_chunks_per_task_partition=maximum_chunks,
        maximum_primitive_steps_per_task_partition=maximum_chunks * chunk,
        planned_maximum_primitive_steps=len(tasks) * 2 * maximum_chunks * chunk,
        budget_status="Schedule only; creating this file does not authorize full collection",
        learner_rule="Fresh SAC once per task+partition; actor/critic/replay/optimizer persist across cases and episodes; no cross-partition replay",
        source_recipe=dict(warmup_chunks=500, warmup="unbounded tied32D N(0,I), independently sampled per decision",
            batch_size=256, utd=20, update_timing="fixed weights/stage during each episode; after a completed episode with replay>500, new_chunks*20 updates",
            sparse_reward="0 success, -1 otherwise per decision", gamma_per_primitive=.999),
        case_split_rule="Whole cases; train states0..39 and validation states40..49; deterministic independent seeds; no source data from targets",
        deployment="Separate evaluation: one support episode plus one query episode; source generation history lengths are not deployment shots")
    result["schedule_sha256"] = content_hash(result)
    return result


def load_schedule(path):
    value = json.loads(Path(path).read_text())
    if (value.get("format") != FORMAT or value.get("source_only") is not True
            or value.get("schedule_sha256") != content_hash({k: v for k, v in value.items() if k != "schedule_sha256"})):
        raise ValueError("Source schedule format/hash differs")
    ids = set()
    task_ids = set()
    for task in value["tasks"]:
        if task["suite"] not in SOURCE_SUITES or task["task_key"] != f"{task['suite']}:{task['task_id']:02d}" or not 0 <= task["task_id"] < 10:
            raise ValueError("Target task cannot enter source schedule")
        if task["task_key"] in task_ids:
            raise ValueError("Repeated source task identity")
        task_ids.add(task["task_key"])
        for split in ("train", "validation"):
            for case in task["partitions"][split]:
                bank = range(40) if split == "train" else range(40, 50)
                if case["case_id"] in ids or case["split"] != split or case["initial_state"] not in bank or case["learner_reset"] is not False:
                    raise ValueError("Case split/identity/reset rule differs")
                ids.add(case["case_id"])
    return value


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--libero-root", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--train-cases", type=int, required=True)
    ap.add_argument("--validation-cases", type=int, required=True)
    ap.add_argument("--episodes-per-case", type=int, default=4)
    ap.add_argument("--target-chunks", type=int, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--task", action="append", dest="task_keys")
    args = ap.parse_args(argv)
    plan = build_schedule(args.libero_root, train_cases=args.train_cases, validation_cases=args.validation_cases,
        episodes_per_case=args.episodes_per_case, target_chunks=args.target_chunks, seed=args.seed, task_keys=args.task_keys)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(plan, stream, indent=2)
        stream.write("\n")
    print(json.dumps({k: plan[k] for k in ("schedule_sha256", "maximum_primitive_steps_per_task_partition", "planned_maximum_primitive_steps", "budget_status")}, indent=2))


if __name__ == "__main__":
    main()
