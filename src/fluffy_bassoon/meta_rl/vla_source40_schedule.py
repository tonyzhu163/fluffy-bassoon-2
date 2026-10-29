"""Deterministic case plans for source-only official LIBERO40 collection.

This module reads static metadata only. It never constructs a simulator or VLA.
"""
import argparse
import ast
import hashlib
import json
import math
from pathlib import Path
import subprocess


SOURCE_SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
LIBERO_COMMIT = "87edbd1b823e464bc66557d213cc45e9f88aa543"
FORMAT = "openpi_pi05_libero40_case_schedule_v1"
COLLECTION_MODES = ("chunk_target", "complete_cases")


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


def partition_budget(case_count, episodes_per_case, *, collection_mode, target_chunks=None,
                     publish_every_cases=1, horizon=400, chunk=10):
    """Conservative plan allowances; storage is an assumption, not a quota check."""
    if (collection_mode not in COLLECTION_MODES or type(case_count) is not int or case_count < 1
            or type(episodes_per_case) is not int or episodes_per_case < 2
            or type(publish_every_cases) is not int or publish_every_cases < 1
            or horizon != 400 or chunk != 10):
        raise ValueError("Invalid complete-case collection budget")
    if collection_mode == "chunk_target" and (type(target_chunks) is not int or target_chunks <= 500):
        raise ValueError("Chunk-target collection requires >500 queried chunks")
    if collection_mode == "complete_cases" and target_chunks is not None:
        raise ValueError("Complete-case collection cannot have a chunk stopping target")
    episodes = case_count * episodes_per_case
    chunks_per_case = episodes_per_case * math.ceil(horizon / chunk)
    chunks = episodes * math.ceil(horizon / chunk)
    if collection_mode == "chunk_target":
        chunks = min(chunks, target_chunks + chunks_per_case)
    publications = list(range(publish_every_cases, case_count + 1, publish_every_cases))
    if not publications or publications[-1] != case_count:
        publications.append(case_count)
    prefix_rows = sum(min(index * chunks_per_case, chunks) for index in publications)
    # Chunk-target mode can terminate before its regular publication cadence.
    # Reserve one additional immutable terminal export for that possibility.
    terminal_extra = int(collection_mode == "chunk_target" and publish_every_cases > 1)
    prefix_rows += terminal_extra * chunks
    # Two raw representations: complete episode rows and write-ahead decision
    # journals. Checkpoints include all growing replay rows and optimizer state.
    raw = chunks * 2 * 1024**2
    datasets = prefix_rows * 40 * 1024
    checkpoints = (len(publications) + terminal_extra + 2) * 256 * 1024**2 + (prefix_rows + 2 * chunks) * 32 * 1024
    return dict(requested_cases=case_count, requested_episodes=episodes,
        maximum_cases=case_count, maximum_episodes=episodes, maximum_chunks=chunks,
        maximum_primitive_steps=chunks * chunk,
        maximum_settling_steps=(episodes + 1) * 10,
        minimum_chunks_if_each_episode_finishes_in_one_decision=episodes,
        warmup_clearing_guaranteed_by_episode_count=episodes > 500,
        immutable_publication_case_numbers=publications,
        additional_terminal_chunk_target_export_allowance=terminal_extra,
        storage_allowance=dict(raw_trace_and_journal_bytes=raw, paired_prefix_dataset_bytes=datasets,
            source_checkpoint_and_replay_bytes=checkpoints,
            total_bytes_with_ten_percent_reserve=((raw + datasets + checkpoints) * 110 + 99) // 100,
            assumptions="Planning allowances:2MiB/decision for combined raw traces+journals;40KiB/row for paired AD/DPT prefixes;32KiB/replay row and256MiB fixed state per full checkpoint; immutable publications plus2 atomic recovery copies. No compression credit. Input VLA weights, exceptional orphan files and repeated process qualification are excluded; measure before approving collection."))


def selected_partition_budget(schedule, split):
    if split not in ("train", "validation"):
        raise ValueError("Invalid source partition")
    if "partition_budgets" in schedule:
        return schedule["partition_budgets"][split]
    # Existing immutable smoke schedules remain readable under their own pin.
    return dict(maximum_primitive_steps=schedule["maximum_primitive_steps_per_task_partition"])


def build_schedule(root, *, train_cases, validation_cases, episodes_per_case, target_chunks=None,
                   seed=0, horizon=400, chunk=10, task_keys=None,
                   collection_mode="chunk_target", publish_every_cases=1):
    root = Path(root)
    if (horizon != 400 or chunk != 10
            or any(type(v) is not int or v < 1 for v in (train_cases, validation_cases))):
        raise ValueError("Require native H10 chunks, horizon400 and nonempty partitions")
    budgets = {split: partition_budget(count, episodes_per_case, collection_mode=collection_mode,
        target_chunks=target_chunks, publish_every_cases=publish_every_cases, horizon=horizon, chunk=chunk)
        for split, count in (("train", train_cases), ("validation", validation_cases))}
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
    maximum_chunks = max(row["maximum_chunks"] for row in budgets.values())
    result = dict(format=FORMAT, libero_commit=actual, task_order_index=0, seed=seed,
        task_map_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), tasks=tasks,
        source_suites=list(SOURCE_SUITES), target_only_suites=["libero_90", "LIBERO-Plus", "LIBERO-Pro"],
        source_only=True, horizon=horizon, executed_chunk=chunk, native_horizon=10,
        collection_mode=collection_mode, publish_every_cases=publish_every_cases,
        episodes_per_case=episodes_per_case, target_chunks_per_task_partition=target_chunks,
        maximum_chunks_per_task_partition=maximum_chunks,
        maximum_primitive_steps_per_task_partition=maximum_chunks * chunk,
        partition_budgets=budgets,
        planned_requested_cases=len(tasks) * (train_cases + validation_cases),
        planned_requested_episodes=len(tasks) * (train_cases + validation_cases) * episodes_per_case,
        planned_maximum_chunks=len(tasks) * sum(row["maximum_chunks"] for row in budgets.values()),
        planned_maximum_primitive_steps=len(tasks) * sum(row["maximum_primitive_steps"] for row in budgets.values()),
        planned_maximum_settling_steps=len(tasks) * sum(row["maximum_settling_steps"] for row in budgets.values()),
        planned_storage_allowance_bytes=len(tasks) * sum(row["storage_allowance"]["total_bytes_with_ten_percent_reserve"] for row in budgets.values()),
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
            pairs = set()
            for case in task["partitions"][split]:
                bank = range(40) if split == "train" else range(40, 50)
                if case["case_id"] in ids or case["split"] != split or case["initial_state"] not in bank or case["learner_reset"] is not False:
                    raise ValueError("Case split/identity/reset rule differs")
                pair = (case["initial_state"], case["scene_seed"])
                if (pair in pairs or type(case["episodes"]) is not int or case["episodes"] < 2
                        or ("episodes_per_case" in value and case["episodes"] != value["episodes_per_case"])):
                    raise ValueError("Seeded cases must be unique and have the requested complete episode count")
                pairs.add(pair)
                ids.add(case["case_id"])
    if "partition_budgets" in value:
        for split in ("train", "validation"):
            counts = {len(task["partitions"][split]) for task in value["tasks"]}
            if len(counts) != 1:
                raise ValueError("Every requested source task must have the declared full case count")
            expected = partition_budget(counts.pop(), value["episodes_per_case"],
                collection_mode=value.get("collection_mode", "chunk_target"),
                target_chunks=value["target_chunks_per_task_partition"],
                publish_every_cases=value.get("publish_every_cases", 1),
                horizon=value["horizon"], chunk=value["executed_chunk"])
            if expected != value["partition_budgets"][split]:
                raise ValueError("Case/chunk/storage budget differs from the requested schedule")
    return value


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--libero-root", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--train-cases", type=int, required=True)
    ap.add_argument("--validation-cases", type=int, required=True)
    ap.add_argument("--episodes-per-case", type=int, default=4)
    ap.add_argument("--collection-mode", choices=COLLECTION_MODES, default="chunk_target")
    ap.add_argument("--target-chunks", type=int)
    ap.add_argument("--publish-every-cases", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--task", action="append", dest="task_keys")
    args = ap.parse_args(argv)
    plan = build_schedule(args.libero_root, train_cases=args.train_cases, validation_cases=args.validation_cases,
        episodes_per_case=args.episodes_per_case, target_chunks=args.target_chunks, seed=args.seed, task_keys=args.task_keys,
        collection_mode=args.collection_mode, publish_every_cases=args.publish_every_cases)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(plan, stream, indent=2)
        stream.write("\n")
    print(json.dumps({k: plan[k] for k in ("schedule_sha256", "collection_mode", "planned_requested_cases", "planned_requested_episodes", "planned_maximum_chunks", "planned_maximum_primitive_steps", "planned_maximum_settling_steps", "planned_storage_allowance_bytes", "budget_status")}, indent=2))


if __name__ == "__main__":
    main()
