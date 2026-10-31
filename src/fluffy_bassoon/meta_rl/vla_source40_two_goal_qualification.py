"""Qualify two source goals, then use the existing small-pilot assembler.

This adds no training stack or production admission exception. Four immutable
source indices are required: Spatial00 and LIBERO10:00, each with an independent
source-local validation learner and actual postwarmup behavior.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from .vla_history import file_sha256
from .vla_source40_schedule import content_hash
from .vla_source40_training import assemble, load_assembly, load_source


TASKS = ("libero_spatial:00", "libero_10:00")
FORMAT = "openpi_pi05_two_goal_subset_qualification_v1"


def postwarmup_evidence(source, index, contract):
    """Require executed learned-policy rows, not merely a saved final actor."""
    recipe = contract.get("source_recipe", {})
    if any(recipe.get(key) != value for key, value in
            (("warmup_chunks", 500), ("batch_size", 256), ("utd", 20))):
        raise ValueError("Two-goal qualification requires the declared500/256/20 source recipe")
    clocks = np.asarray(source["datasets"]["ad"]["teacher_steps"])
    if (index["chunks"] <= 500 or clocks[0] != 0 or clocks[-1] <= 0
            or clocks[-1] > index["source_updates"]
            or not np.all(np.diff(clocks) >= 0) or not np.any(np.diff(clocks) > 0)):
        raise ValueError("Require actual chronological postwarmup source-policy updates")
    warmup = learned = cursor = 0
    learned_episodes = 0
    for row in index["episode_receipts"]:
        stages = row.get("source_stages", {})
        if (set(stages) - {"warmup_gaussian", "sac_online"}
                or sum(stages.values()) != row["chunks"]
                or sum(value > 0 for value in stages.values()) != 1):
            raise ValueError("Require actual fixed-stage behavior receipts for every episode")
        episode_clock = clocks[cursor:cursor + row["chunks"]]
        if (len(episode_clock) != row["chunks"] or not np.all(episode_clock == episode_clock[0])
                or bool(stages.get("sac_online", 0)) != bool(episode_clock[0] > 0)):
            raise ValueError("Actual episode teacher clocks differ from their fixed behavior-stage receipts")
        cursor += row["chunks"]
        warmup += stages.get("warmup_gaussian", 0)
        learned += stages.get("sac_online", 0)
        learned_episodes += int(stages.get("sac_online", 0) > 0)
    if warmup <= 500 or learned <= 0:
        raise ValueError("Require actual postwarmup SAC behavior, not only final actor updates")
    recovery = Path(source["index_path"]).parent / "source-recovery.pkl"
    if file_sha256(recovery) != index.get("latest_recovery_sha256"):
        raise ValueError("Full source learner/replay/optimizer/RNG recovery checksum differs")
    seed = contract.get("learner_seed")
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("Source learner requires its recorded independent RNG seed")
    return dict(task=source["task"], history_id=source["history_id"], learner_seed=seed,
        source_contract_sha256=index["source_contract_sha256"],
        contract_file_sha256=file_sha256(Path(source["index_path"]).parent / "contract.json"),
        source_updates=index["source_updates"], warmup_behavior_chunks=warmup,
        learned_behavior_chunks=learned, learned_behavior_episodes=learned_episodes,
        teacher_steps_first=int(clocks[0]), teacher_steps_last=int(clocks[-1]),
        full_recovery_path=str(recovery), full_recovery_sha256=file_sha256(recovery),
        actor_checkpoint=source["selected"]["source_checkpoint"],
        actor_checkpoint_sha256=source["selected"]["source_checkpoint_sha256"],
        source_generation_cost_count_once=source["costs"],
        actual_initial_state_ids=sorted(state for _, state in source["states"]),
        status="Executed SAC source candidate; no expert, improvement or optimality claim")


def qualify_two_goal(train_indices, validation_indices, output, *, train_sha256,
        validation_sha256, qualification_subset=False):
    if not qualification_subset:
        raise ValueError("Require explicit --qualification-subset; production40 admission is unchanged")
    groups, evidence = {}, {}
    for partition, paths, hashes in (("train", train_indices, train_sha256),
                                     ("validation", validation_indices, validation_sha256)):
        if len(paths) != 2 or len(hashes) != 2:
            raise ValueError("Require two hash-pinned indices in each source partition")
        sources, proofs = [], []
        for path, expected in zip(paths, hashes):
            path = Path(path).resolve()
            if len(expected) != 64 or file_sha256(path) != expected:
                raise ValueError("Explicit source index file SHA256 differs")
            source = load_source(path, partition, allow_small_pilot=True)
            index = json.loads(path.read_text())
            contract = json.loads((path.parent / "contract.json").read_text())
            if "Independent SAC/replay" not in contract.get("validation", ""):
                raise ValueError("Require the independent train/validation learner/replay contract")
            sources.append(source)
            proofs.append(postwarmup_evidence(source, index, contract))
        if {row["task"] for row in sources} != set(TASKS):
            raise ValueError("Two-goal subset requires exactly Spatial00 and LIBERO10:00 in both partitions")
        groups[partition], evidence[partition] = sources, proofs
    all_sources = groups["train"] + groups["validation"]
    if len({content_hash(source["feature_contract"]) for source in all_sources}) != 1:
        raise ValueError("All source goals must retain one exact frozen motor/feature/noise contract")
    if len({row["learner_seed"] for rows in evidence.values() for row in rows}) != 4:
        raise ValueError("Require four independent task/partition learner seeds")
    # Existing assembler checks complete chronological cases, state/hash and
    # history separation, exact motor/noise provenance and all source costs.
    value = assemble(train_indices, validation_indices, output, allow_small_pilot=True)
    output = Path(output).resolve()
    load_assembly(output / "assembly.json", allow_small_pilot=True)
    receipt = dict(format=FORMAT, qualification_subset=True, source_only=True,
        task_keys=list(TASKS), evidence=evidence,
        assembly_path=str(output / "assembly.json"), assembly_file_sha256=file_sha256(output / "assembly.json"),
        assembly_sha256=value["assembly_sha256"], collection_costs_count_once=value["collection_costs_count_once"],
        additional_simulator_actions=0,
        training="Use the existing vla_source40_subset_training with all four original indices and this hash-pinned assembly",
        interpretation="Two source goals, source-local disjoint validation; no full40 coverage, target data or robot-performance claim")
    receipt["content_sha256"] = content_hash(receipt)
    (output / "two-goal-qualification.json").write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    return receipt


def parser():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--qualification-subset", action="store_true", required=True)
    for partition in ("train", "validation"):
        ap.add_argument(f"--{partition}-index", type=Path, nargs=2, required=True)
        ap.add_argument(f"--{partition}-sha256", nargs=2, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    return ap


def main(argv=None):
    args = parser().parse_args(argv)
    value = qualify_two_goal(args.train_index, args.validation_index, args.output_dir,
        train_sha256=args.train_sha256, validation_sha256=args.validation_sha256,
        qualification_subset=args.qualification_subset)
    print(json.dumps(value, allow_nan=False))


if __name__ == "__main__":
    main()
