"""Explicit whole-case data scales from one qualified source learning history.

This is a diagnostic helper, not an alternate all40 production admission path.
It keeps the original source/teacher costs at every scale. A warmup-only prefix
is behavior cloning rather than algorithm distillation and is marked as such.
"""
import argparse
import copy
import json
from pathlib import Path

import numpy as np

from .vla_history import FIELDS, file_sha256, save_dataset
from .vla_source40_schedule import content_hash
from .vla_source40_training import (ASSEMBLY_FORMAT, NOISE_CONTRACT, OBJECTIVES,
    load_assembly, load_source)


FORMAT = "openpi_pi05_source40_whole_case_scales_v1"


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def prefix_selection(index, cases):
    """Select the first N entire chronological cases without outcome filtering."""
    receipts = index["episode_receipts"]
    chronology = list(dict.fromkeys(row["case_id"] for row in receipts))
    if type(cases) is not int or not 1 <= cases <= len(chronology):
        raise ValueError("Data scale must select a positive number of available complete cases")
    selected = chronology[:cases]
    rows = [row for row in receipts if row["case_id"] in set(selected)]
    if rows != receipts[:len(rows)]:
        raise ValueError("Case selection is not a chronological episode prefix")
    for case in selected:
        episodes = [row["episode_in_case"] for row in rows if row["case_id"] == case]
        if episodes != list(range(len(episodes))):
            raise ValueError("Selected case does not contain all its chronological episodes")
    return dict(case_ids=selected, cases=cases, episodes=len(rows),
        transitions=sum(row["chunks"] for row in rows),
        nominal_state_ids=sorted({row["initial_state"] for row in rows}),
        nominal_state_hashes=sorted({row["initial_state_vector_sha256"] for row in rows}))


def slice_trial(trial, selection):
    end = selection["transitions"]
    if not 0 < end <= len(trial["observations"]) or not trial["dones"][end - 1]:
        raise ValueError("A whole-case prefix must end on an actual episode boundary")
    result = {field: np.asarray(trial[field])[:end].copy() for field in FIELDS}
    if int(result["dones"].sum()) != selection["episodes"]:
        raise ValueError("Case receipts and selected dataset episode boundaries differ")
    return result


def selected_source(source, *, selection=None):
    # selected_publication identifies the original full actor-label snapshot.
    # used_prefix is separate, so neither DPT teacher interactions nor the
    # original history are silently reduced by a training-array slice.
    result = dict(index_path=source["index_path"], index_sha256=source["index_sha256"],
        task=source["task"], history_id=source["history_id"],
        selected_publication=source["selected"], episodes=source["episodes"],
        case_ids=sorted(source["case_ids"]),
        task_initial_state_ids=[list(value) for value in sorted(source["states"])],
        task_nominal_state_hashes=[list(value) for value in sorted(source["state_hashes"])])
    if selection is not None:
        result["used_prefix"] = selection
    return result


def make_scales(train_index, validation_index, output_dir, *, case_counts=(8, 16, 20),
                qualification_subset=False):
    if not qualification_subset:
        raise ValueError("Whole-case scales require explicit --qualification-subset")
    counts = list(case_counts)
    if (not counts or any(type(count) is not int or count < 1 for count in counts)
            or counts != sorted(set(counts))):
        raise ValueError("Require increasing unique positive whole-case scales")
    train = load_source(train_index, "train", allow_small_pilot=True)
    validation = load_source(validation_index, "validation", allow_small_pilot=True)
    if train["task"] != validation["task"] or train["feature_contract"] != validation["feature_contract"]:
        raise ValueError("This diagnostic requires one matched source goal and frozen feature contract")
    for name in ("case_ids", "states", "state_hashes"):
        if train[name] & validation[name]:
            raise ValueError("Training prefix and unchanged validation share source case/state identities")
    index = json.loads(Path(train_index).read_text())
    selections = [prefix_selection(index, count) for count in counts]
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    costs = {partition: copy.deepcopy(source["costs"])
        for partition, source in (("train", train), ("validation", validation))}
    result = dict(format=FORMAT, qualification_subset=True, source_only=True,
        source_task_keys=[train["task"]], validation_task_keys=[validation["task"]],
        original_train_index=dict(path=train["index_path"], sha256=train["index_sha256"]),
        original_validation_index=dict(path=validation["index_path"], sha256=validation["index_sha256"]),
        original_collection_costs_count_once=costs,
        additional_simulator_actions=0,
        label_provenance="DPT labels retain the final full-source actor; every scale retains its complete teacher/source cost",
        interpretation="Training-array amount diagnostic, not a DPT source-interaction-efficiency curve",
        scales=[])
    for selection in selections:
        directory = output_dir / f"cases-{selection['cases']:03d}"
        directory.mkdir()
        selected = dict(train=[selected_source(train, selection=selection)],
            validation=[selected_source(validation)])
        sliced = {objective: slice_trial(train["datasets"][objective], selection) for objective in OBJECTIVES}
        clock = sliced["ad"]["teacher_steps"]
        has_learning_clock = bool(clock[-1] > clock[0])
        assembly = dict(format=ASSEMBLY_FORMAT, source_only=True,
            motor_sha256=train["feature_contract"]["checkpoint_receipt_sha256"],
            feature_contract=train["feature_contract"], noise_contract=NOISE_CONTRACT,
            small_pilot=True, source_task_keys=[train["task"]], validation_task_keys=[validation["task"]],
            campaign=None, datasets={}, selected_sources=selected,
            collection_costs_count_once=copy.deepcopy(costs),
            whole_case_data_scale=dict(selection, original_train_transitions=index["chunks"],
                teacher_steps_first=int(clock[0]), teacher_steps_last=int(clock[-1]),
                fixed_final_teacher_checkpoint_sha256=train["selected"]["source_checkpoint_sha256"]),
            interpretation="Explicit single-goal subset; full source/teacher collection costs retained; no robot or meta-learning benefit established")
        paths = {}
        for partition, source in (("train", train), ("validation", validation)):
            assembly["datasets"][partition] = {}
            for objective in OBJECTIVES:
                actual_objective = ("behavior_cloning" if partition == "train" and objective == "ad"
                    and not has_learning_clock else objective)
                path = directory / f"{partition}-{actual_objective}.npz"
                trial = sliced[objective] if partition == "train" else source["datasets"][objective]
                metadata = dict(motor_sha256=assembly["motor_sha256"], feature_contract=assembly["feature_contract"],
                    noise_contract=NOISE_CONTRACT, global_task_split="source40", local_scene_split=partition,
                    trial_ids=[source["history_id"]], source_case_ids=(selection["case_ids"] if partition == "train"
                        else sorted(source["case_ids"])), source_task_keys=[source["task"]], campaign=None,
                    selected_source_histories=selected[partition], small_pilot=True,
                    teacher_status="Final full-source SAC candidate retained; no expert/optimality claim",
                    original_collection_costs_count_once=copy.deepcopy(costs),
                    data_amount_scope="Selected training arrays only; no teacher/source interaction saving claimed",
                    publication_rule="Chronological whole-case diagnostic prefix; validation arrays unchanged")
                manifest = save_dataset(path, [trial], objective=actual_objective, metadata=metadata)
                item = dict(path=path.name, sha256=manifest["sha256"],
                    trials=manifest["trials"], transitions=manifest["transitions"])
                assembly["datasets"][partition][objective] = item
                paths[f"{partition}_{actual_objective}"] = dict(path=str(path), sha256=item["sha256"])
        scale = dict(selection, ad_status="learning_history" if has_learning_clock else "raw_only_fixed_teacher",
            teacher_steps_first=int(clock[0]), teacher_steps_last=int(clock[-1]),
            dpt_status="Final full-source actor labels; full source costs retained",
            datasets=paths, validation_arrays_unchanged=True,
            original_collection_costs_count_once=copy.deepcopy(costs))
        if has_learning_clock:
            assembly["assembly_sha256"] = content_hash(assembly)
            assembly_path = directory / "assembly.json"
            write_json(assembly_path, assembly)
            # Existing production loader must admit only its explicit small
            # pilot branch. Its all40 default is neither changed nor bypassed.
            load_assembly(assembly_path, allow_small_pilot=True)
            scale["assembly"] = dict(path=str(assembly_path), sha256=file_sha256(assembly_path),
                assembly_sha256=assembly["assembly_sha256"])
        else:
            scale["assembly"] = None
            scale["reason"] = "All selected teacher clocks are fixed; AD cannot be trained honestly on this prefix. BC and DPT remain available."
        result["scales"].append(scale)
    result["content_sha256"] = content_hash(result)
    write_json(output_dir / "index.json", result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-index", type=Path, required=True)
    parser.add_argument("--validation-index", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--case-counts", type=int, nargs="+", default=(8, 16, 20))
    parser.add_argument("--qualification-subset", action="store_true")
    args = parser.parse_args(argv)
    value = make_scales(args.train_index, args.validation_index, args.output_dir,
        case_counts=args.case_counts, qualification_subset=args.qualification_subset)
    print(json.dumps(value, allow_nan=False))


if __name__ == "__main__":
    main()
