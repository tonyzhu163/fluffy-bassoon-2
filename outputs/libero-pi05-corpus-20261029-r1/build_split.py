"""Build a frozen split from static pinned LIBERO files; no simulator/model imports."""
import argparse
import ast
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import pickletools
import random
import re
import zipfile


LIBERO_COMMIT = "87edbd1b823e464bc66557d213cc45e9f88aa543"
SOURCE_INVENTORY_SHA256 = "d76dd91d819b9be67b721776335f6b6bc94863562985b700f4fd59bfec8066a4"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def sexp(text):
    tokens = iter(re.findall(r"\(|\)|[^\s()]+", re.sub(r";[^\n]*", "", text)))
    def read(token):
        if token != "(":
            return token.lower()
        values = []
        for token in tokens:
            if token == ")":
                return values
            values.append(read(token))
        raise ValueError("Unclosed BDDL expression")
    result = read(next(tokens))
    if next(tokens, None) is not None:
        raise ValueError("Extra BDDL expression")
    return result


def normalize_goal(value):
    if isinstance(value, str):
        return value
    if value and value[0] in ("and", "or"):
        return [value[0], *sorted((normalize_goal(v) for v in value[1:]), key=lambda v: json.dumps(v))]
    return [normalize_goal(v) for v in value]


def atoms(value):
    if isinstance(value, str):
        return {value}
    return set().union(*(atoms(v) for v in value))


def bindings(forms, goal):
    declared = {}
    for name in (":objects", ":fixtures"):
        values = forms[name][1:]
        pending = []
        for index, token in enumerate(values):
            if token == "-":
                for obj in pending:
                    declared[obj] = values[index + 1]
                pending = []
            elif index == 0 or values[index - 1] != "-":
                pending.append(token)
        if pending:
            raise ValueError("Untyped BDDL object")
    referenced = atoms(goal)
    return {obj: kind for obj, kind in sorted(declared.items())
            if any(token == obj or token.startswith(obj + "_") for token in referenced)}


def static_numpy_shape(path):
    """Read ndarray shape opcodes without unpickling or importing Torch/NumPy."""
    with zipfile.ZipFile(path) as archive:
        data = archive.read(next(name for name in archive.namelist() if name.endswith("data.pkl")))
    ops = list(pickletools.genops(data))
    if not any(op.name == "GLOBAL" and arg == "numpy ndarray" for op, arg, _ in ops[:12]):
        raise ValueError("Expected pinned NumPy ndarray serialization")
    for index in range(len(ops) - 4):
        segment = ops[index:index + 5]
        if ([op.name for op, _, _ in segment] == ["MARK", "BININT1", "BININT1", "BININT1", "TUPLE2"]
                and segment[1][1] == 1):
            return [segment[2][1], segment[3][1]]
    raise ValueError("Unrecognized pinned initial-state shape")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path(__file__).parent)
    args = parser.parse_args()
    source, output = args.source_root.resolve(), args.output.resolve()
    task_map_file = source / "libero/libero/benchmark/libero_suite_task_map.py"
    task_map = ast.literal_eval(ast.parse(task_map_file.read_text()).body[0].value)
    files = ["README.md", str(task_map_file.relative_to(source)), "libero/libero/benchmark/__init__.py"]
    for suite in ("libero_90", "libero_10"):
        for name in task_map[suite]:
            files.extend([f"libero/libero/bddl_files/{suite}/{name}.bddl",
                          f"libero/libero/init_files/{suite}/{name}.pruned_init"])
    inventory = [dict(path=path, sha256=digest(source / path)) for path in sorted(files)]
    if canonical_hash(inventory) != SOURCE_INVENTORY_SHA256:
        raise ValueError("Static source bytes differ from the verified pinned LIBERO inventory")
    records = []
    for suite in ("libero_90", "libero_10"):
        for task_id, name in enumerate(task_map[suite]):
            bddl = source / "libero/libero/bddl_files" / suite / (name + ".bddl")
            init = source / "libero/libero/init_files" / suite / (name + ".pruned_init")
            forms = {v[0]: v for v in sexp(bddl.read_text()) if isinstance(v, list) and v}
            goal = normalize_goal(forms[":goal"][1])
            instruction = " ".join(forms[":language"][1:])
            normalized_instruction = " ".join(re.findall(r"[a-z0-9]+", instruction.lower()))
            goal_bindings = bindings(forms, goal)
            shape = static_numpy_shape(init)
            if shape[0] != 50:
                raise ValueError(f"Cannot apply 40/5/5 state split to {suite}:{task_id}: {shape}")
            key = f"{suite}:{task_id:02d}"
            records.append(dict(task_key=key, suite=suite, task_id=task_id, task_order_index=0,
                semantic_task=name, instruction=instruction, normalized_instruction=normalized_instruction,
                scene_family=re.match(r"(.+_SCENE\d+)_", name).group(1),
                bddl_path=str(bddl.relative_to(source)), bddl_sha256=digest(bddl),
                normalized_full_goal=goal, referenced_object_bindings=goal_bindings,
                goal_binding_sha256=canonical_hash(dict(goal=goal, object_bindings=goal_bindings)),
                initial_states=dict(path=str(init.relative_to(source)), sha256=digest(init),
                    shape=shape, count=shape[0], inspection="static NumPy ndarray shape pickle opcodes; no unpickling")))
    # Union transitive full-goal/instruction equivalences before assigning splits.
    parent = list(range(len(records)))
    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index
    def union(a, b):
        parent[find(b)] = find(a)
    for field in ("goal_binding_sha256", "normalized_instruction"):
        seen = {}
        for index, row in enumerate(records):
            if row[field] in seen:
                union(index, seen[row[field]])
            else:
                seen[row[field]] = index
    grouped = defaultdict(list)
    for index in range(len(records)):
        grouped[find(index)].append(index)
    families = []
    for indices in grouped.values():
        members = sorted(records[i]["task_key"] for i in indices)
        family = "full-goal-family-" + canonical_hash(members)[:16]
        for index in indices:
            records[index]["variant_base_family"] = family
        families.append(dict(family=family, members=members,
            touches_official_test=any(records[i]["suite"] == "libero_10" for i in indices)))
    families.sort(key=lambda row: row["family"])
    pool_families = [row for row in families if not row["touches_official_test"]]
    random.Random(0).shuffle(pool_families)
    # Choose exactly ten validation tasks using whole equivalence families.
    choices = {0: []}
    for family in pool_families:
        size = len(family["members"])
        for total, selected in list(choices.items()):
            if total + size <= 10 and total + size not in choices:
                choices[total + size] = selected + [family["family"]]
    if 10 not in choices:
        raise ValueError("Whole-family partition cannot supply ten validation tasks")
    validation = set(choices[10])
    test_families = {row["family"] for row in families if row["touches_official_test"]}
    for row in records:
        family = row["variant_base_family"]
        row["split"] = ("test" if row["suite"] == "libero_10" else
            "test_auxiliary_excluded_90" if family in test_families else
            "validation" if family in validation else "train")
        row["teacher_collection_allowed"] = row["split"] in ("train", "validation")
        row["meta_gradient_allowed"] = row["split"] == "train"
        row["checkpoint_selection_allowed"] = row["split"] == "validation"
        row["source_state_indices"] = (list(range(40)) if row["teacher_collection_allowed"] else [])
        row["auxiliary_familiar_task_state_indices"] = (dict(validation=list(range(40, 45)), test=list(range(45, 50)))
            if row["split"] == "train" else None)
        row["heldout_task_eval_state_indices"] = (list(range(50))
            if row["split"] in ("test", "test_auxiliary_excluded_90") else [])
        row["state_use"] = (dict(source_teacher_training_and_history_collection=list(range(40)),
            familiar_task_validation=list(range(40, 45)), familiar_task_test=list(range(45, 50)))
            if row["split"] == "train" else
            dict(validation_only_teacher_generation=list(range(40)), policy_selection_rollouts=list(range(40, 45)),
                reserved_no_selection=list(range(45, 50))) if row["split"] == "validation" else
            dict(heldout_task_evaluation=list(range(50))))
    counts = dict(Counter(row["split"] for row in records))
    trainval = [row for row in records if row["split"] in ("train", "validation")]
    test = [row for row in records if row["split"] == "test"]
    overlaps = {}
    for field in ("semantic_task", "normalized_instruction", "goal_binding_sha256", "variant_base_family", "scene_family"):
        overlaps[field] = sorted({row[field] for row in trainval} & {row[field] for row in test})
    assert not any(overlaps[field] for field in overlaps if field != "scene_family")
    audit = dict(exact_bddl_name_overlap_90_vs_10=sorted(set(task_map["libero_90"]) & set(task_map["libero_10"])),
        excluded_90_task_keys=[row["task_key"] for row in records if row["split"] == "test_auxiliary_excluded_90"],
        surviving_train_validation_vs_official_test_overlap=overlaps,
        shared_scene_is_permitted="Task/goal-composition transfer is tested; unseen physical scenes or primitive goals are not claimed.",
        initial_state_counts=dict(Counter(str(row["initial_states"]["count"]) for row in records)))
    metadata = dict(format="libero_pi05_corpus_split_v1", seed=0, builder_sha256=digest(Path(__file__)),
        benchmark=dict(repository="https://github.com/Lifelong-Robot-Learning/LIBERO", commit=LIBERO_COMMIT,
            task_map_path=str(task_map_file.relative_to(source)), task_map_sha256=digest(task_map_file),
            benchmark_api_sha256=digest(source / "libero/libero/benchmark/__init__.py"),
            readme_sha256=digest(source / "README.md"), task_order_index=0),
        static_source_inventory=dict(files=len(inventory), sha256=SOURCE_INVENTORY_SHA256),
        scientific_claim="Frozen released pi0.5 attachment adaptation on tasks/goals held out from source-teacher and meta-training; VLA pretraining novelty and unseen scene/primitive transfer are not claimed.",
        counts=counts, family_rule="Connected components under equal normalized full BDDL goal with referenced object bindings OR identical normalized full public instruction, across scenes; all connected siblings share partition.",
        validation_selection="Seed-0 shuffle of sorted family IDs, first reachable whole-family subset of exactly ten tasks via deterministic dynamic programming; all other eligible LIBERO90 tasks train.",
        initial_state_rule=dict(train=list(range(40)), validation=list(range(40, 45)), auxiliary_seen_task_test=list(range(45, 50)),
            heldout_task_test=list(range(50)), count_verified_all_tasks=50),
        restrictions=dict(test_source_teacher_collection=False, test_source_dsrl_updates=False,
            test_meta_gradients=False, test_feature_normalization_fitting=False,
            test_checkpoint_selection=False, test_history_relabeling=False,
            supervised_validation_teacher_collection=True, supervised_validation_gradients=False,
            deployment_online_dsrl_comparator_allowed=True,
            deployment_online_dsrl_comparator="Separate evaluation-only clone with its own K-attempt feedback/updates, charged interactions, and no exports to teacher/meta-training/selection data.",
            split_unit="whole task family, then whole multi-episode trial; sampled windows inherit parent-trial split",
            source_teacher_train_tasks_cannot_use_validation_tasks=True,
            shared_motor_feature_space="one released pi0.5 checkpoint/receipt and one fixed common feature contract; no per-task changing feature coordinates"),
        future_ood=dict(suites=["LIBERO-Plus", "LIBERO-Pro"], status="locked future final tests; task/variant identities not selected using results",
            lineage="Every variant must record originating canonical LIBERO task and variant kind; all descendants inherit their base family partition. Variants of meta-training tasks assess perturbation robustness, not unseen base-task transfer.",
            tuning="No final Plus/Pro feedback may select teachers, curriculum, features, checkpoints or hyperparameters; any development variants must be separately declared before testing."),
        audit=audit, families=families, tasks=records)
    metadata["metadata_sha256"] = canonical_hash(metadata)
    output.mkdir(parents=True, exist_ok=True)
    (output / "split.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(dict(counts=counts, audit=audit, metadata_sha256=metadata["metadata_sha256"]), indent=2))


if __name__ == "__main__":
    main()
