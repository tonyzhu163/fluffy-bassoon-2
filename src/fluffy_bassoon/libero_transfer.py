"""Static, content-addressed LIBERO-40 -> LIBERO-90 transfer manifest.

No model, simulator, NumPy, or executable pickle imports. This is a separate
experiment from libero_split.py's existing 78/10/10 split.
"""
import argparse
import ast
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import subprocess


LIBERO_COMMIT = "87edbd1b823e464bc66557d213cc45e9f88aa543"
TASK_MAP_SHA256 = "0c950df0a785aa55de968bb38ccd865d2017f71ddbe6f48cfd05ac0742b6d62d"
TRAIN_SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
TEST_SUITE = "libero_90"


def content_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sexp(text):
    tokens = iter(re.findall(r"\(|\)|[^\s()]+", re.sub(r";[^\n]*", "", text)))

    def read(token):
        if token != "(":
            return token.lower()
        result = []
        for token in tokens:
            if token == ")":
                return result
            result.append(read(token))
        raise ValueError("Unclosed BDDL expression")

    result = read(next(tokens))
    if next(tokens, None) is not None:
        raise ValueError("Extra BDDL expression")
    return result


def normalize_goal(value):
    if isinstance(value, str):
        return value
    values = [normalize_goal(v) for v in value]
    if values and values[0] in ("and", "or"):
        return [values[0], *sorted(values[1:], key=lambda v: json.dumps(v))]
    return values


def goal_bindings(forms, goal):
    declared = {}
    for name in (":objects", ":fixtures"):
        pending = []
        values = forms[name][1:]
        for index, token in enumerate(values):
            if token == "-":
                declared.update({obj: values[index + 1] for obj in pending})
                pending = []
            elif index == 0 or values[index - 1] != "-":
                pending.append(token)
        if pending:
            raise ValueError("Untyped BDDL object")
    tokens = set(re.findall(r"[a-z0-9_]+", json.dumps(goal)))
    return {obj: kind for obj, kind in sorted(declared.items())
            if any(t == obj or t.startswith(obj + "_") for t in tokens)}


def partition(records):
    """Keep all 40 source tasks; flag equivalent 90 tasks as overlap controls.

    Equivalence is transitive under equal full goal+object bindings OR equal
    full normalized instruction. Familiar atomic subgoals are allowed.
    """
    records = [dict(row) for row in records]
    parent = list(range(len(records)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for field in ("goal_binding_sha256", "normalized_instruction"):
        seen = {}
        for i, row in enumerate(records):
            if row[field] in seen:
                parent[find(i)] = find(seen[row[field]])
            else:
                seen[row[field]] = i
    groups = defaultdict(list)
    for i in range(len(records)):
        groups[find(i)].append(i)
    for indices in groups.values():
        members = sorted(records[i]["task_key"] for i in indices)
        family = "full-goal-family-" + content_hash(members)[:16]
        seen_in_train = any(records[i]["suite"] in TRAIN_SUITES for i in indices)
        for i in indices:
            row = records[i]
            row["family"] = family
            row["split"] = ("train40" if row["suite"] in TRAIN_SUITES else
                            "test90_overlap_control" if seen_in_train else "test90_strict")
            row["meta_gradient_allowed"] = row["split"] == "train40"
            row["source_teacher_allowed"] = row["split"] == "train40"
            row["selection_allowed"] = row["split"] == "train40"
    return records


def validate(manifest):
    claimed = manifest.get("metadata_sha256")
    if claimed != content_hash({k: v for k, v in manifest.items() if k != "metadata_sha256"}):
        raise ValueError("Manifest metadata_sha256 differs")
    rows = manifest["tasks"]
    if len({r["task_key"] for r in rows}) != 130:
        raise ValueError("Require 130 distinct canonical suite/task IDs")
    expected = {**{suite: 10 for suite in TRAIN_SUITES}, TEST_SUITE: 90}
    if dict(Counter(r["suite"] for r in rows)) != expected:
        raise ValueError("Unexpected suite counts")
    computed = partition(rows)
    for row, reference in zip(rows, computed):
        for field in ("family", "split", "meta_gradient_allowed", "source_teacher_allowed", "selection_allowed"):
            if row[field] != reference[field]:
                raise ValueError(f"Invalid {field} for {row['task_key']}")
    if manifest["counts"] != dict(Counter(r["split"] for r in rows)):
        raise ValueError("Manifest counts differ")
    return manifest


def load(path):
    return validate(json.loads(Path(path).read_text()))


def build(source):
    source = Path(source).resolve()
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip()
    dirty = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"],
                                    cwd=source, text=True).strip()
    if revision != LIBERO_COMMIT or dirty:
        raise ValueError("Require clean pinned official LIBERO source")
    task_map_path = source / "libero/libero/benchmark/libero_suite_task_map.py"
    if file_hash(task_map_path) != TASK_MAP_SHA256:
        raise ValueError("Pinned task-map hash differs")
    task_map = ast.literal_eval(ast.parse(task_map_path.read_text()).body[0].value)
    rows, inventory = [], []
    for suite in (*TRAIN_SUITES, TEST_SUITE):
        for task_id, name in enumerate(task_map[suite]):
            bddl = source / "libero/libero/bddl_files" / suite / f"{name}.bddl"
            init = source / "libero/libero/init_files" / suite / f"{name}.pruned_init"
            forms = {v[0]: v for v in sexp(bddl.read_text()) if isinstance(v, list) and v}
            goal = normalize_goal(forms[":goal"][1])
            instruction = " ".join(forms[":language"][1:])
            objects = goal_bindings(forms, goal)
            for path in (bddl, init):
                inventory.append(dict(path=str(path.relative_to(source)), sha256=file_hash(path)))
            rows.append(dict(task_key=f"{suite}:{task_id:02d}", suite=suite, task_id=task_id,
                semantic_task=name, instruction=instruction,
                normalized_instruction=" ".join(re.findall(r"[a-z0-9]+", instruction.lower())),
                normalized_full_goal=goal, referenced_object_bindings=objects,
                goal_binding_sha256=content_hash(dict(goal=goal, object_bindings=objects)),
                bddl_path=str(bddl.relative_to(source)), bddl_sha256=file_hash(bddl),
                initial_states_path=str(init.relative_to(source)), initial_states_sha256=file_hash(init)))
    rows = partition(rows)
    counts = dict(Counter(r["split"] for r in rows))
    result = dict(format="libero40_to_90_v1", benchmark_commit=revision,
        task_map_sha256=TASK_MAP_SHA256, source_inventory_sha256=content_hash(sorted(inventory, key=lambda r: r["path"])),
        counts=counts, tasks=rows,
        family_rule="Transitive equality of normalized full goal+referenced bindings OR normalized full instruction; syntactic audit, not exhaustive semantic equivalence.",
        initial_states=dict(training=list(range(40)), validation=list(range(40, 45)),
                            familiar_task_test=list(range(45, 50)), test90=list(range(50))),
        restrictions=dict(test90_source_teacher=False, test90_meta_gradients=False,
            test90_checkpoint_selection=False, test90_normalization_fitting=False,
            final_plus_pro_tuning=False, online_dsrl="Independent evaluation clone per trial; charge all interactions; no return to training."),
        reporting=dict(primary="test90_strict", secondary="all 90, with overlap controls separately reported",
            plus_pro="Variants of train40 bases test robustness. Joint task+shift transfer requires independently held-out base families.",
            motor_exposure="Released LIBERO-130 motor has seen test90 tasks; unseen applies to steering-controller training."))
    result["metadata_sha256"] = content_hash(result)
    return validate(result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = build(args.source_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        stream.write(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({k: manifest[k] for k in ("counts", "metadata_sha256", "source_inventory_sha256")}))


if __name__ == "__main__":
    main()
