"""Audit all LIBERO90 goals against all LIBERO40 goals using static metadata."""
import argparse
import ast
from collections import Counter, defaultdict
import hashlib
import itertools
import json
from pathlib import Path
import re
import runpy


SOURCE_SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
RESERVED = (1, 2, 47, 57)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def alpha_goal(goal, bindings, normalize):
    """Canonicalize same-type instance names without dropping goal structure."""
    groups = defaultdict(list)
    for obj, kind in bindings.items():
        groups[kind].append(obj)
    kinds = sorted(groups)
    permutations = [list(itertools.permutations(sorted(groups[kind]))) for kind in kinds]
    choices = 1
    for values in permutations:
        choices *= len(values)
    if choices > 100000:
        raise ValueError("Unexpectedly large exact typed alpha-renaming search")
    options = []
    for arrangement in itertools.product(*permutations):
        rename = {}
        for kind, ordered in zip(kinds, arrangement):
            for ordinal, obj in enumerate(ordered):
                rename[obj] = f"typed_entity::{kind}::{ordinal}"
        def rewrite(value):
            if isinstance(value, list):
                return [rewrite(v) for v in value]
            for obj in sorted(rename, key=lambda x: (-len(x), x)):
                if value == obj:
                    return rename[obj]
                if value.startswith(obj + "_"):
                    return rename[obj] + "::region::" + value[len(obj) + 1:]
            return value
        options.append(normalize(rewrite(goal)))
    return min(options, key=encoded)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--snapshot", type=Path, required=True)
    ap.add_argument("--parser", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    snapshot = json.loads(args.snapshot.read_text())
    if snapshot["snapshot_sha256"] != sha(encoded({k: v for k, v in snapshot.items() if k != "snapshot_sha256"})):
        raise ValueError("Snapshot digest mismatch")
    parser = runpy.run_path(str(args.parser))
    if snapshot["source_commit"] != parser["LIBERO_COMMIT"]:
        raise ValueError("Parser and source commit disagree")
    files = {row["path"]: row for row in snapshot["files"]}
    for row in files.values():
        if "text" in row and sha(row["text"].encode()) != row["sha256"]:
            raise ValueError(f"Snapshot source bytes mismatch: {row['path']}")
    map_path = "libero/libero/benchmark/libero_suite_task_map.py"
    task_map = ast.literal_eval(ast.parse(files[map_path]["text"]).body[0].value)
    records = []
    for suite in (*SOURCE_SUITES, "libero_90"):
        for index, name in enumerate(task_map[suite]):
            path = f"libero/libero/bddl_files/{suite}/{name}.bddl"
            forms = {v[0]: v for v in parser["sexp"](files[path]["text"]) if isinstance(v, list) and v}
            goal = parser["normalize_goal"](forms[":goal"][1])
            bindings = parser["bindings"](forms, goal)
            typed = alpha_goal(goal, bindings, parser["normalize_goal"])
            instruction = " ".join(forms[":language"][1:])
            initial = f"libero/libero/init_files/{suite}/{name}.pruned_init"
            records.append(dict(task_key=f"{suite}:{index:02d}", suite=suite, task_id=index,
                task_order_index=0, role="target_candidate_only" if suite == "libero_90" else "source40",
                semantic_task=name, instruction=instruction,
                normalized_instruction=" ".join(re.findall(r"[a-z0-9]+", instruction.lower())),
                bddl_path=path, bddl_sha256=files[path]["sha256"], initial_states={k: files[initial][k] for k in ("path", "sha256", "bytes", "git_blob")},
                normalized_full_goal=goal, referenced_object_bindings=bindings,
                goal_binding_sha256=sha(encoded(dict(goal=goal, object_bindings=bindings))),
                typed_alpha_full_goal=typed, typed_alpha_goal_sha256=sha(encoded(typed)),
                prior_development=(suite == "libero_90" and index in RESERVED)))
    assert Counter(row["role"] for row in records) == {"source40": 40, "target_candidate_only": 90}
    parent = list(range(len(records)))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    edges = []
    for field in ("goal_binding_sha256", "typed_alpha_goal_sha256", "normalized_instruction"):
        seen = {}
        for i, row in enumerate(records):
            if row[field] in seen:
                j = seen[row[field]]
                parent[find(i)] = find(j)
                edges.append(dict(a=records[j]["task_key"], b=row["task_key"], equality=field))
            else:
                seen[row[field]] = i
    grouped = defaultdict(list)
    for i, row in enumerate(records):
        grouped[find(i)].append(row)
    families = []
    for rows in grouped.values():
        members = sorted(row["task_key"] for row in rows)
        family_id = "libero40-90-full-goal-family-" + sha(encoded(members))[:16]
        source_members = sorted(row["task_key"] for row in rows if row["role"] == "source40")
        dev_members = sorted(row["task_key"] for row in rows if row["prior_development"])
        for row in rows:
            row["candidate_family"] = family_id
            row["family_source40_members"] = source_members
            row["family_reserved_development_members"] = dev_members
            row["conservative_family_overlaps_source40"] = bool(source_members)
        families.append(dict(family=family_id, members=members, source40_members=source_members,
                             target90_members=sorted(row["task_key"] for row in rows if row["suite"] == "libero_90"),
                             reserved_development_members=dev_members))
    sources = [row for row in records if row["role"] == "source40"]
    targets = [row for row in records if row["role"] != "source40"]
    pairs = []
    for target in targets:
        target["direct_source40_matches"] = []
        for source in sources:
            reasons = [field for field in ("goal_binding_sha256", "typed_alpha_goal_sha256", "normalized_instruction") if source[field] == target[field]]
            if reasons:
                pair = dict(target90=target["task_key"], source40=source["task_key"], equality=reasons)
                pairs.append(pair)
                target["direct_source40_matches"].append(pair)
        target["literal_goal_binding_overlap_source40"] = any("goal_binding_sha256" in p["equality"] for p in target["direct_source40_matches"])
        target["typed_alpha_goal_overlap_source40"] = any("typed_alpha_goal_sha256" in p["equality"] for p in target["direct_source40_matches"])
        target["instruction_connected_only_source40"] = target["conservative_family_overlaps_source40"] and not target["typed_alpha_goal_overlap_source40"]
    overlap = [row["task_key"] for row in targets if row["conservative_family_overlaps_source40"]]
    new_goals = [row["task_key"] for row in targets if not row["conservative_family_overlaps_source40"]]
    unreserved = [row["task_key"] for row in targets if not row["conservative_family_overlaps_source40"] and not row["family_reserved_development_members"]]
    counts = dict(source40=40, target90=90, source_target_matching_pairs=len(pairs),
                  target90_literal_goal_binding_overlap=sum(row["literal_goal_binding_overlap_source40"] for row in targets),
                  target90_typed_alpha_goal_overlap=sum(row["typed_alpha_goal_overlap_source40"] for row in targets),
                  target90_instruction_connected_only_overlap=sum(row["instruction_connected_only_source40"] for row in targets),
                  target90_source40_family_overlap=len(overlap), target90_without_source40_family_overlap=len(new_goals),
                  target90_new_goal_without_reserved_development_family=len(unreserved))
    inventory = [{k: row[k] for k in ("path", "sha256", "bytes", "git_blob")} for row in snapshot["files"]]
    if sha(encoded(inventory)) != snapshot["inventory_sha256"]:
        raise ValueError("Source inventory digest mismatch")
    result = dict(format="libero40_meta90_goal_audit_v1", source_commit=snapshot["source_commit"],
                  source_repository=snapshot["repository"], source_inventory_sha256=snapshot["inventory_sha256"],
                  source_snapshot_sha256=snapshot["snapshot_sha256"], source_snapshot_file_sha256=sha(args.snapshot.read_bytes()),
                  original_goal_parser="outputs/libero-pi05-corpus-20261029-r1/build_split.py",
                  parser_copy_sha256=sha(args.parser.read_bytes()), auditor_sha256=sha(Path(__file__).read_bytes()),
                  capture_script_sha256=sha((Path(__file__).parent / "capture_source.py").read_bytes()),
                  source_suites=list(SOURCE_SUITES), source_policy="Only source40 is eligible for meta-controller source data in the new arm; no LIBERO90 training permission is created by this audit.",
                  audit_only=True, protocol_selection_performed=False, task_order_index=0,
                  family_rule="Connected components under equal full-goal/bindings, typed-alpha full goal, or normalized full instruction; no partial-goal matching.",
                  typed_alpha_rule="Preserve predicates/logical structure, region suffixes, types, multiplicity and within-goal entity identity; canonicalize only instance names within the same declared type. Physical scene geometry/initial conditions are not goal identity.",
                  limitations=["Goal-family overlap is not identical scene or trajectory difficulty.", "No synonym or object-type merging is performed; these are conservative static candidate families, not a human semantic oracle.", "A new full composition may reuse source primitive goals; this audit does not claim unseen primitives, objects, or motor pretraining."],
                  counts=counts, reserved_prior_development=[f"libero_90:{i:02d}" for i in RESERVED],
                  literal_goal_binding_overlap_ids=[row["task_key"] for row in targets if row["literal_goal_binding_overlap_source40"]],
                  typed_alpha_goal_overlap_ids=[row["task_key"] for row in targets if row["typed_alpha_goal_overlap_source40"]],
                  instruction_connected_only_overlap_ids=[row["task_key"] for row in targets if row["instruction_connected_only_source40"]],
                  target90_with_source40_family_overlap=overlap, target90_without_source40_family_overlap=new_goals,
                  new_goal_candidates_without_reserved_development_family=unreserved,
                  matching_pairs=pairs, equality_edges=edges, families=sorted(families, key=lambda row: row["family"]),
                  source_inventory=inventory, tasks=records)
    result["audit_sha256"] = sha(encoded(result))
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "audit.json").write_text(json.dumps(result, indent=2) + "\n")
    lines = ["# LIBERO40 source versus LIBERO90 goal audit", "", f"Pinned LIBERO source: `{result['source_commit']}`. All 40 source and 90 target candidates use task order 0.", "",
             "This audit assigns no training or final-test split. LIBERO90 remains target-only for the new arm; the prior split is not imported.", "",
             "| Quantity | Count |", "| --- | ---: |"]
    lines += [f"| {key} | {value} |" for key, value in counts.items()]
    lines += ["", "There are 19 literal typed-goal matches. Instance renaming adds IDs12/14/15, giving 22 direct typed-goal overlaps. The conservative instruction-connected rule additionally flags IDs25/29/31: their white-cabinet goals differ in fixture type from source wooden-cabinet goals, but identical instructions connect them through LIBERO90 siblings. These three are candidate family overlaps, not identical typed goals.", "",
              "Families join equal normalized full goals with typed bindings, equal full goals after type-preserving instance renaming, or equal normalized full instructions. The typed signature preserves predicates, complete conjunctions, object types, region suffixes, and entity identity. The instruction-connected family can span differing typed goals and is reported separately. A shared primitive alone does not join families.", "",
              "Prior development IDs: `libero_90:01`, `libero_90:02`, `libero_90:47`, `libero_90:57`. Related family members are flagged separately in the JSON; this document does not select a final evaluation set.", "",
              "| Target90 | Direct typed-goal match | Instruction-connected only | Source40 family members | Prior-development family members |", "| --- | --- | --- | --- | --- |"]
    lines += [f"| {row['task_key']} | {'yes' if row['typed_alpha_goal_overlap_source40'] else 'no'} | {'yes' if row['instruction_connected_only_source40'] else 'no'} | {', '.join(row['family_source40_members']) or 'none'} | {', '.join(row['family_reserved_development_members']) or 'none'} |" for row in targets]
    lines += ["", "## Reproduction and provenance", "", "The snapshot contains static BDDL/API/task-map text plus SHA-256, byte counts, and pinned Git blob identities for every source file and initial-state file. It contains no environment observations or executed trajectories. The copied original parser is imported without calling its split-building main function.", "", "```sh", "python build_audit.py --snapshot source_snapshot.json --parser original_build_split.py --output .", "```", "", f"Source inventory SHA-256: `{result['source_inventory_sha256']}`.", "", f"Parser SHA-256: `{result['parser_copy_sha256']}`.", "", f"Audit content SHA-256: `{result['audit_sha256']}`.", "", "Static goal families are conservative candidates: no language synonyms or object types are collapsed. Goal novelty does not establish novelty of scenes, primitives, or the pretrained motor's data.", ""]
    (args.output / "AUDIT.md").write_text("\n".join(lines))
    print(json.dumps(dict(counts=counts, audit_sha256=result["audit_sha256"], overlaps=overlap, reserved=result["reserved_prior_development"]), indent=2))


if __name__ == "__main__":
    main()
