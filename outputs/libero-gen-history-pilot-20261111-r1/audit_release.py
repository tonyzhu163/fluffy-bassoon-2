"""Static, standard-library-only release inventory; never loads a model."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess


def sexpr(text):
    stack = [[]]
    for token in re.findall(r"\(|\)|[^\s()]+", text):
        if token == "(":
            stack.append([])
        elif token == ")":
            if len(stack) == 1:
                raise ValueError("Unbalanced BDDL closing parenthesis")
            node = stack.pop()
            stack[-1].append(node)
        else:
            stack[-1].append(token)
    if len(stack) != 1 or len(stack[0]) != 1:
        raise ValueError("Unbalanced BDDL")
    return stack[0][0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.source.resolve()
    commit = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    expected = "58125e682acceefef3d60c071212dc483a4db1c9"
    if commit != expected:
        raise SystemExit(f"Wrong source: {commit} != {expected}")
    base = root / "behavior_prompting/train_network"
    source_paths = [
        "config/libero_policy_dunetp.yaml", "config/task/libero_defaults.yaml",
        "config/task/liberogen_spatial_combination.yaml", "env_runner/libero_image_runner.py",
        "policy/diffusion_unet_policy.py", "model/prompt/prompt_obs_encoder.py",
        "scripts/libero/gen_extra_libero_envs_configs/tasks_spec.yaml",
    ]
    splits = {}
    groups = {}
    for split in ("libero_spatial_selected_combinations_view", "libero_spatial_selected_combinations_inverse_view"):
        rows = []
        for path in sorted((base / "env/libero/bddl_files" / split).glob("*.bddl")):
            tree = sexpr(path.read_text())
            sections = {node[0]: node[1:] for node in tree if isinstance(node, list) and node}
            physical = {k: sections[k] for k in (":regions", ":fixtures", ":objects", ":init")}
            physical_hash = hashlib.sha256(json.dumps(physical, sort_keys=True).encode()).hexdigest()
            row = {"task": path.stem, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                   "goal": sections.get(":goal"), "physical_description_sha256": physical_hash}
            rows.append(row)
            groups.setdefault(physical_hash, []).append({"split": split, **row})
        splits[split] = {"bddl_count": len(rows), "tasks": rows}
    report = {"source_commit": commit, "audit_type": "static_only_no_simulator_or_model",
              "source_file_sha256": {p: hashlib.sha256((base / p).read_bytes()).hexdigest() for p in source_paths},
              "splits": splits,
              "shared_physical_description_groups": [g for g in groups.values() if len(g) > 1],
              "limitations": ["Matching BDDL physical descriptions does not prove matching reset arrays.",
                              "HDF5 availability and frozen motor execution remain unverified.",
                              "No custom relational layout or learned reader has been qualified."]}
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"commit": commit, "counts": {k: v["bddl_count"] for k, v in splits.items()},
                      "shared_physical_groups": len(report["shared_physical_description_groups"])}))


if __name__ == "__main__":
    main()
