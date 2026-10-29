"""Read only pinned LIBERO metadata; stdout is the reproducible source snapshot."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess


PIN = "87edbd1b823e464bc66557d213cc45e9f88aa543"
SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_90")


def digest(data):
    return hashlib.sha256(data).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.source_root.resolve()
    git = lambda *args: subprocess.check_output(["git", "-C", str(root), *args]).decode().strip()
    commit, dirty = git("rev-parse", "HEAD"), git("status", "--porcelain")
    if commit != PIN or dirty:
        raise ValueError(f"Expected clean pinned source {PIN}; got {commit}, dirty={bool(dirty)}")
    map_path = "libero/libero/benchmark/libero_suite_task_map.py"
    task_map = ast.literal_eval(ast.parse((root / map_path).read_text()).body[0].value)
    paths = ["README.md", map_path, "libero/libero/benchmark/__init__.py"]
    for suite in SUITES:
        expected = 90 if suite == "libero_90" else 10
        if len(task_map[suite]) != expected:
            raise ValueError(f"Unexpected task count for {suite}")
        for name in task_map[suite]:
            paths.extend([f"libero/libero/bddl_files/{suite}/{name}.bddl",
                          f"libero/libero/init_files/{suite}/{name}.pruned_init"])
    files = []
    for relative in sorted(paths):
        data = (root / relative).read_bytes()
        blob = git("rev-parse", f"HEAD:{relative}")
        if git("hash-object", relative) != blob:
            raise ValueError(f"Working bytes differ from source commit: {relative}")
        row = dict(path=relative, sha256=digest(data), bytes=len(data), git_blob=blob)
        if not relative.endswith(".pruned_init"):
            row["text"] = data.decode()
        files.append(row)
    inventory = [{k: row[k] for k in ("path", "sha256", "bytes", "git_blob")} for row in files]
    result = dict(format="libero40_meta90_static_source_v1", repository="https://github.com/Lifelong-Robot-Learning/LIBERO",
                  source_root=str(root), source_commit=commit, source_clean=True,
                  collection="stdlib file reads and Git metadata only; no simulator, model, NumPy, Torch, or pickle execution",
                  suites=list(SUITES), inventory_sha256=digest(canonical(inventory)), files=files)
    result["snapshot_sha256"] = digest(canonical(result))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
