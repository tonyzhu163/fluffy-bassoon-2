"""Static, simulator-free checks for the complete RoboTwin object installation."""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import re


OBJECTS_REVISION = "3dc3b798668feb99ac61cc9086d84cbcc3d79186"
OBJECTS_SHA256 = "6aa56b3cf1e1064f7c809308144da36b00815f8b137fef2d7e4de856f8becf27"
MANIFEST_NAME = ".fb2-archive-manifest.json"


def validate_robotwin_assets(root, tasks):
    """Require every archived file for every object family named by the tasks.

    The installation manifest comes from a SHA-256-verified official archive.
    Task sources are parsed, never imported, so a missing asset fails before
    either the model or simulator is loaded. All variants are checked, including
    object names selected dynamically from the task's literal candidate lists.
    File sizes are checked here; archive CRCs are checked during installation.
    """
    root = Path(root).resolve()
    tasks = tuple(tasks)
    objects = root / "assets" / "objects"
    manifest_path = objects / MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Incomplete RoboTwin objects installation: missing {manifest_path}. "
            "Install and verify the full pinned official objects archive first."
        )
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    if (manifest.get("archive_sha256") != OBJECTS_SHA256
            or manifest.get("hf_revision") != OBJECTS_REVISION):
        raise ValueError("RoboTwin object manifest is not the pinned verified archive")
    files = manifest.get("files", [])
    if not files:
        raise ValueError("RoboTwin object manifest has no file inventory")
    by_family = {}
    for entry in files:
        relative = Path(entry["path"])
        if relative.is_absolute() or ".." in relative.parts or len(relative.parts) < 2:
            raise ValueError("Unsafe or invalid RoboTwin object manifest path")
        by_family.setdefault(relative.parts[0], []).append(entry)

    task_families = {}
    for task in tasks:
        if not re.fullmatch(r"[a-zA-Z0-9_]+", task):
            raise ValueError(f"Invalid RoboTwin task name: {task!r}")
        source = root / "envs" / f"{task}.py"
        tree = ast.parse(source.read_text(), filename=str(source))
        families = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                match = re.match(r"^(\d{3}_[^/{}\s]+)(?:/|$)", node.value)
                if match:
                    families.add(match.group(1))
        if not families:
            raise ValueError(f"No statically discoverable object families for {task}")
        task_families[task] = sorted(families)

    families = sorted({name for names in task_families.values() for name in names})
    missing = []
    checked_count = checked_bytes = 0
    for family in families:
        entries = by_family.get(family, [])
        if not entries:
            missing.append(f"{family}: absent from archive manifest")
            continue
        if not any(Path(e["path"]).name.startswith("model_data")
                   and e["path"].endswith(".json") for e in entries):
            missing.append(f"{family}: no model_data JSON in archive manifest")
        for entry in entries:
            path = objects / entry["path"]
            expected_size = int(entry["size"])
            if not path.is_file() or path.stat().st_size != expected_size:
                missing.append(entry["path"])
            checked_count += 1
            checked_bytes += expected_size
    if missing:
        raise FileNotFoundError(
            f"Incomplete RoboTwin assets ({len(missing)} missing/wrong-size files): "
            + ", ".join(missing[:20])
        )
    return {
        "status": "passed",
        "check": "full_archived_object_families_static_file_sizes",
        "robotwin_root": str(root),
        "hf_revision": OBJECTS_REVISION,
        "archive_sha256": OBJECTS_SHA256,
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "task_object_families": task_families,
        "object_family_count": len(families),
        "checked_file_count": checked_count,
        "checked_file_bytes": checked_bytes,
    }
