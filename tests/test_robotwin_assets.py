"""Static install guard tests; run on allocated Linux with the runtime suite."""
import json

import pytest

from fluffy_bassoon.rl.robotwin_assets import (
    MANIFEST_NAME,
    OBJECTS_REVISION,
    OBJECTS_SHA256,
    validate_robotwin_assets,
)


def test_dynamic_candidate_families_require_every_archived_variant(tmp_path):
    (tmp_path / "envs").mkdir()
    (tmp_path / "envs" / "example.py").write_text(
        'object_list = ["001_bottle", "018_microphone"]\n'
    )
    objects = tmp_path / "assets" / "objects"
    objects.mkdir(parents=True)
    entries = []
    for family in ("001_bottle", "018_microphone"):
        for name in ("model_data0.json", "model_data1.json", "visual/base1.glb"):
            relative = f"{family}/{name}"
            path = objects / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"{}")
            entries.append({"path": relative, "size": 2, "crc32": 0})
    (objects / MANIFEST_NAME).write_text(json.dumps({
        "archive_sha256": OBJECTS_SHA256,
        "hf_revision": OBJECTS_REVISION,
        "files": entries,
    }))
    receipt = validate_robotwin_assets(tmp_path, ["example"])
    assert receipt["object_family_count"] == 2
    assert receipt["checked_file_count"] == 6
    # A single nominal model being present is insufficient for a sampled task.
    (objects / "018_microphone" / "visual" / "base1.glb").unlink()
    with pytest.raises(FileNotFoundError, match="visual/base1.glb"):
        validate_robotwin_assets(tmp_path, ["example"])


def test_legacy_partial_install_is_not_admitted(tmp_path):
    with pytest.raises(FileNotFoundError, match="full pinned official"):
        validate_robotwin_assets(tmp_path, ["handover_mic"])
