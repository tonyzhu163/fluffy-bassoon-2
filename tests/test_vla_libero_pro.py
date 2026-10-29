import hashlib
import json

import pytest

from fluffy_bassoon.meta_rl import vla_libero_pro as pro


def test_predeclared_pro_cases_keep_training_lineage_and_protocol():
    assert set(pro.CASES) == {"swap", "appearance"}
    assert {x["case_id"] for x in pro.CASES.values()} == {
        "pro_object_swap_init0", "pro_object_appearance_init0"}
    assert pro.ORIGIN_BASE["train_task_key"] == "libero_90:47"
    assert pro.ORIGIN_BASE["split"] == "train"
    assert pro.SETTLE_STEPS == 5


def test_changed_data_rejected_before_deserialization(tmp_path, monkeypatch):
    spec = pro.CASES["swap"]
    files = {}
    for kind, suffix in [("bddl_files", ".bddl"), ("init_files", ".pruned_init")]:
        path = tmp_path / kind / spec["family"] / f"{pro.TASK_STEM}{suffix}"
        path.parent.mkdir(parents=True)
        path.write_bytes(b"fixture")
        files[kind] = path
    receipt = tmp_path / "READY.json"
    receipt.write_text(json.dumps(dict(status="verified", revision=pro.DATA_REVISION,
        source_commit=pro.PRO_COMMIT)))
    monkeypatch.setattr(pro, "DATA_RECEIPT", hashlib.sha256(receipt.read_bytes()).hexdigest())
    fixture_sha = hashlib.sha256(b"fixture").hexdigest()
    monkeypatch.setitem(pro.CASES, "swap", dict(spec, bddl_sha256=fixture_sha, init_sha256=fixture_sha))
    assert pro.declared_files(tmp_path, "swap")[:2] == (files["bddl_files"], files["init_files"])
    files["init_files"].write_bytes(b"altered")
    with pytest.raises(ValueError, match="bytes changed"):
        pro.declared_files(tmp_path, "swap")
    with pytest.raises(ValueError, match="predeclared"):
        pro.declared_files(tmp_path, "test-variant")
