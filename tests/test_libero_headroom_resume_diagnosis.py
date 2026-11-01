"""Four focused pure-CPU inspector regression cases; no model initialization."""
import importlib.util
from pathlib import Path
import pytest

spec=importlib.util.spec_from_file_location("resume_inspector",Path("scripts/diagnostics/libero_headroom_resume_diagnosis.py"))
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)

def test_four_original_predicates_are_reported_independently():
    model=dict(updates=3,replay_rows=2,signature="saved")
    fields=module.guard_predicates({"e":1},{"e":1},3,2,"other",model)
    assert fields==dict(training_equal=True,updates_equal=True,replay_rows_equal=True,raw_restored_signature_equal=False)
    assert module.classify(fields,True,True,True)=="identical_saved_state_raw_representation_differs"
    for name in ("training_equal","updates_equal","replay_rows_equal"):
        changed=dict(fields);changed[name]=False
        assert module.classify(changed,True,True,True)=="cursor_or_counter_mismatch"

def test_order_change_does_not_waive_leaf_or_rng_mismatch():
    expected={"actor":{"z":1,"a":2}};actual={"actor":{"a":2,"z":1}}
    assert module.mapping_order_changes(expected,actual)==[dict(path="state/actor",expected_keys=["z","a"],actual_keys=["a","z"])]
    fields=dict(training_equal=True,updates_equal=True,replay_rows_equal=True,raw_restored_signature_equal=False)
    assert module.classify(fields,True,False,True)=="restored_state_or_rng_mismatch"
    assert module.classify(fields,True,True,False)=="restored_state_or_rng_mismatch"
    assert module.classify(fields,False,True,True)=="stored_agent_bytes_do_not_match_original_cursor_signature"

@pytest.mark.parametrize("kind",["changed","symlink"])
def test_bound_input_rejects_changed_bytes_or_alias_before_payload_loading(tmp_path,kind):
    path=tmp_path/"payload";path.write_bytes(b"closed")
    expected=module.sha(path)
    if kind=="changed":path.write_bytes(b"changed")
    else:
        alias=tmp_path/"alias";alias.symlink_to(path);path=alias
    with pytest.raises(ValueError,match="Exact regular SHA-bound input"):
        module.read_bound(dict(path=str(path),sha256=expected))
