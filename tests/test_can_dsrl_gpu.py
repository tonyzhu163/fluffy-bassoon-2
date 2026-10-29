"""CPU fixtures for allocation evidence; execute in the allocated Linux job."""
from datetime import datetime, timedelta, timezone
import json

import pytest

from fluffy_bassoon.meta_rl.can_dsrl_gpu import _parent_report


SOURCE = "a" * 40
HOST = "job-fixture123-master-0"


def fixture_report(count):
    assigned = [f"GPU-{i:08x}-0000-0000-0000-{i:012x}" for i in range(count)]
    report = dict(status="passed", purpose="training", allocation_type="baige_container",
                  hostname=HOST, source_commit=SOURCE,
                  created_at_utc=datetime.now(timezone.utc).isoformat(),
                  gpu_identity_mapping=dict(assigned_nvml_uuids=assigned),
                  compute_processes_before=[],
                  devices=[dict(uuid=uuid, allocation_ordinal=i, visibility_token=str(i),
                                allocation_check=dict(elements=1024, result=2048))
                           for i, uuid in enumerate(assigned)])
    return report, assigned


def validate(tmp_path, report, assigned, source=SOURCE, hostname=HOST):
    path = tmp_path / "parent.json"
    path.write_text(json.dumps(report))
    result, digest = _parent_report(path, source, assigned, hostname)
    assert len(digest) == 64
    return result


@pytest.mark.parametrize("count", [1, 8])
def test_complete_parent_allocation(tmp_path, count):
    report, assigned = fixture_report(count)
    assert validate(tmp_path, report, assigned) == report


def test_generic_initial_qualification_is_bound_by_worker_source(tmp_path):
    report, assigned = fixture_report(1)
    del report["source_commit"]
    assert validate(tmp_path, report, assigned) == report
    report, assigned = fixture_report(8)
    del report["source_commit"]
    with pytest.raises(ValueError, match="source"):
        validate(tmp_path, report, assigned)


@pytest.mark.parametrize("count", [1, 8])
def test_recorded_source_must_match(tmp_path, count):
    report, assigned = fixture_report(count)
    report["source_commit"] = "b" * 40
    with pytest.raises(ValueError, match="source"):
        validate(tmp_path, report, assigned)


@pytest.mark.parametrize("mutation", ["hostname", "assigned", "ordinal", "visibility", "cuda_evidence", "occupied"])
def test_parent_identity_and_evidence_cannot_be_substituted(tmp_path, mutation):
    report, assigned = fixture_report(8)
    if mutation == "hostname":
        report["hostname"] = "job-other123-master-0"
    elif mutation == "assigned":
        assigned = list(reversed(assigned[:-1]))
    elif mutation == "ordinal":
        report["devices"][-1]["allocation_ordinal"] = 0
    elif mutation == "visibility":
        report["devices"][-1]["visibility_token"] = "0"
    elif mutation == "cuda_evidence":
        del report["devices"][0]["allocation_check"]
    else:
        report["compute_processes_before"] = [dict(pid=42)]
    with pytest.raises(ValueError):
        validate(tmp_path, report, assigned)


@pytest.mark.parametrize("offset_seconds", [-3601, 60])
def test_stale_or_future_parent_is_rejected(tmp_path, offset_seconds):
    report, assigned = fixture_report(8)
    report["created_at_utc"] = (datetime.now(timezone.utc) + timedelta(seconds=offset_seconds)).isoformat()
    with pytest.raises(ValueError, match="stale|future"):
        validate(tmp_path, report, assigned)


def test_timestamp_requires_timezone(tmp_path):
    report, assigned = fixture_report(8)
    report["created_at_utc"] = datetime.now().isoformat()
    with pytest.raises(ValueError, match="UTC offset"):
        validate(tmp_path, report, assigned)
