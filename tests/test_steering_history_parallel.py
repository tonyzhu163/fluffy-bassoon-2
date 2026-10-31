"""Allocated Linux only: synthetic process/artifact tests, no scientific fit.

Source/model admission is isolated in unit fixtures; the production path calls
the existing actual source gate and final-candidate loader without a bypass.
"""
from copy import deepcopy
import json
from pathlib import Path
import random
import socket
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from fluffy_bassoon.meta_rl import steering_history_parallel as parallel
from test_steering_history import profile


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    p = profile()
    p["fit"] = deepcopy(parallel.SCIENTIFIC_FIT)
    p["prefix_episodes"] = 24
    p["shared_reservation"]["fit_gpu_seconds"] = 1440
    prefix = dict(lineages=deepcopy(p["lineage_slots"]), fixture="unit admission isolated")
    refs, admitted = {}, []
    def admit(actual_profile, actual_prefix, *, objective):
        admitted.append(objective)
        assert actual_profile == p and actual_prefix == prefix
        return actual_prefix["lineages"]
    monkeypatch.setattr(parallel.history, "admit_prefix", admit)
    monkeypatch.setattr(parallel, "check_source", lambda source: None)
    for objective in ("ad", "dpt"):
        data = {}
        for partition in ("train", "validation"):
            path = tmp_path/(objective+"-"+partition+".npz")
            path.write_bytes((objective+":"+partition+":not-executed").encode())
            metadata = dict(objective=objective, metadata=dict(partition=partition,
                trial_ids=sorted(row["lineage_id"] for row in prefix["lineages"] if row["partition"]==partition),
                profile_sha256=parallel.history.content_hash(p), prefix_sha256=parallel.history.content_hash(prefix)))
            path.with_suffix(".json").write_text(json.dumps(metadata))
            data[partition] = dict(parallel.file_ref(path), manifest_sha256=parallel.history.digest(path.with_suffix(".json")))
        assembly = dict(format=parallel.history.ASSEMBLY, completed=True, objective=objective,
            profile=p, prefix=prefix, datasets={objective:data})
        path = tmp_path/(objective+"-assembly.json")
        path.write_text(json.dumps(assembly))
        refs[objective] = parallel.file_ref(path)
    root = tmp_path/"fits"
    manifest = parallel.build_manifest(refs, output_root=root, source_commit="a"*40,
        run_id="synthetic-parallel-fixture", expected_hostname=socket.gethostname())
    path = tmp_path/"manifest.json"
    path.write_text(json.dumps(manifest))
    return SimpleNamespace(p=p, prefix=prefix, refs=refs, manifest=manifest,
        ref=parallel.file_ref(path), root=root, admitted=admitted)


def test_fixed_scientific_commands_reuse_gates_and_pair_source_split(prepared):
    assert prepared.admitted == ["ad", "dpt"]
    assert tuple(prepared.manifest["fits"]) == parallel.FIT_IDS
    for key, fit in prepared.manifest["fits"].items():
        command = fit["argv_template"]
        for flag, value in (("--updates","2500"),("--width","128"),("--layers","2"),
                ("--heads","4"),("--batch-size","8"),("--context-length","256"),
                ("--eval-every","250"),("--validation-batches","16"),("--checkpoint-selection","final")):
            assert command[command.index(flag)+1] == value
        assert Path(fit["output"]).name == key
    assert len({fit["output"] for fit in prepared.manifest["fits"].values()}) == 4
    assert len({fit["control"] for fit in prepared.manifest["fits"].values()}) == 4


@pytest.mark.parametrize("bad", ["fixture-profile", "split", "gate", "arbitrary-command"])
def test_pending_data_and_mutated_profiles_splits_commands_fail_before_launch(prepared, monkeypatch, bad):
    if bad == "arbitrary-command":
        value = deepcopy(prepared.manifest)
        value["fits"]["ad-full"]["argv_template"][0] = "/not-the-trainer"
        path = Path(prepared.ref["path"])
        path.write_text(json.dumps(value))
        with pytest.raises(ValueError, match="scientific schedule"):
            parallel.read_manifest(parallel.file_ref(path), expected_hostname=socket.gethostname())
        return
    if bad == "gate":
        def pending(*args, **kwargs): raise ValueError("Actual A2 gate remains pending")
        monkeypatch.setattr(parallel.history, "admit_prefix", pending)
    else:
        for objective, reference in prepared.refs.items():
            value = parallel.history.read_json(reference)
            if bad == "fixture-profile":
                value["profile"]["fit"]["updates"] = 100
            elif objective == "dpt":
                value["datasets"][objective]["train"], value["datasets"][objective]["validation"] = (
                    value["datasets"][objective]["validation"], value["datasets"][objective]["train"])
            path = Path(reference["path"])
            path.write_text(json.dumps(value))
            prepared.refs[objective] = parallel.file_ref(path)
    with pytest.raises(ValueError):
        parallel.build_manifest(prepared.refs, output_root=prepared.root, source_commit="a"*40,
            run_id="synthetic-parallel-fixture", expected_hostname=socket.gethostname())
    assert not prepared.root.exists()


def barrier_children(monkeypatch, prepared, *, failed=None):
    real_popen = parallel.subprocess.Popen
    barrier = prepared.root/"barrier"
    barrier.mkdir(parents=True)
    # A serial launch/wait implementation cannot pass this four-child barrier.
    script = """
import json, os, pathlib, random, sys, time
root, key, fail = pathlib.Path(sys.argv[1]), sys.argv[2], sys.argv[3]
random.seed(0)
(root/(key+'.start')).write_text(json.dumps({'pid':os.getpid(),'draw':random.random(),
    'visible':os.environ.get('CUDA_VISIBLE_DEVICES')}))
deadline=time.monotonic()+10
while len(list(root.glob('*.start'))) < 4:
    if time.monotonic()>deadline: raise SystemExit(9)
    time.sleep(.01)
raise SystemExit(3 if key==fail else 0)
"""
    def spawn(command, **kwargs):
        key = Path(command[command.index("--output-dir")+1]).name
        return real_popen([sys.executable,"-c",script,str(barrier),key,failed or ""], **kwargs)
    monkeypatch.setattr(parallel.subprocess, "Popen", spawn)
    return barrier


def launch(prepared):
    return parallel.run_fits(prepared.ref, placements={key:"cuda:0" for key in parallel.FIT_IDS},
        expected_hostname=socket.gethostname(), execution_batches={"ad-full":2}, threads=1)


def test_real_four_process_barrier_rng_output_and_scheduler_visibility_are_independent(prepared, monkeypatch):
    barrier = barrier_children(monkeypatch, prepared)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-parent-owned")
    records = launch(prepared)
    assert all(r["returncode"]==0 for r in records.values())
    starts = [json.loads(path.read_text()) for path in barrier.glob("*.start")]
    assert len(starts)==4 and len({r["pid"] for r in starts})==4
    assert len({r["draw"] for r in starts})==1  # Same seed, four independent interpreters.
    assert all(r["visible"]=="GPU-parent-owned" for r in starts)
    for key, record in records.items():
        command = parallel.history.read_json(record["command"])
        assert command["output"] == str(prepared.root/key)
        assert command["argv"][command["argv"].index("--batch-size")+1] == "8"
    with pytest.raises(ValueError, match="Never reuse"):
        launch(prepared)


def test_one_fit_failure_does_not_cancel_other_independent_fits_or_allow_aggregation(prepared, monkeypatch):
    barrier_children(monkeypatch, prepared, failed="dpt-full")
    records = launch(prepared)
    assert records["dpt-full"]["returncode"]==3
    assert sum(r["returncode"]==0 for r in records.values())==3
    output = prepared.root/"aggregate.json"
    with pytest.raises(ValueError):
        parallel.collect_fits(prepared.ref, expected_hostname=socket.gethostname(), output=output)
    assert not output.exists()


def completed_artifact_fixture(prepared, monkeypatch):
    """Synthetic metadata/recovery; common model admission is tested elsewhere."""
    weights = {"fixture.weight":torch.tensor([1.])}
    calls = []
    class Policy:
        def state_dict(self): return weights
    def load_student(ref, assembly, *, mode, source_commit, device):
        key = Path(ref["path"]).parent.name
        calls.append(key)
        assert mode==prepared.manifest["fits"][key]["mode"] and source_commit=="a"*40 and device=="cpu"
        contract = json.loads((Path(ref["path"]).parent/"contract.json").read_text())
        return Policy(), dict(contract=contract,validation_nll=2. if mode=="full" else 4.)
    monkeypatch.setattr(parallel.history, "load_student", load_student)
    for key in parallel.FIT_IDS:
        folder = prepared.root/key
        folder.mkdir()
        mode = prepared.manifest["fits"][key]["mode"]
        contract = dict(fixture=True,mode=mode,source_commit="a"*40)
        (folder/"contract.json").write_text(json.dumps(contract))
        (folder/"candidate.pt").write_bytes(b"unit-model-admission-isolated")
        torch.save(dict(format="frozen_vla_history_resume_v1",update=2500,contract=contract,policy=weights,
            optimizer=dict(state={0:dict(step=torch.tensor(2500.))},param_groups=[dict(params=[0])]),
            rng=dict(sampler=np.random.default_rng(0).bit_generator.state,python=random.getstate(),
                numpy=np.random.get_state(),torch=torch.get_rng_state(),cuda=[]),
            best_validation_nll=2.,elapsed_seconds=1.),folder/"latest.pt")
        (folder/"metrics.jsonl").write_text(json.dumps(dict(update=2500,nll=1. if mode=="full" else 2.,
            validation_nll=2. if mode=="full" else 4.))+"\n")
        path = Path(prepared.manifest["fits"][key]["control"])/"process-result.json"
        record = json.loads(path.read_text())
        record["artifacts"]={name:parallel.file_ref(folder/name) for name in
            ("candidate.pt","latest.pt","metrics.jsonl","contract.json")}
        path.write_text(json.dumps(record))
    return calls


def test_aggregate_requires_all_four_final_candidates_and_full_rng_recoveries(prepared, monkeypatch):
    barrier_children(monkeypatch, prepared)
    launch(prepared)
    calls = completed_artifact_fixture(prepared, monkeypatch)
    output = prepared.root/"aggregate.json"
    result = parallel.collect_fits(prepared.ref, expected_hostname=socket.gethostname(), output=output)
    assert calls==list(parallel.FIT_IDS) and set(result["results"])==set(parallel.FIT_IDS)
    assert result["comparisons"]["ad"]["full_minus_static_validation_nll"]==-2.


@pytest.mark.parametrize("mutation", ["partial", "missing-rng", "different-weights", "missing-fit"])
def test_incomplete_or_inconsistent_recovery_never_creates_complete_aggregate(prepared, monkeypatch, mutation):
    barrier_children(monkeypatch, prepared)
    launch(prepared)
    completed_artifact_fixture(prepared, monkeypatch)
    folder = prepared.root/"dpt-static"
    if mutation=="missing-fit":
        (folder/"candidate.pt").unlink()
    else:
        path = folder/"latest.pt"
        value = torch.load(path,weights_only=False)
        if mutation=="partial": value["update"]=100
        elif mutation=="missing-rng": value["rng"].pop("sampler")
        else: value["policy"]["fixture.weight"]=torch.tensor([999.])
        torch.save(value,path)
        record_path = Path(prepared.manifest["fits"]["dpt-static"]["control"])/"process-result.json"
        record = json.loads(record_path.read_text())
        record["artifacts"]["latest.pt"]=parallel.file_ref(path)
        record_path.write_text(json.dumps(record))
    output = prepared.root/"aggregate.json"
    with pytest.raises((ValueError,FileNotFoundError)):
        parallel.collect_fits(prepared.ref, expected_hostname=socket.gethostname(), output=output)
    assert not output.exists()
