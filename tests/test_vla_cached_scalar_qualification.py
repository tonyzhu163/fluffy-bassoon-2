"""Prepared model-free gate tests; execute only on the owner's Linux resources."""
from collections import Counter
from copy import deepcopy
import builtins
import hashlib
import importlib
import json
from pathlib import Path
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import numpy as np
import pytest


SOURCE="a"*40
RUN="libero-cached-scalar-cpu-fixture"
OPENPI="215abfb217dbac7d5f1273282331b9b1866c0479"
GOALS=[f"{suite}:{index:02d}" for index in range(4)
       for suite in ("libero_spatial","libero_object","libero_goal","libero_10")][:14]


def bind(path):
    path=Path(path).resolve()
    return dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def write_json(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value))
    return bind(path)


@pytest.fixture
def gate(monkeypatch):
    # execute_gate installs its probe map for the real engine; restore the
    # test process's original bindings even when callbacks inject a fake map.
    monkeypatch.setenv("LYCO_BAIGE_GPU_MAP","")
    monkeypatch.setenv("LYCO_BAIGE_ISOLATED_UUID","")
    original=builtins.__import__
    def no_model_runtime(name,*args,**kwargs):
        if name.split(".")[0] in {"jax","jaxlib","flax","openpi","openpi_client"}:
            raise AssertionError("Pure gate helper imported model runtime: "+name)
        return original(name,*args,**kwargs)
    monkeypatch.setattr(builtins,"__import__",no_model_runtime)
    return importlib.import_module("fluffy_bassoon.meta_rl.vla_cached_scalar_qualification")


@pytest.fixture
def inputs(tmp_path):
    root=tmp_path/"recorded-inputs";root.mkdir()
    refs={name:write_json(tmp_path/(name+".json"),dict(sha256="f"*64,leaves=1,
                 elements_by_dtype={"float32":8})) for name in
          ("source_receipt","root_qualification_inputs","model_receipt","model_parameter_signature")}
    payload=b"immutable fake recorded-input payload"
    digest=hashlib.sha256(payload).hexdigest();records=[]
    for goal_index,goal in enumerate(GOALS):
        count=57 if goal_index<11 else 56
        for rank in range(count):
            records.append(dict(path=str(root/f"{goal.replace(':','-')}-{rank:03d}.pkl.gz"),
                                sha256=digest,goal_key=goal,rank=rank))
    index=dict(schema="libero_cached_scalar_inputs_v1",goal_keys=GOALS.copy(),population_count=795,
        records=records,provenance=dict(input_root=str(root),source_commit="9"*40,
                                      openpi_commit=OPENPI,**refs))
    return SimpleNamespace(index=index,ref=write_json(tmp_path/"index.json",index),root=root,payload=payload)


@pytest.fixture
def funding(tmp_path,inputs):
    authority=write_json(tmp_path/"original-authority.json",dict(scope="CPU fixture only"))
    reservation=write_json(tmp_path/"canonical-reservation.json",dict(scope="CPU fixture only"))
    return dict(schema="libero_cached_scalar_gate_funding_v1",state="funded",funded_by="root",
        run_id=RUN,source_commit=SOURCE,max_gpu_seconds=600,robot_actions=0,sgd_updates=0,
        original_compute_budget_hours=32,budget_expanded=False,input_index=inputs.ref,
        original_32h_ledger_authority=authority,canonical_reservation=reservation)


def test_funding_accepts_only_the_exact_root_bound_record(gate,funding,inputs):
    gate.validate_funding(funding,run_id=RUN,source_commit=SOURCE,inputs_ref=inputs.ref)


@pytest.mark.parametrize("field,value",[
    ("state","unfunded"),("funded_by","agent"),("max_gpu_seconds",601),
    ("robot_actions",1),("sgd_updates",1),("original_compute_budget_hours",33),
    ("budget_expanded",True),("source_commit","b"*40),("run_id","another-run")])
def test_funding_rejects_budget_scope_or_identity_expansion(gate,funding,inputs,field,value):
    funding[field]=value
    with pytest.raises((ValueError,RuntimeError)):
        gate.validate_funding(funding,run_id=RUN,source_commit=SOURCE,inputs_ref=inputs.ref)


@pytest.mark.parametrize("reference",["input_index","original_32h_ledger_authority","canonical_reservation"])
def test_funding_hash_bindings_cannot_be_substituted(gate,funding,inputs,reference):
    funding[reference]=dict(funding[reference],sha256="0"*64)
    with pytest.raises((ValueError,RuntimeError)):
        gate.validate_funding(funding,run_id=RUN,source_commit=SOURCE,inputs_ref=inputs.ref)


def test_selection_is_fixed_balanced_first_middle_last_and_does_not_claim795_tests(gate,inputs):
    selected=gate.select_records(inputs.index)
    assert len(selected)==42 and inputs.index["population_count"]==795
    assert Counter(row["goal_key"] for row in selected)==Counter({goal:3 for goal in GOALS})
    assert [row["goal_key"] for row in selected]==GOALS*3
    for index,goal in enumerate(GOALS):
        count=57 if index<11 else 56
        assert [row["rank"] for row in selected if row["goal_key"]==goal]==[0,count//2,count-1]
    assert selected==gate.select_records(deepcopy(inputs.index))


@pytest.mark.parametrize("change",["population","missing_record","missing_goal","target_goal","duplicate_rank","relative_path","bad_sha"])
def test_declared_population_or_goal_identity_mismatch_fails_before_models(gate,inputs,change):
    index=inputs.index
    if change=="population":index["population_count"]=794
    elif change=="missing_record":index["records"].pop()
    elif change=="missing_goal":index["goal_keys"].pop()
    elif change=="target_goal":index["goal_keys"][0]="libero_90:00"
    elif change=="duplicate_rank":index["records"][1]["rank"]=0
    elif change=="relative_path":index["records"][0]["path"]="relative.pkl.gz"
    else:index["records"][0]["sha256"]="invalid"
    with pytest.raises((ValueError,RuntimeError)):
        gate.select_records(index)


def materialize_selected(gate,inputs):
    for row in gate.select_records(inputs.index):
        path=Path(row["path"]);path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(inputs.payload)
    return write_json(Path(inputs.ref["path"]),inputs.index)


def test_selected_input_hashes_and_provenance_are_read_before_model_admission(gate,inputs):
    reference=materialize_selected(gate,inputs)
    loaded=gate.load_index(reference)
    assert gate.select_records(loaded)==gate.select_records(inputs.index)


@pytest.mark.parametrize("change",["input_hash","provenance_hash","outside_root","symlink_outside"])
def test_input_loader_rejects_changed_or_escaping_selected_records(gate,inputs,tmp_path,change):
    materialize_selected(gate,inputs)
    selected=gate.select_records(inputs.index)[0]
    if change=="input_hash":Path(selected["path"]).write_bytes(b"changed input")
    elif change=="provenance_hash":
        Path(inputs.index["provenance"]["source_receipt"]["path"]).write_text("changed provenance")
    else:
        outside=tmp_path/"outside.pkl.gz";outside.write_bytes(inputs.payload)
        replacement=outside
        if change=="symlink_outside":
            replacement=inputs.root/"outside-link.pkl.gz";replacement.symlink_to(outside)
        match=next(row for row in inputs.index["records"]
                   if row["goal_key"]==selected["goal_key"] and row["rank"]==selected["rank"])
        match["path"]=str(replacement)
    reference=write_json(Path(inputs.ref["path"]),inputs.index)
    with pytest.raises((ValueError,RuntimeError)):
        gate.load_index(reference)


def junit(path,*,cases=43,failures=0,errors=0,skipped=0):
    root=ET.Element("testsuites")
    suite=ET.SubElement(root,"testsuite",tests=str(cases),failures=str(failures),
                        errors=str(errors),skipped=str(skipped))
    for index in range(cases):
        case=ET.SubElement(suite,"testcase",classname="tests.test_vla_cached_scalar_motor",name=f"case-{index}")
        for tag,total in (("failure",failures),("error",errors),("skipped",skipped)):
            if index<total:ET.SubElement(case,tag)
    ET.ElementTree(root).write(path)
    return path


def test_cpu_gate_requires_a_real_complete43_case_xml_receipt(gate,tmp_path):
    receipt=gate.verify_cpu_report(junit(tmp_path/"cpu.xml"),0)
    assert isinstance(receipt,dict)


@pytest.mark.parametrize("change",["returncode","42","44","failure","error","skipped","lying_count"])
def test_failed_partial_or_inconsistent_cpu_gate_cannot_admit_models(gate,tmp_path,change):
    kwargs={};code=1 if change=="returncode" else 0
    if change in ("42","44"):kwargs["cases"]=int(change)
    if change in ("failure","error","skipped"):kwargs[{"failure":"failures","error":"errors","skipped":"skipped"}[change]]=1
    path=junit(tmp_path/"cpu.xml",**kwargs)
    if change=="lying_count":
        tree=ET.parse(path);tree.getroot().find("testsuite").set("tests","42");tree.write(path)
    with pytest.raises((ValueError,RuntimeError)):
        gate.verify_cpu_report(path,code)


def arrays():
    return np.zeros(4104,np.float32),np.zeros((10,7),np.float32),[[1,2],[3,4]]


def test_original_feature_action_tolerances_accept_only_finite_matching_shapes(gate):
    feature,action,keys=arrays();other_feature=feature.copy();other_action=action.copy()
    other_feature[0]=1e-4;other_action[0,0]=5e-6
    result=gate.pair_comparison(feature,other_feature,action,other_action,keys,deepcopy(keys),expected_keys=deepcopy(keys))
    assert isinstance(result,dict)


@pytest.mark.parametrize("change",["feature_limit","action_limit","feature_shape","action_shape","feature_nan","action_inf"])
def test_feature_or_action_parity_failure_never_relaxes_original_thresholds(gate,change):
    feature,action,keys=arrays();other_feature=feature.copy();other_action=action.copy()
    if change=="feature_limit":other_feature[0]=2.1e-4
    elif change=="action_limit":other_action[0,0]=1.1e-5
    elif change=="feature_shape":other_feature=other_feature[:-1]
    elif change=="action_shape":other_action=other_action[:-1]
    elif change=="feature_nan":other_feature[0]=np.nan
    else:other_action[0,0]=np.inf
    with pytest.raises((ValueError,RuntimeError)):
        gate.pair_comparison(feature,other_feature,action,other_action,keys,deepcopy(keys),expected_keys=deepcopy(keys))


@pytest.mark.parametrize("change",["cached","both_skip","both_double","intermediate"])
def test_equal_endpoints_alone_do_not_qualify_rng_progression(gate,change):
    feature,action,expected=arrays();native=deepcopy(expected);cached=deepcopy(expected)
    if change=="cached":cached[-1]=[5,6]
    elif change=="both_skip":native=cached=[[1,2],[1,2]]
    elif change=="both_double":native=cached=[[1,2],[5,6]]
    else:
        expected=[[1,2],[3,4],[5,6]];native=[[1,2],[8,9],[5,6]];cached=deepcopy(expected)
    with pytest.raises((ValueError,RuntimeError)):
        gate.pair_comparison(feature,feature,action,action,native,cached,expected_keys=expected)


class Engine:
    """Explicit fake kernel engine; no model/runtime/simulator is imported."""
    def __init__(self,*,clock=None,signature_changes=False,fail_operation=None,partial_after=None):
        self.clock=clock;self.signature_changes=signature_changes;self.fail_operation=fail_operation
        self.partial_after=partial_after;self.pairs=[];self.stales=[];self.signatures=0
        self.expected_signature=dict(sha256="f"*64,leaves=1,elements_by_dtype={"float32":8})

    def signature(self):
        self.signatures+=1;value=deepcopy(self.expected_signature)
        if self.signature_changes and self.signatures>1:value["sha256"]="e"*64
        return value

    def pair(self,ref,*,unbounded=False,repeat=False):
        self.pairs.append(dict(ref=deepcopy(ref),unbounded=unbounded,repeat=repeat))
        if self.partial_after==len(self.pairs) and self.clock is not None:self.clock[0]=450.
        return dict(passed=self.fail_operation!="pair",maximum_feature_delta=0.,maximum_action_delta=0.,
                    rng_sequence_equal=True)

    def stale(self,ref,kind):
        self.stales.append(dict(ref=deepcopy(ref),kind=kind))
        return dict(passed=self.fail_operation!="stale",rng_unchanged=True)

    def counts(self):
        return dict(native_infer_calls=len(self.pairs),cached_infer_calls=len(self.pairs),
                    stale_checks=len(self.stales),full_parameter_hashes=self.signatures)


def test_full_fake_replay_reports42_recorded_inputs_separately_from795_population(gate,inputs,tmp_path,monkeypatch):
    clock=[0.];monkeypatch.setattr(gate.time,"monotonic",lambda:clock[0])
    engine=Engine();output=tmp_path/"replay";output.mkdir()
    report=gate.run_replay(inputs.index,engine,gate.Deadline(seconds=540,started=0.),output)
    assert report["status"]=="passed" and report["qualified"] and report["adoption_allowed"]
    assert report["population_records"]==795
    assert report["expected_records"]==report["completed_records"]==42
    assert report["expected_noise_probes"]==report["completed_noise_probes"]==14
    assert report["expected_stale_probes"]==report["completed_stale_probes"]==28
    assert len(engine.pairs)==56 and len(engine.stales)==28 and engine.signatures==2
    assert Counter(row["ref"]["goal_key"] for row in engine.pairs if not row["unbounded"])==Counter({key:3 for key in GOALS})
    assert Counter(row["ref"]["goal_key"] for row in engine.pairs if row["unbounded"])==Counter({key:1 for key in GOALS})
    assert sum(row["repeat"] for row in engine.pairs)==14
    assert all(row["repeat"] for row in engine.pairs[:14])
    assert Counter(row["kind"] for row in engine.stales)==Counter(content=14,instruction=14)
    assert set(report["tested_goal_keys"])==set(GOALS)
    assert report["weights_unchanged"] is True
    assert (output/"replay-report.json").is_file()


def test_deadline_partial_never_adopts_or_inflates_tested_population(gate,inputs,tmp_path,monkeypatch):
    clock=[0.];monkeypatch.setattr(gate.time,"monotonic",lambda:clock[0])
    engine=Engine(clock=clock,partial_after=5);output=tmp_path/"partial";output.mkdir()
    report=gate.run_replay(inputs.index,engine,gate.Deadline(seconds=540,started=0.),output)
    assert report["status"]=="partial" and not report["qualified"] and not report["adoption_allowed"]
    assert report["population_records"]==795 and report["completed_records"]==5
    assert len(engine.pairs)==5 and not engine.stales
    assert report["completed_noise_probes"]==report["completed_stale_probes"]==0


@pytest.mark.parametrize("operation",["pair","stale"])
def test_failed_comparison_stops_without_retry_or_adoption(gate,inputs,tmp_path,monkeypatch,operation):
    monkeypatch.setattr(gate.time,"monotonic",lambda:0.)
    engine=Engine(fail_operation=operation);output=tmp_path/operation;output.mkdir()
    report=gate.run_replay(inputs.index,engine,gate.Deadline(seconds=540,started=0.),output)
    assert report["status"]=="failed" and not report["qualified"] and not report["adoption_allowed"]
    assert len(engine.pairs)==(1 if operation=="pair" else 56)
    assert len(engine.stales)==(0 if operation=="pair" else 1)


def test_changed_full_parameter_signature_rejects_even_perfect_output_parity(gate,inputs,tmp_path,monkeypatch):
    monkeypatch.setattr(gate.time,"monotonic",lambda:0.)
    engine=Engine(signature_changes=True);output=tmp_path/"weights";output.mkdir()
    report=gate.run_replay(inputs.index,engine,gate.Deadline(seconds=540,started=0.),output)
    assert report["status"]=="failed" and not report["qualified"] and not report["adoption_allowed"]
    assert report["weights_unchanged"] is False and engine.signatures==2


def test_initial_model_signature_must_match_bound_source_parameters(gate,inputs,tmp_path,monkeypatch):
    monkeypatch.setattr(gate.time,"monotonic",lambda:0.)
    engine=Engine();engine.expected_signature=dict(engine.expected_signature,sha256="0"*64)
    engine.signature=lambda:dict(sha256="f"*64,leaves=1,elements_by_dtype={"float32":8})
    output=tmp_path/"wrong-source-weights";output.mkdir()
    report=gate.run_replay(inputs.index,engine,gate.Deadline(seconds=540,started=0.),output)
    assert not report["qualified"] and not report["adoption_allowed"] and not engine.pairs


def execution_args(tmp_path,inputs,funding):
    funding_ref=write_json(tmp_path/"funding.json",funding)
    return SimpleNamespace(run_id=RUN,source_commit=SOURCE,expected_hostname="job-cpu-fake",
        funding=Path(funding_ref["path"]),funding_sha256=funding_ref["sha256"],
        inputs=Path(inputs.ref["path"]),inputs_sha256=inputs.ref["sha256"],output=tmp_path/"gate",
        checkpoint=tmp_path/"model",openpi_root=tmp_path/"openpi")


def test_admission_cpu43_and_probe_all_precede_engine_creation(gate,inputs,funding,tmp_path,monkeypatch):
    args=execution_args(tmp_path,inputs,funding);events=[]
    monkeypatch.setattr(gate,"validate_context",lambda value:events.append("context"))
    monkeypatch.setattr(gate,"load_index",lambda *a,**k:deepcopy(inputs.index))
    monkeypatch.setattr(gate.time,"monotonic",lambda:0.)
    def cpu(*values):events.append("cpu43");return dict(passed=True,tests=43)
    def probe(*values):events.append("probe");return dict(status="passed",devices=[dict(uuid="GPU-fake")])
    def engine(*values):events.append("engine");return Engine()
    report=gate.execute_gate(args,cpu_runner=cpu,probe_runner=probe,engine_factory=engine)
    assert events.index("context")<events.index("cpu43")<events.index("probe")<events.index("engine")
    assert report["qualified"] and report["adoption_allowed"]


@pytest.mark.parametrize("failure",["funding","index","cpu43","probe"])
def test_bad_admission_never_constructs_a_model(gate,inputs,funding,tmp_path,monkeypatch,failure):
    if failure=="funding":funding["state"]="unfunded"
    args=execution_args(tmp_path,inputs,funding);events=[]
    monkeypatch.setattr(gate,"validate_context",lambda value:None)
    def index(*args,**kwargs):
        if failure=="index":raise ValueError("input hash mismatch")
        return deepcopy(inputs.index)
    monkeypatch.setattr(gate,"load_index",index)
    def cpu(*values):events.append("cpu43");return dict(passed=failure!="cpu43",tests=43)
    def probe(*values):events.append("probe");return dict(status="failed" if failure=="probe" else "passed",devices=[dict(uuid="GPU-fake")])
    def engine(*values):events.append("engine");raise AssertionError("model was constructed before admission")
    with pytest.raises((ValueError,RuntimeError)):
        gate.execute_gate(args,cpu_runner=cpu,probe_runner=probe,engine_factory=engine)
    assert "engine" not in events
    if failure in ("funding","index"):assert not events
    if failure=="cpu43":assert events==["cpu43"]
