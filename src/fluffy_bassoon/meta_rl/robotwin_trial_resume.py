"""Hash-bound recovery of an already selected trial at an episode boundary.

This is artifact validation, not simulator-state recovery. Completed SAC
episodes already present in replay are never observed or optimized again.
Only the trained episode-reset PPO controller can recover a nonempty prefix;
persistent recurrent state is deliberately not guessed from partial data.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re

import numpy as np
import torch


SCHEMA = "robotwin_episode_boundary_resume_v1"


def file_sha256(path):
    digest=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda:stream.read(8<<20),b""):digest.update(block)
    return digest.hexdigest()


def artifact_digest(value):
    """Same typed tensor/array digest as the independent checkpoint audit."""
    digest=hashlib.sha256()
    def visit(item):
        if isinstance(item,torch.Tensor):
            digest.update(b"tensor"+str(item.dtype).encode()+str(tuple(item.shape)).encode())
            digest.update(item.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(item,np.ndarray):
            digest.update(b"ndarray"+str(item.dtype).encode()+str(item.shape).encode())
            digest.update(np.ascontiguousarray(item).tobytes())
        elif isinstance(item,dict):
            digest.update(b"dict")
            for key in sorted(item,key=lambda key:(str(type(key)),str(key))):visit(key);visit(item[key])
        elif isinstance(item,(list,tuple)):
            digest.update(str(type(item)).encode())
            for row in item:visit(row)
        else:digest.update(str(type(item)).encode()+repr(item).encode())
    visit(value)
    return digest.hexdigest()


def _require(condition,message):
    if not condition:raise ValueError("Trial continuation: "+message)


def _bound(ref,*,root=None):
    _require(isinstance(ref,dict) and set(("path","sha256"))<=set(ref),"missing artifact binding")
    path=Path(ref["path"])
    if root is not None:
        path=(root/path).resolve()
        _require(path.is_relative_to(root.resolve()),"artifact escapes its corpus")
    _require(re.fullmatch(r"[0-9a-f]{64}",str(ref["sha256"])) is not None,"invalid artifact SHA256")
    _require(path.is_file() and file_sha256(path)==ref["sha256"],f"artifact hash mismatch: {path}")
    return path


def _json(ref,*,root=None):return json.loads(_bound(ref,root=root).read_text())


def _arrays(ref,root):
    path=_bound(ref,root=root)
    with np.load(path,allow_pickle=False) as archive:
        values={key:archive[key].copy() for key in archive.files}
    _require(set(values)==set(ref["arrays"]),"NPZ array keys differ from their receipt")
    for name,array in values.items():
        metadata=ref["arrays"][name]
        digest=hashlib.sha256(str(array.dtype).encode()+str(array.shape).encode()+array.tobytes()).hexdigest()
        _require(not array.dtype.hasobject and str(array.dtype)==metadata["dtype"]
                 and list(array.shape)==metadata["shape"] and digest==metadata["sha256"],"array identity mismatch")
        _require(array.dtype.kind not in "fc" or np.isfinite(array).all(),"nonfinite recorded array")
    return values


@dataclass
class TrialContinuation:
    method: str
    manifest_sha256: str
    receipt: dict
    trial: dict | None
    episodes: list
    prefix_rollout: dict | None
    prefix_steps: int
    prefix_chunks: int

    @property
    def start_episode(self):return len(self.episodes)


def load_trial_continuation(path,sha256,*,checkpoint_path,checkpoint,execution_commit,lineage_commit):
    """Validate all evidence before a source learner or simulator is advanced."""
    path=Path(path)
    _require(file_sha256(path)==sha256,"manifest hash mismatch")
    manifest=json.loads(path.read_text())
    _require(manifest.get("schema")==SCHEMA,"unknown manifest schema")
    _require(set(("execution_source_commit","lineage_source_commit","method","checkpoint","expected",
        "corpus_root","corpus_contract","progress","episodes_file","ledger_snapshot","kind","prefix"))<=set(manifest),
        "manifest evidence is incomplete")
    _require(manifest.get("execution_source_commit")==execution_commit
             and manifest.get("lineage_source_commit")==lineage_commit,"execution/lineage pin mismatch")
    _require(re.fullmatch(r"[0-9a-f]{40}",lineage_commit or "") is not None,"invalid lineage pin")
    _require(re.fullmatch(r"[0-9a-f]{40}",execution_commit or "") is not None,"invalid execution pin")
    _require(_bound(manifest["checkpoint"]).resolve()==Path(checkpoint_path).resolve(),"different resume checkpoint")
    contract=checkpoint["contract"];method=manifest["method"]
    _require(contract["source_commit"]==lineage_commit and contract["methods"]==[method],"single-arm lineage contract mismatch")
    _require(contract.get("source_iterations") and contract["train_episodes"]==4,"whole-K4 source contract required")
    saved=checkpoint["states"][method];learner=saved["learner"];expected=manifest["expected"]
    for name in ("source_steps","completed_source_trials","completed_source_iterations"):
        _require(saved[name]==expected[name],f"checkpoint {name} counter mismatch")
    _require(artifact_digest(saved["schedule"])==expected["schedule_digest"],"task sampler identity mismatch")
    _require([artifact_digest(rollout) for rollout,_ in saved.get("pending_rollouts",[])]
             ==expected["pending_rollout_digests"],"pending PPO data identity mismatch")
    for name,value in expected["learner_counters"].items():
        _require(learner.get(name)==value,f"learner {name} counter mismatch")
    root=Path(manifest["corpus_root"]).resolve()
    corpus_contract=_json(manifest["corpus_contract"])
    _require(corpus_contract["source_commit"]==lineage_commit,"corpus lineage differs")
    progress=[json.loads(line) for line in _bound(manifest["progress"]).read_text().splitlines() if line]
    episode_rows=[json.loads(line) for line in _bound(manifest["episodes_file"]).read_text().splitlines() if line]
    ledger=_json(manifest["ledger_snapshot"])
    _require(ledger["header"]["contract"]==json.loads(json.dumps(contract,allow_nan=False)),"physical ledger contract differs")
    _require(ledger.get("pending_actions",0)==ledger.get("uncertain_actions",0)==0
             and not ledger.get("unresolved_decisions"),"unresolved physical action intents")
    trial=manifest.get("trial");prefix=manifest.get("prefix",[])
    receipt=dict(schema=SCHEMA,manifest_path=str(path.resolve()),manifest_sha256=sha256,
        execution_source_commit=execution_commit,lineage_source_commit=lineage_commit,
        checkpoint=manifest["checkpoint"],method=method,ledger_journal_sha256=ledger["journal_sha256"])
    for episode_manifest_path in (root/"episodes").glob("*/manifest.json"):
        _require(not json.loads(episode_manifest_path.read_text()).get("pending_decision"),
                 "corpus contains an unresolved simulator decision")
    if manifest.get("kind")=="whole_trial_boundary":
        _require(trial is None and not prefix and saved["schedule"]["trials"]==saved["completed_source_trials"],
                 "whole-trial continuation has an uncompleted selected task")
        _require(learner["primitive_steps"]==saved["source_steps"],"uncredited learner actions at whole-trial boundary")
        return TrialContinuation(method,sha256,receipt,None,[],None,0,0)
    _require(manifest.get("kind")=="selected_trial" and isinstance(trial,dict),"selected trial is missing")
    _require(saved["schedule"]["trials"]==saved["completed_source_trials"]+1==trial["schedule_trials"],
             "selected task cursor was not preserved")
    _require(trial["trial_id"]==f"source:{method}:{trial['schedule_trials']}"
             and len(trial["seeds"])==4 and len(set(trial["seeds"]))==4,"invalid original trial identity")
    starts=[row for row in progress if row.get("stage")=="source_trial_start" and row.get("method")==method]
    _require(bool(starts),"selected-trial progress is missing")
    last=starts[-1]
    for key in ("task","seeds","source_iteration","trial_in_iteration"):
        _require(last[key]==trial[key],f"selected trial {key} differs from progress")
    _require(last["source_steps"]==saved["source_steps"]
             and trial["source_iteration"]==saved["completed_source_iterations"]+1
             and trial["trial_in_iteration"]==saved["completed_source_trials"]%contract["ppo_trials_per_update"]+1,
             "selected trial progress counters differ")
    _require(0<=len(prefix)<4,"prefix must end before the original final episode")
    _require(method!="ppo" or not prefix,"persistent nonempty partial trial cannot be reconstructed")
    if method=="ppo_reset":_require(learner["config"]["reset_each_episode"],"partial PPO requires trained episode reset")
    expected_dirs=[];rows=[];chunks=[];prefix_steps=0;prefix_chunk_count=0
    for index,item in enumerate(prefix):
        reset_path=_bound(item["reset"]);episode_dir=reset_path.parent
        _require(episode_dir.is_relative_to(root/"episodes"),"episode is outside the bound corpus")
        expected_dirs.append(episode_dir.resolve());reset=json.loads(reset_path.read_text())
        episode_manifest=_json(item["manifest"])
        _require(Path(item["manifest"]["path"]).resolve()==episode_dir/"manifest.json","episode manifest path differs")
        _require(reset["context"]["trial_id"]==trial["trial_id"] and reset["context"]["method"]==method
                 and reset["context"]["phase"]=="source","prefix belongs to another trial")
        _require(reset["reset_info"]["requested_seed"]==trial["seeds"][index]
                 and reset["reset_info"]["task"]==trial["task"],"original episode seed/task differs")
        _require(not episode_manifest.get("pending_decision") and episode_manifest["completed_chunks"]>0,
                 "prefix episode has an unresolved decision")
        refs=episode_manifest["chunks"]
        _require(len(refs)==episode_manifest["completed_chunks"],"episode manifest chunk count differs")
        _require(len(list((episode_dir/"chunks").glob("*.json")))==len(refs),"unlisted episode decision")
        row=episode_rows[item["episode_row_index"]]
        _require(row["episode"]==index+1 and row["seed"]==trial["seeds"][index]
                 and row["method"]==method and row["phase"]=="source" and row["task"]==trial["task"]
                 and not row.get("budget_truncated",False),"completed episode row differs")
        steps=0;reward=0.;previous_frame=reset["initial_observation"]["sha256"]
        for chunk_index,ref in enumerate(refs):
            record=_json(ref,root=episode_dir)
            _require(record["status"]=="complete" and record["chunk_index"]==chunk_index+1,"unfinished or reordered decision")
            _require(record["context"]["trial_id"]==trial["trial_id"] and record["context"]["method"]==method,
                     "decision trial identity differs")
            _require(record["observation"]["sha256"]==previous_frame,"observation chain differs")
            arrays=_arrays(record["decision_arrays"],root);native=_arrays(record["native_result"]["arrays"],root)
            observation=_arrays(record["observation"],root);_arrays(record["next_observation"],root)
            previous_frame=record["next_observation"]["sha256"]
            learning=record["learning_result"];count=record["native_result"]["executed_primitive_steps"]
            _require(not learning["budget_truncated"] and 0<count<=record["issued_primitive_steps"]
                     and int(native["valid_action_mask"].sum())==count,
                     "external cut or inconsistent primitive-action count")
            if chunk_index==0:
                feature_hash=hashlib.sha256(np.ascontiguousarray(observation["features"],dtype=np.float32).tobytes()).hexdigest()
                _require(feature_hash==row["initial_observation_sha256"],"episode initial public features differ")
            steps+=count;reward+=learning["reward"]
            done=bool(learning["terminated"] or learning["truncated"])
            _require(done==(chunk_index==len(refs)-1),"prefix is not a closed episode boundary")
            _require(learning["episode_primitive_steps"]==steps,"episode primitive clock differs")
            metadata=record["decision_metadata"]
            _require(metadata["episode_index"]==index and metadata["chunk_index"]==chunk_index,
                     "recorded episode/chunk clock differs")
            chunks.append((record,arrays,observation,index,prefix_chunk_count+chunk_index))
        _require(steps==row["primitive_steps"]==episode_manifest["executed_primitive_steps"]
                 and len(refs)==row["chunks"] and reward==row["return_"],"episode summary differs")
        prefix_steps+=steps;prefix_chunk_count+=len(refs);rows.append(deepcopy(row))
    # A selected zero-action task must not have any raw episode; a completed
    # prefix must list every episode of that task, including empty resets.
    actual_dirs=[]
    for reset_path in (root/"episodes").glob("*/reset.json"):
        context=json.loads(reset_path.read_text()).get("context",{})
        if context.get("phase")=="source" and context.get("method")==method and context.get("trial_id")==trial["trial_id"]:
            actual_dirs.append(reset_path.parent.resolve())
    _require(set(actual_dirs)==set(expected_dirs),"selected trial contains omitted or unclosed episodes")
    _require(learner["primitive_steps"]==saved["source_steps"]+(prefix_steps if method=="sac" else 0),
             "learner retained-prefix action count differs")
    prefix_rollout=None
    if method=="sac" and chunks:
        metadata=chunks[-1][0]["decision_metadata"]
        config=learner["contract"]["config"]
        ready=(learner["environment_steps"]>config["warmup_chunks"] and len(learner["replay"])>=config["batch_size"])
        _require(metadata["source_warmup"]==(metadata["history_index"]<config["warmup_chunks"])
                 and learner["pending_chunks"]==0
                 and learner["updates"]==metadata["optimizer_updates"]+(config["utd"] if ready else 0)
                 and learner["environment_steps"]==metadata["history_index"]+1,"last SAC transition/SGD is not fully retained")
    if method=="ppo_reset" and chunks:
        prefix_rollout=_reset_prefix_rollout(chunks,saved,contract)
    receipt.update(selected_trial=deepcopy(trial),completed_prefix_episodes=len(rows),
                   prefix_primitive_steps=prefix_steps,prefix_chunks=prefix_chunk_count)
    return TrialContinuation(method,sha256,receipt,deepcopy(trial),rows,prefix_rollout,prefix_steps,prefix_chunk_count)


def _reset_prefix_rollout(chunks,saved,contract):
    records={key:[] for key in ("observations","raw_actions","log_probs","values","rewards",
                                "trial_ends","attempt_ends","resets","episode_limits")}
    pending=saved.get("pending_rollouts",[])
    reference=pending[0][0] if pending else None
    action_dim=int(np.prod(contract["noise_shape"]));feature_dim=contract["observation_dim"]
    for record,arrays,observation,episode,history_index in chunks:
        metadata=record["decision_metadata"];learning=record["learning_result"]
        def array(name):return arrays[metadata[name]["numeric_array"]]
        inputs=array("policy_input");raw=array("pre_squash_actions")
        _require(inputs.shape==(feature_dim+action_dim+4,) and raw.shape==(action_dim,),"PPO public-input/action shape differs")
        _require(metadata["history_index"]==history_index and metadata["policy_update"]==saved["learner"]["updates"],
                 "PPO history/policy version differs")
        _require(metadata["reset_episode_feedback"] is True and metadata["reward_blind"] is False,"PPO prefix reset convention differs")
        _require(np.array_equal(inputs[:feature_dim],observation["features"]),"PPO public features differ")
        _require(np.allclose(np.asarray(arrays["sampled_noise"]).reshape(-1),
                            np.tanh(raw)*contract["action_limit"],atol=2e-6,rtol=2e-6),"PPO sampled action differs")
        reset=bool(array("recurrent_reset_mask").reshape(-1)[0])
        _require(reset==(metadata["chunk_index"]==0),"PPO episode reset mask differs")
        if reset:_require(np.all(inputs[feature_dim:feature_dim+action_dim+2]==0),"PPO reset retains prior noise/reward")
        _require(np.isclose(inputs[-2],episode/max(contract["train_episodes"]-1,1)),"PPO episode clock differs")
        values=dict(observations=torch.as_tensor(inputs.copy()),raw_actions=torch.as_tensor(raw.copy()),
            log_probs=torch.tensor(float(array("old_log_prob").reshape(-1)[0]),dtype=torch.float32),
            values=torch.tensor(float(array("old_value").reshape(-1)[0]),dtype=torch.float32),
            rewards=torch.tensor(float(learning["reward"]),dtype=torch.float32),trial_ends=torch.tensor(False),
            attempt_ends=torch.tensor(bool(learning["terminated"] or learning["truncated"])),
            resets=torch.tensor(reset),episode_limits=torch.tensor(metadata["episode_limit"],dtype=torch.int64))
        for key,value in values.items():records[key].append(value)
    rollout={key:torch.stack(values)[:,None] for key,values in records.items()}
    rollout["action_mask"]=torch.ones_like(rollout["trial_ends"],dtype=torch.bool)
    if reference is not None:
        _require(set(rollout)==set(reference),"PPO prefix/pending rollout fields differ")
        for key,value in rollout.items():
            _require(value.shape[1:]==reference[key].shape[1:] and value.dtype==reference[key].dtype,
                     "PPO prefix/pending rollout shape or dtype differs")
    return rollout


def combine_episode_rollouts(prefix,suffix):
    """Join genuine reset-controller episodes; only original final episode ends the trial."""
    _require(set(prefix)==set(suffix) and "bootstrap" not in prefix,"rollout assembly fields differ")
    _require(not prefix["trial_ends"].any() and prefix["attempt_ends"][-1].all(),"prefix does not end at an episode boundary")
    _require(not suffix["trial_ends"][:-1].any() and suffix["trial_ends"][-1].all(),"suffix does not finish the original trial")
    joined={}
    for key,value in prefix.items():
        _require(value.shape[1:]==suffix[key].shape[1:] and value.dtype==suffix[key].dtype,"rollout tensor shape/dtype differs")
        joined[key]=torch.cat((value,suffix[key]),dim=0)
    return joined


def write_boundary_manifest(path,*,checkpoint_path,checkpoint,corpus_root,progress_path,
                            episodes_path,ledger_snapshot_path,execution_commit,lineage_commit):
    """Publish the next hash-bound handoff before a native environment closes."""
    method,=checkpoint["contract"]["methods"]
    saved=checkpoint["states"][method];learner=saved["learner"]
    _require(saved["schedule"]["trials"]==saved["completed_source_trials"],"cannot publish an unfinished selected trial")
    _require(learner["primitive_steps"]==saved["source_steps"],"cannot publish uncredited source actions")
    episodes_path=Path(episodes_path);episodes_path.touch(exist_ok=True)
    def bind(item):
        item=Path(item).resolve()
        return dict(path=str(item),sha256=file_sha256(item))
    counters={key:learner[key] for key in ("primitive_steps","updates","optimizer_steps","chunks",
        "environment_steps","replay_position","pending_chunks","update_seconds","target_replay_mode",
        "source_replay_rows_at_target_start") if key in learner}
    value=dict(schema=SCHEMA,kind="whole_trial_boundary",method=method,
        execution_source_commit=execution_commit,lineage_source_commit=lineage_commit,
        checkpoint=bind(checkpoint_path),corpus_root=str(Path(corpus_root).resolve()),
        corpus_contract=bind(Path(corpus_root)/"contract.json"),progress=bind(progress_path),
        episodes_file=bind(episodes_path),ledger_snapshot=bind(ledger_snapshot_path),trial=None,prefix=[],
        expected=dict(source_steps=saved["source_steps"],completed_source_trials=saved["completed_source_trials"],
            completed_source_iterations=saved["completed_source_iterations"],schedule_digest=artifact_digest(saved["schedule"]),
            learner_counters=counters,pending_rollout_digests=[artifact_digest(r) for r,_ in saved["pending_rollouts"]]))
    path=Path(path);temporary=path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+"\n");temporary.replace(path)
    digest=file_sha256(path)
    path.with_suffix(".sha256").write_text(digest+"\n")
    return dict(path=str(path.resolve()),sha256=digest)
