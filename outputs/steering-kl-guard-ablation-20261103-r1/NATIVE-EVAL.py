#!/usr/bin/env python3
"""Paired native evaluation of copied first-update KL candidates; no SGD."""
import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import traceback


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()


def write(path, data):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    q = p.with_name(p.name + f".tmp-{os.getpid()}")
    q.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    q.replace(p)


def cases():
    tasks = ("OpenDrawer", "PickPlaceCounterToCabinet", "PickPlaceDrawerToCounter",
             "TurnOffStove", "CoffeeSetupMug")
    return [dict(bank="kl_guard_diagnostic", task=task, split="target",
        native_seed=8_900_000 + task_index * 100_000 + slot, reset_ordinal=0,
        slot=slot, counted=True, layout_source="native_target_constructor_seed_first_reset")
        for slot in range(50) for task_index, task in enumerate(tasks)]


def parse():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--profile", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--deadline-unix", type=float, required=True)
    p.add_argument("--worker-rank", type=int)
    p.add_argument("--validate-only", action="store_true")
    return p.parse_args()


def inputs(profile):
    import torch
    manifest = json.loads(Path(profile["CPU_result"]).read_text())
    if (not manifest["completed"] or not manifest["source_files_unchanged"]
            or sha(profile["CPU_result"]) != profile["CPU_result_sha256"]):
        raise ValueError("Require the complete hash-bound copied-candidate CPU comparison")
    result = {}
    for arm in ("velocity", "kv_all_layers"):
        record = manifest["arms"][arm]
        names = {"initial": record["initial_artifact"], **{
            name: record["variants"][name]["artifact"] for name in ("guarded1", "unguarded1")}}
        result[arm] = {}
        for name, ref in names.items():
            if sha(ref["path"]) != ref["sha256"]:
                raise ValueError("Candidate weights changed")
            value = torch.load(ref["path"], map_location="cpu", weights_only=False)
            if value["arm"] != arm or value["candidate"] != name or value["source_policy_version"] != 0:
                raise ValueError("Wrong counterfactual source or interface")
            result[arm][name] = dict(reference=ref, state=value)
    return result


def child(args, profile):
    import torch
    from fluffy_bassoon.meta_rl.joint_steering_ppo import JointActionJournal, summarize_episodes
    from fluffy_bassoon.meta_rl.joint_steering_ppo_worker import (
        _RecordingEnvironment, _load_native, _save, _versions, collect_bank, native_video_writer)
    from fluffy_bassoon.meta_rl.residual_ppo import ResidualActorCritic, ResidualPPOConfig
    from fluffy_bassoon.meta_rl.rlinf_native_ppo import tensor_hash
    rank = args.worker_rank
    arm, shard = ("velocity" if rank < 2 else "kv_all_layers"), rank % 2
    root = args.output / "workers" / f"rank-{rank:02d}"
    root.mkdir(parents=True, exist_ok=False)
    candidates = inputs(profile)[arm]
    initial = candidates["initial"]["state"]
    evaluation_config = initial["policy_config"]
    motor = None
    row = dict(completed=False, rank=rank, arm=arm, shard=shard, started_unix=time.time(), policies={})
    try:
        request = dict(model_directory=profile["model_directory"], norm_path=profile["norm_path"],
            source_commit=profile["source_commit"], runtime_commit=profile["runtime_commit"],
            policy_config=evaluation_config, spec=initial["interface_spec"], lanes=48,
            synchronize_initial_lanes=True)
        motor, factory, signature = _load_native(request, Path(profile["code"]))
        if any(signature[k] != profile[k] for k in ("model_sha256", "model_config_sha256", "norm_sha256")):
            raise ValueError("Original frozen native motor fingerprint differs")
        row["native_signature"] = signature
        frozen = signature["parameter_versions_sha256"]
        for sequence, name in enumerate(("initial", "guarded1", "unguarded1"), 1):
            remaining_cases = (4 - sequence) * 125
            # Use the previous complete bank to forecast the remaining work.
            if sequence > 1:
                duration = row["policies"]["initial"]["elapsed_seconds"]
                if time.time() + duration * remaining_cases / 125 + 90 > args.deadline_unix:
                    raise TimeoutError("Measured complete-bank forecast exceeds finite evaluation ceiling")
            directory = root / name
            selected = candidates[name]["state"]
            if selected["interface_spec"] != initial["interface_spec"]:
                raise ValueError("Candidates use different steering maps")
            policy = ResidualActorCritic(ResidualPPOConfig(**evaluation_config)).to("cuda").eval()
            policy.load_state_dict(selected["policy"], strict=True)
            model_hash = tensor_hash(policy.state_dict())
            request.update(sequence=sequence, phase="test_before" if name == "initial" else "test_after",
                iteration=0 if name == "initial" else 1, policy_iteration=0 if name == "initial" else 1,
                cases=cases()[shard::2], deterministic=True, zero_steering=name == "initial",
                allocation_id=f"{socket.gethostname()}-{os.getpid()}", native_cuda_uuid=signature["cuda_uuid"],
                diagnostic_candidate=name, training_config=selected["policy_config"],
                candidate_reference=candidates[name]["reference"])
            _save(root / f"request-{sequence:06d}.pt", dict(request, policy=policy.state_dict()))
            journal = JointActionJournal(args.output / "accounting", rank=rank,
                category="eval", scope=f"kl-diagnostic-{arm}-{name}")
            environments = [_RecordingEnvironment(factory, request["cases"], horizon=600,
                output=directory / "videos" / f"lane-{lane:02d}",
                monitoring_root=directory / "monitoring" / f"lane-{lane:02d}",
                journal=journal, writer_factory=native_video_writer, rank=rank,
                episode_id_prefix=f"{arm}-{name}-{shard}-lane{lane:02d}-",
                canonical_root=args.output / "canonical" / arm,
                source=dict(rank=rank, sequence=sequence, phase=request["phase"], iteration=request["iteration"],
                    source_commit=profile["source_commit"], runtime_commit=profile["runtime_commit"],
                    diagnostic_candidate=name)) for lane in range(48)]
            begin = time.time()
            collected = collect_bank(policy, motor, environments, request=request,
                output=directory / "history", rank=rank, deadline_unix=args.deadline_unix,
                qualification_path=root / "qualification.json" if sequence == 1 else None)
            if (tensor_hash(policy.state_dict()) != model_hash or motor.model.training
                    or any(p.requires_grad for p in motor.model.parameters())
                    or tensor_hash(_versions(motor.model)) != frozen):
                raise ValueError("Evaluation changed candidate weights or the frozen native motor")
            if len(collected["episodes"]) != 125 or any(e["horizon"] != 600 or not e["completed"] for e in collected["episodes"]):
                raise ValueError("Require all125 full600-control episodes in each case shard")
            ref = _save(root / f"result-{sequence:06d}.pt", dict(collected,
                candidate=name, native_signature=signature, candidate_reference=candidates[name]["reference"]))
            summary = summarize_episodes(collected["episodes"])
            row["policies"][name] = dict(elapsed_seconds=time.time()-begin, summary=summary,
                result_reference=ref, candidate_weights_unchanged=True,
                actual_policy_controls=125*600)
            write(root / "PROGRESS.json", row)
            print(json.dumps(dict(rank=rank, arm=arm, candidate=name, summary=summary,
                elapsed_seconds=time.time()-begin)), flush=True)
        row.update(completed=True, finished_unix=time.time())
        write(root / "RESULT.json", row)
    except BaseException:
        row.update(error=traceback.format_exc(), finished_unix=time.time())
        write(root / "FAILURE.json", row)
        raise


def main():
    args = parse()
    profile = json.loads(args.profile.read_text())
    sys.path.insert(0, str(Path(profile["code"]) / "src"))
    import torch
    from fluffy_bassoon.meta_rl.rlinf_native_ppo import checked_checkout
    checked_checkout(Path(profile["code"]), profile["source_commit"])
    checked_checkout(Path(profile["runtime_path"]), profile["runtime_commit"])
    if sys.platform != "linux":
        raise RuntimeError("All computation belongs on allocated Linux")
    candidates = inputs(profile)
    if args.validate_only:
        assert len(cases()) == 250 and len({(x["task"],x["native_seed"]) for x in cases()}) == 250
        for arm, variants in candidates.items():
            from fluffy_bassoon.meta_rl.residual_ppo import ResidualActorCritic, ResidualPPOConfig
            for item in variants.values():
                cfg=ResidualPPOConfig(**item["state"]["policy_config"])
                policy=ResidualActorCritic(cfg);policy.load_state_dict(item["state"]["policy"],strict=True)
        print(json.dumps(dict(completed=True, candidates=6, cases=250, CPU_only=True)),flush=True)
        return
    if args.worker_rank is not None:
        child(args, profile)
        return
    from fluffy_bassoon.baige_gpu import mapping
    devices = mapping()
    if len(devices) != 4 or torch.cuda.device_count() != 4:
        raise RuntimeError("Require exactly four attested allocated GPUs")
    if not time.time() < args.deadline_unix <= profile["process_started_unix"] + 5280:
        raise ValueError("Leave120seconds cleanup within the6GPU-hour four-GPU ceiling")
    args.output.mkdir(parents=True, exist_ok=True)
    children, streams = [], []
    try:
        for rank, device in enumerate(devices):
            log = (args.output / f"worker-{rank:02d}-console.log").open("x")
            streams.append(log)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=device["visibility_token"],
                LYCO_BAIGE_ISOLATED_UUID=device["uuid"], MUJOCO_EGL_DEVICE_ID=str(device["allocation_ordinal"]),
                ROBOCASA_NATIVE_OUTPUT=str(args.output / f"native-runtime-{rank:02d}"),
                OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
            children.append(subprocess.Popen([sys.executable,"-B",__file__,"--profile",str(args.profile),
                "--output",str(args.output),"--deadline-unix",str(args.deadline_unix),"--worker-rank",str(rank)],
                env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True))
        while any(p.poll() is None for p in children):
            if any(p.poll() not in (None,0) for p in children):
                raise RuntimeError("An evaluation worker failed; retain all sibling completed evidence")
            if time.time() >= args.deadline_unix:
                raise TimeoutError("Finite native evaluation deadline")
            time.sleep(2)
        if any(p.returncode != 0 for p in children):
            raise RuntimeError("Evaluation worker exit failed")
        import numpy as np
        from scipy.stats import binomtest
        from fluffy_bassoon.meta_rl.joint_steering_ppo import TASKS, summarize_episodes
        combined = dict(completed=True, arms={}, scientific_source_updates=0,
            new_training_episodes=0, finished_unix=time.time())
        for arm, ranks in (("velocity",(0,1)),("kv_all_layers",(2,3))):
            banks = {}
            for sequence,name in enumerate(("initial","guarded1","unguarded1"),1):
                episodes=[]
                for rank in ranks:
                    result=torch.load(args.output/"workers"/f"rank-{rank:02d}"/f"result-{sequence:06d}.pt",map_location="cpu",weights_only=False)
                    episodes.extend(result["episodes"])
                banks[name]={(e["case"]["task"],e["case"]["native_seed"]):e for e in episodes}
                if len(banks[name])!=250:raise ValueError("Final panel is incomplete")
            comparisons={}
            for left,right in (("initial","guarded1"),("initial","unguarded1"),("guarded1","unguarded1")):
                if set(banks[left])!=set(banks[right]):raise ValueError("Paired diagnostic cases differ")
                metrics={}
                for metric in ("success_once","success_at_end"):
                    by_task={}
                    rng=np.random.default_rng(921120); boot=np.zeros(10000)
                    for task in TASKS:
                        keys=[key for key in banks[left] if key[0]==task]
                        delta=np.array([int(bool(banks[right][k][metric]))-int(bool(banks[left][k][metric])) for k in keys])
                        wins,losses=int((delta>0).sum()),int((delta<0).sum())
                        by_task[task]=dict(cases=len(keys),wins=wins,losses=losses,gain=float(delta.mean()))
                        boot+=delta[rng.integers(len(delta),size=(len(boot),len(delta)))].mean(1)/len(TASKS)
                    wins=sum(r["wins"] for r in by_task.values());losses=sum(r["losses"] for r in by_task.values())
                    metrics[metric]=dict(per_task=by_task,wins=wins,losses=losses,gain=(wins-losses)/250,
                        exact_p=float(binomtest(wins,wins+losses,.5).pvalue) if wins+losses else 1.,
                        paired_task_stratified_bootstrap_95ci=np.quantile(boot,[.025,.975]).tolist())
                comparisons[left+"_to_"+right]=metrics
            combined["arms"][arm]=dict(summaries={k:summarize_episodes(list(v.values())) for k,v in banks.items()},
                paired_comparisons=comparisons, interpretation="Exploratory first-update diagnostic, no convergence or task-generalization claim")
        escrow=json.loads((args.output/"accounting/escrow.json").read_text())
        if escrow["pending"]:raise ValueError("Completed evaluation left unresolved physical calls")
        combined["accounting"]=escrow
        write(args.output/"RESULT.json",combined)
    finally:
        for p in children:
            if p.poll() is None:os.killpg(p.pid,signal.SIGTERM)
        for p in children:
            try:p.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(p.pid,signal.SIGKILL);p.wait()
        for log in streams:log.close()
        write(args.output/"RELEASED.json",dict(child_pids=[p.pid for p in children],
            child_exit_codes=[p.returncode for p in children],all_children_released=all(p.poll() is not None for p in children),
            finished_unix=time.time()))


if __name__ == "__main__":
    main()
