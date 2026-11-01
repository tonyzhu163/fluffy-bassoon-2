"""Allocated CPU-only isolation of an existing OfficialSAC recovery guard."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import pickle
import random
import socket
import subprocess
import sys
import time

def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()

def read_bound(ref):
    path=Path(ref["path"])
    if not path.is_absolute() or path.is_symlink() or sha(path)!=ref["sha256"]:
        raise ValueError("Exact regular SHA-bound input required")
    return path

def guard_predicates(saved_training, progress_training, updates, replay_rows, restored_signature, model):
    return dict(training_equal=saved_training==progress_training,
        updates_equal=updates==model["updates"], replay_rows_equal=replay_rows==model["replay_rows"],
        raw_restored_signature_equal=restored_signature==model["signature"])

def mapping_order_changes(expected, actual, path="state"):
    rows=[]
    if isinstance(expected,dict) and isinstance(actual,dict):
        if set(expected)==set(actual) and list(expected)!=list(actual):
            rows.append(dict(path=path, expected_keys=list(expected), actual_keys=list(actual)))
        for key in sorted(set(expected)&set(actual),key=str):
            rows.extend(mapping_order_changes(expected[key],actual[key],path+"/"+str(key)))
    elif isinstance(expected,(list,tuple)) and isinstance(actual,(list,tuple)):
        for index,(a,b) in enumerate(zip(expected,actual)):
            rows.extend(mapping_order_changes(a,b,path+"/"+str(index)))
    return rows

def classify(predicates, legacy_saved_signature_matches, canonical_equal, rng_equal):
    if not all(predicates[k] for k in ("training_equal","updates_equal","replay_rows_equal")):
        return "cursor_or_counter_mismatch"
    if not legacy_saved_signature_matches:
        return "stored_agent_bytes_do_not_match_original_cursor_signature"
    if not canonical_equal or not rng_equal:
        return "restored_state_or_rng_mismatch"
    return "all_checks_match" if predicates["raw_restored_signature_equal"] else "identical_saved_state_raw_representation_differs"

def main(argv=None):
    parser=argparse.ArgumentParser()
    for name in ("profile","ticket","output"):
        parser.add_argument("--"+name,type=Path,required=True)
    parser.add_argument("--profile-sha256",required=True)
    args=parser.parse_args(argv)
    if sys.platform!="linux" or not socket.gethostname().startswith("job-"):
        raise RuntimeError("Actual allocated Linux CPU only")
    if sha(args.profile)!=args.profile_sha256:
        raise ValueError("Profile SHA differs")
    profile=json.loads(args.profile.read_text());ticket=json.loads(args.ticket.read_text())
    if (ticket["actual_GPU_count"]!=0 or ticket["actual_CPU_cores"]!=8
            or ticket["actual_memory_bytes"]!=64*1024**3 or ticket["cuda_initialized"]
            or ticket["source_commit"]!=profile["inspector_source_commit"]):
        raise ValueError("Actual zeroGPU 8CPU/64GiB ticket required")
    deadline=ticket["absolute_deadline_unix"]
    if time.time()>=deadline:
        raise TimeoutError("Original CPU deadline")
    paths={k:read_bound(profile[k]) for k in ("checkpoint","cursor","contract","vla_libero","motor_rng_helper")}
    before={k:sha(v) for k,v in paths.items()}
    cursor=json.loads(paths["cursor"].read_text());contract=json.loads(paths["contract"].read_text())
    model=cursor["model"]
    if (model["sha256"]!=profile["checkpoint"]["sha256"] or model["updates"]!=301700
            or model["replay_rows"]!=15579 or cursor["training"]["episodes"]!=1720):
        raise ValueError("Exact closed301700 cursor required, no selected substitute")
    source_root=Path(profile["qualified_code_worktree"])
    source=subprocess.check_output(["git","rev-parse","HEAD"],cwd=source_root,text=True).strip()
    if source!="b6655659d8bff2c6b5d14eaf05d63f4647176d4f":
        raise ValueError("Exact original learner implementation required")
    sys.path.insert(0,str(source_root/"src"))
    os.environ["JAX_PLATFORMS"]="cpu"
    import numpy as np
    import jax
    from flax import serialization
    from fluffy_bassoon.meta_rl.vla_libero import OfficialSAC,state_differences,structure_digest
    from fluffy_bassoon.meta_rl.vla_source40_corpus import save_motor_rng,restore_motor_rng
    if any(device.platform!="cpu" for device in jax.devices()):
        raise ValueError("JAX CPU backend only")
    import fluffy_bassoon.meta_rl.vla_libero as implementation
    if Path(implementation.__file__).resolve()!=paths["vla_libero"].resolve():
        raise ValueError("Loaded learner module differs")
    with paths["checkpoint"].open("rb") as stream:
        payload=pickle.load(stream)
    saved_tree=serialization.msgpack_restore(payload["agent"])
    saved_signature=hashlib.sha256(payload["agent"]).hexdigest()
    learner=OfficialSAC(Path(profile["dsrl_root"]),0)
    sampler=np.random.default_rng(0)
    progress=learner.restore(paths["checkpoint"],sampler,contract)
    # Capture restoration before any diagnostic operation can draw a sample.
    restored_numpy=np.random.get_state();restored_python=random.getstate()
    restored_agent_rng=np.asarray(jax.device_get(learner.agent._rng))
    restored_sampler=sampler.bit_generator.state
    actual_tree=serialization.to_state_dict(learner.agent._save_dict)
    restored_bytes=serialization.to_bytes(learner.agent._save_dict)
    restored_signature=hashlib.sha256(restored_bytes).hexdigest()
    differences=state_differences(saved_tree,actual_tree,rtol=0.,atol=0.)
    canonical_equal=not differences
    components={}
    for key in sorted(saved_tree):
        component_differences=state_differences(saved_tree[key],actual_tree[key],path=str(key),rtol=0.,atol=0.)
        components[key]=dict(exact_equal=not component_differences,
            expected_sha256=structure_digest(saved_tree[key]),actual_sha256=structure_digest(actual_tree[key]),
            differences=len(component_differences))
    replay_expected=structure_digest(payload["replay"]);replay_actual=structure_digest(learner.replay)
    motor=progress["motor_rng"]
    motor_roundtrip=save_motor_rng(restore_motor_rng(motor))
    rng=dict(agent_exact=np.array_equal(payload["agent_rng"],restored_agent_rng)
            and np.asarray(payload["agent_rng"]).dtype==restored_agent_rng.dtype,
        sampler_exact=structure_digest(payload["sampler"])==structure_digest(restored_sampler),
        numpy_global_exact=structure_digest(payload["numpy_rng"])==structure_digest(restored_numpy),
        python_global_exact=payload["python_rng"]==restored_python,
        motor_key_roundtrip_exact=structure_digest(motor)==structure_digest(motor_roundtrip))
    predicates=guard_predicates(progress["training"],cursor["training"],learner.updates,len(learner.replay),restored_signature,model)
    after={k:sha(v) for k,v in paths.items()}
    result=dict(format="libero_headroom_resume_guard_CPU_diagnosis_v1",
        source_commit=profile["inspector_source_commit"],qualified_learner_source=source,
        checkpoint=profile["checkpoint"],cursor=profile["cursor"],original_guard_predicates=predicates,
        original_guard_would_pass=all(predicates.values()),legacy_cursor_signature=model["signature"],
        saved_agent_bytes_SHA=saved_signature,legacy_saved_signature_matches=saved_signature==model["signature"],
        restored_raw_serialization_SHA=restored_signature,
        canonical_expected_SHA=structure_digest(saved_tree),canonical_actual_SHA=structure_digest(actual_tree),
        canonical_full_state_exact=canonical_equal,canonical_components=components,
        exact_differences=[{k:v for k,v in r.items() if k not in ("expected_preview","actual_preview")} for r in differences[:64]],
        exact_difference_count=len(differences),mapping_order_changes=mapping_order_changes(saved_tree,actual_tree),
        RNG=rng,replay=dict(rows=len(learner.replay),expected_SHA=replay_expected,actual_SHA=replay_actual,
            exact=replay_expected==replay_actual),updates=learner.updates,
        training_episodes=progress["training"]["episodes"],
        input_SHA_before=before,input_SHA_after=after,all_inputs_unchanged=before==after,
        classification=classify(predicates,saved_signature==model["signature"],
            canonical_equal and replay_expected==replay_actual,all(rng.values())),
        actual_JAX_devices=[str(device) for device in jax.devices()],VLA_constructed=False,
        simulator_constructed=False,new_SGD=0,new_robot_actions=0,
        scope="Only original small pixelSAC CPU initialization/restore/exactstate+RNG comparison; no actor sample/update/VLA constructor/weights/simulator")
    if time.time()>=deadline:
        raise TimeoutError("Original CPU deadline; result not admitted")
    if before!=after:
        raise RuntimeError("Input changed during read-only inspection")
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    print(json.dumps(dict(classification=result["classification"],predicates=predicates,
        canonical_equal=canonical_equal,replay_exact=result["replay"]["exact"],RNG=rng)),flush=True)

if __name__=="__main__":
    main()
