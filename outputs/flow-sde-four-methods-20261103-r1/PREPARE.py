"""Allocated Linux GPU source preparation and finite native qualification.

No new PPO, no converged-student claim. Every completed offline shard is
hash-bound and reusable without a second teacher query. Numerical fit probes
are separate from later scientific student fits.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
    tmp.replace(path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def interrupted(signum, frame):
    raise TimeoutError("Finite preparation deadline; retain all completed shards and partial native evidence")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--code", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-hostname", required=True)
    parser.add_argument("--max-seconds", type=int, default=2400)
    args = parser.parse_args()
    out, code = args.output, args.code
    out.mkdir(parents=True, exist_ok=False)
    started = time.time()
    result = dict(completed=False, source_commit=args.source_commit, hostname=socket.gethostname(),
        started_unix=started, PPO_updates=0, source_student_fits=0, native_controls=0)
    def stage(name, **extra):
        write(out / "PROGRESS.json", dict(stage=name, checked_unix=time.time(), elapsed=time.time()-started, **extra))
        print(json.dumps(dict(stage=name, elapsed=time.time()-started, **extra)), flush=True)
    try:
        if (sys.platform != "linux" or socket.gethostname() != args.expected_hostname
                or not 60 <= args.max_seconds <= 2400):
            raise ValueError("Require exact allocated Linux hostname and finite <=2400s preparation")
        for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGALRM):
            signal.signal(sig, interrupted)
        signal.setitimer(signal.ITIMER_REAL, args.max_seconds)
        root = Path("/mowenjie/fluffy-bassoon-2")
        audit = root / "outputs/flow-four-method-source-audit-20261103-r1"
        raw_path = audit / "RAW-PUBLIC-CORPUS.npz"
        if sha(raw_path) != "5bf7c71056ca84b37f1d23589696b834c1a3401bbbe61e75f0b6eb5249498eb6":
            raise ValueError("Audited source corpus changed")
        if os.statvfs(root).f_bavail * os.statvfs(root).f_frsize < 850 * 2**30:
            raise ValueError("Protect remaining steering-history reservation before source preparation")
        runtime = root / "sources/robocasa365-public-pi05-nnx-leaves-20261101-r1/RLinf"
        model = root / "inputs/robocasa365-public-pi05-c484-torch-20261101-r2"
        norm = root / "inputs/robocasa365-public-pi05-c484-75000-20261101-r1/assets/norm_stats.json"
        teacher = root / "outputs/robocasa-opendrawer-flow-sde-20261102-r4/opendrawer_flow_sde/checkpoints/global_step_18/actor/model_state_dict"
        teacher_file = teacher / "full_weights.pt"
        if sha(teacher_file) != "7e65e25408ed653d4c635a7f23b788ac621e8ff9909e866cec88173f519eff61":
            raise ValueError("Pinned final Flow checkpoint18 changed")
        os.environ.update(RLINF_PATH=str(runtime), ROBOCASA_NATIVE_OUTPUT=str(out / "native"),
            ROBOCASA_PI05_TORCH_MODEL=str(model), ROBOCASA_PI05_NATIVE_NORM=str(norm))
        sys.path[:0] = [str(code / "src"), str(runtime)]
        from fluffy_bassoon.meta_rl.rlinf_native_ppo import checked_checkout
        checked_checkout(code, args.source_commit)
        checked_checkout(runtime, "8a1233c3fae6f5905adfda7684c455842b0ecfb8")
        stage("allocated_gpu_probe")
        probe_path = out / "GPU-PROBE.json"
        subprocess.run([sys.executable, "-B", str(code / "scripts/[train]gpu_probe.py"),
            "--expected-count", "1", "--output", str(probe_path), "--baige-container", "--purpose", "training",
            "--baige-exclusive-training"], check=True, timeout=100)
        probe = json.loads(probe_path.read_text())
        if probe["status"] != "passed" or probe["hostname"] != args.expected_hostname:
            raise ValueError("Actual allocation did not pass GPU qualification")
        device = probe["devices"][0]
        os.environ.update(LYCO_BAIGE_GPU_MAP=str(probe_path), LYCO_BAIGE_ISOLATED_UUID=device["uuid"],
            MUJOCO_EGL_DEVICE_ID=str(device["allocation_ordinal"]))
        import numpy as np
        import torch
        from hydra import compose, initialize_config_dir
        from rlinf.models.embodiment.openpi import get_model
        from fluffy_bassoon.meta_rl.flow_history import (DirectActionCodec, FlowDirectMotor, export_corpus, teacher_noise)
        from fluffy_bassoon.meta_rl.joint_steering_ppo_worker import _load_native, _versions, _save, native_video_writer
        from fluffy_bassoon.meta_rl.rlinf_native_ppo import tensor_hash
        from fluffy_bassoon.meta_rl.robocasa_interface import FrozenRoboCasaInterface
        from fluffy_bassoon.meta_rl.robocasa_residual import public_observation, combine_observations
        from types import SimpleNamespace
        torch.set_num_threads(2)
        request = dict(model_directory=str(model), norm_path=str(norm), source_commit=args.source_commit,
            runtime_commit="8a1233c3fae6f5905adfda7684c455842b0ecfb8")
        stage("load_original_frozen_VLM_and_final18_teacher")
        native, factory, signature = _load_native(request, code)
        result["initializer"] = signature
        if (signature["model_sha256"] != "661bf172a2ae9dfca75216697eeaa543408bcbdfa53c98132759b93e0e8993a5"
                or signature["norm_sha256"] != "4aed1af411bd0e0f49d2b0e6d6832b11b8682917231a07173fbc30fd493bbdee"):
            raise ValueError("Original frozen feature initializer differs")
        with initialize_config_dir(config_dir=str(code / "configs/native_ppo"), version_base=None):
            cfg = compose(config_name="robocasa365_pi05_eval")
        cfg.rollout.model.model_path = str(teacher)
        cfg.rollout.model.openpi_data.norm_stats_path = str(norm)
        final = get_model(cfg.rollout.model, torch_dtype=torch.bfloat16).to("cuda").eval().requires_grad_(False)
        stats = json.loads(norm.read_text()); stats = stats.get("norm_stats", stats)
        final_port = FrozenRoboCasaInterface(SimpleNamespace(model=final, stats=stats))
        codec = DirectActionCodec.from_stats(stats)
        motor = FlowDirectMotor(native, codec)
        frozen_versions = tensor_hash(_versions(native.model))
        references = json.loads((audit / "SOURCE-CHUNKS.json").read_text())
        with np.load(raw_path, allow_pickle=False) as source:
            raw = {name: source[name] for name in source.files}
        mapping = {(int(ref), int(row)): (s, t) for s in range(128) for t, (ref, row)
                   in enumerate(raw["raw_chunk_references"][s])}
        features = np.zeros((128, 144, 2073), np.float32)
        labels = np.zeros((128, 144, 50, 12), np.float32)
        done = np.zeros((128, 144), bool)
        def read_reference(item):
            index, reference = item
            data = Path(reference["path"]).read_bytes()
            if hashlib.sha256(data).hexdigest() != reference["sha256"]:
                raise ValueError("A retained physical source chunk changed")
            payload = torch.load(io.BytesIO(gzip.decompress(data)), map_location="cpu", weights_only=False)
            return index, reference, public_observation(payload["current_observation"])
        shards = []
        pending, rows = [], 0
        groups = []
        for index, reference in enumerate(references):
            if rows + reference["rows"] > 32:
                groups.append(pending); pending, rows = [], 0
            pending.append((index, reference)); rows += reference["rows"]
        if pending: groups.append(pending)
        extraction_started = time.time()
        with ThreadPoolExecutor(max_workers=2) as readers, torch.no_grad():
            for group_index, group in enumerate(groups):
                loaded = list(readers.map(read_reference, group))
                observation = combine_observations([obs for _, _, obs in loaded])
                if group_index == 0:
                    small = {k: v[:1] for k, v in observation.items()}
                    _, original_prepared = native.prepare_batch(small)
                    _, final_prepared = final_port.prepare_batch(small)
                    a, b = original_prepared["prefix"], final_prepared["prefix"]
                    if (not torch.equal(a[0], b[0]) or not torch.equal(a[1], b[1])
                            or len(a[2]) != 18 or any(not torch.equal(x, y)
                                for pair_a, pair_b in zip(a[2], b[2]) for x, y in zip(pair_a, pair_b))):
                        raise ValueError("Final teacher changed the frozen native visual prefix/cache")
                    result["original_final_native_prefix_bitwise_parity"] = True
                    native._prepared = final_port._prepared = None
                    del a, b, original_prepared, final_prepared
                f, prepared = motor.prepare_batch(observation)
                noise = np.stack([teacher_noise(ref["sha256"], row)
                    for _, ref, _ in loaded for row in range(ref["rows"])])
                original_builder = final.build_prefix_cache
                try:
                    final.build_prefix_cache = lambda *a, **kw: prepared["prefix"]
                    action, _ = final.predict_action_batch(prepared["observation"], mode="eval",
                        noise=torch.as_tensor(noise, device=final.device))
                    action = codec.mapped_physical(action.detach().float().cpu().numpy())
                finally:
                    final.build_prefix_cache = original_builder
                    native._prepared = None
                positions, cursor = [], 0
                for index, reference, obs in loaded:
                    for row in range(reference["rows"]):
                        s, t = mapping[index, row]
                        if done[s, t] or not np.array_equal(f[cursor, :25], raw["public_states"][s, t]):
                            raise ValueError("Source query mapping or state identity differs")
                        features[s, t], labels[s, t], done[s, t] = f[cursor], action[cursor], True
                        positions.append([s, t, index, row]); cursor += 1
                shard = _save(out / "shards" / f"group-{group_index:04d}.pt", dict(
                    positions=positions, features=f, teacher_mapped_actions=action,
                    source_references=[ref for _, ref, _ in loaded],
                    teacher_sha256=sha(teacher_file) if group_index == 0 else result["teacher_sha256"],
                    sampling="flow_final18_query_noise_v1; native eval ODE10; one seeded draw/query, not conditional mean"))
                shards.append(shard)
                result["teacher_sha256"] = "7e65e25408ed653d4c635a7f23b788ac621e8ff9909e866cec88173f519eff61"
                if group_index % 10 == 0 or group_index + 1 == len(groups):
                    write(out / "SHARDS.json", shards)
                    stage("source_features_and_final18_queries", groups=group_index+1, total_groups=len(groups),
                        complete_queries=int(done.sum()), queries_per_second=float(done.sum())/(time.time()-extraction_started))
        if not done.all() or tensor_hash(_versions(native.model)) != frozen_versions:
            raise ValueError("Incomplete source mapping or frozen initializer mutation")
        stage("export_all_four_complete_corpora")
        np.savez(out / "FEATURES-TEACHER.npz", features=features, mapped_teacher_actions=labels)
        provenance = dict(motor_sha256=signature["model_sha256"], raw_corpus_sha256=sha(raw_path),
            source_commit=args.source_commit, runtime_commit=request["runtime_commit"],
            teacher=dict(path=str(teacher_file), sha256=result["teacher_sha256"], iteration=18,
                sampling="native eval ODE10; fixed independent per-source-query PCG64 initial Gaussian"),
            teacher_label_kind="single stochastic final-teacher action sample; not analytic mean",
            source_quality="paired35to43/64 p=.18493; source improvement not established")
        datasets = export_corpus(out / "corpus", raw, features, labels, codec, provenance=provenance)
        assembly = dict(completed=True, producer_recipe=dict(finalized=True, iterations=18,
            source="completed Flow-SDE", physical_controls=921600), source_physical_horizons=[400],
            datasets=datasets, provenance=provenance, codec=codec.record(), feature_dim=2073, action_dim=600,
            teacher_queries=18432, source_episodes=2304, source_TRAIN_controls=921600,
            student_fit_ready=True, native_readout_ready=False)
        write(out / "ASSEMBLY.json", assembly)
        result.update(completed=True, source_queries=18432, source_episodes=2304,
            feature_and_teacher_seconds=time.time()-extraction_started, source_ready=True,
            native_readout_ready=False, peak_cuda_bytes=torch.cuda.max_memory_allocated())
        stage("source_preparation_complete")
    except BaseException as error:
        import traceback
        result.update(error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        result.update(ended_unix=time.time(), elapsed_seconds=time.time()-started)
        write(out / "RESULT.json", result)


if __name__ == "__main__":
    main()
