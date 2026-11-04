#!/usr/bin/env python3
"""One of four disjoint BOS streaming shards for 130-update Flow-SDE corpora.

Streams environment/rank-00..07/training-i001..130-round00-chunk01..12.pt.gz
directly from the verified BOS tar archive in memory without unpacking raw
chunks onto PFS, extracts frozen VLM 2073-d features and global_step_130
teacher actions in GPU batches, writes TAR-INDEX.json for on-demand query-step
preads, and exports the four canonical (ad, dpt, dit, ic_iql) .npz corpora.
"""
import argparse
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import sys
import tarfile
import time


TASKS = {
    "OpenDrawer": {
        "q1_stage": "q1-r2",
        "prepared_receipt": "/mowenjie/fluffy-bassoon-2/outputs/robocasa-flow-sde-10m-per-task-20261103-r1/CORPUS-ARCHIVE-OpenDrawer-R1-PREPARED.json",
        "prepared_sha256": "12055d1ca468333971884470a5ea411ea3e511907b8bc6a5dc7b78eeef3c66e8",
        "archive_sha256": "c0817a8be0a250c357d8f40d57ec1508da1d634a5bac7e54ba98f3d2c1ef44c6",
        "boundary_sha256": "10deeac6761a5018ce8aa6eaed978d449e0f642552cfe68c8ae26bb56c613e57",
    },
    "CloseToasterOvenDoor": {
        "q1_stage": "q1-r1",
        "prepared_receipt": "/mowenjie/fluffy-bassoon-2/outputs/robocasa-flow-sde-10m-per-task-20261103-r1/CORPUS-ARCHIVE-CloseToasterOvenDoor-R1-PREPARED.json",
        "prepared_sha256": "2a3df641c4e0b6cbb8123add201b17e16b874d3c1ffc49b2f002675350637659",
        "archive_sha256": "788b6e3357bdd17786fbec1fabb6811a74b1f4e8ef1d985a05e28d6c64fd322a",
        "boundary_sha256": "5e525b1b8df3a41df79cd8d4d76ccae918d93eb12eac58c0ad9d9e49449583ef",
    },
    "CoffeeSetupMug": {
        "q1_stage": "q1-r1",
        "prepared_receipt": "/mowenjie/fluffy-bassoon-2/outputs/robocasa-flow-sde-10m-per-task-20261103-r1/CORPUS-ARCHIVE-CoffeeSetupMug-R1-PREPARED.json",
        "prepared_sha256": "f9474e04c21f4a8708ae75e077b8f238dc95e90019266d600f4637d3da1a2842",
        "archive_sha256": "1ad7c6674e642ca0da8d54687e62c0b31b16d1ddca01e0757b00d5514ef02da1",
        "boundary_sha256": "d04c3136edc34189ff7337983725a6378844a823c79c55cb28849532a94bc09f",
    },
}

CHUNK_RE = re.compile(
    r"^(?P<task>[^/]+)/(?P<stage>q1-r[12]|production-r1)/environment/rank-(?P<rank>\d{2})/"
    r"training-i(?P<iter>\d{3})-round(?P<round>\d{2})-chunk(?P<chunk>\d{2})\.pt\.gz$"
)


def valid_env_member(task, name):
    m = CHUNK_RE.match(name)
    if not m or m.group("task") != task:
        return None
    stage = m.group("stage")
    rank = int(m.group("rank"))
    version = int(m.group("iter"))
    round_idx = int(m.group("round"))
    chunk = int(m.group("chunk"))
    q1_stage = TASKS[task]["q1_stage"]
    if not (0 <= rank < 8 and round_idx == 0 and 1 <= chunk <= 12):
        return None
    if (stage == q1_stage and version == 0) or (stage == "production-r1" and 1 <= version <= 129):
        return dict(stage=stage, rank=rank, version=version, chunk=chunk)
    return None


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    tmp.replace(path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", choices=sorted(TASKS), required=True)
    p.add_argument("--code", type=Path, required=True)
    p.add_argument("--source-commit", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--expected-hostname", required=True)
    p.add_argument("--max-seconds", type=int, default=12000)
    p.add_argument("--files-per-batch", type=int, default=4)
    p.add_argument("--shard-index", type=int, required=True)
    p.add_argument("--allocation-probe", type=Path, required=True)
    args = p.parse_args()
    if not 0 <= args.shard_index < 4 or args.files_per_batch < 1 or args.max_seconds < 1:
        raise ValueError("Require one of four disjoint preparation shards and positive bounds")

    if sys.platform != "linux" or socket.gethostname() != args.expected_hostname:
        raise ValueError("Require exact allocated Linux hostname")
    root = Path("/mowenjie/fluffy-bassoon-2")
    if os.statvfs(root).f_bavail * os.statvfs(root).f_frsize < 220 * (1 << 30):
        raise ValueError("Refuse to run when PFS free space is near the 200 GiB floor")

    spec_info = TASKS[args.task]
    prepared_path = Path(spec_info["prepared_receipt"])
    if sha(prepared_path) != spec_info["prepared_sha256"]:
        raise ValueError("Prepared archive receipt hash mismatch")
    prepared_meta = json.loads(prepared_path.read_text())
    archive_path = Path(prepared_meta["archive"])
    manifest_path = Path(prepared_meta["manifest"])
    if sha(manifest_path) != prepared_meta["manifest_sha256"]:
        raise ValueError("BOS tar manifest hash mismatch")
    manifest_files = json.loads(manifest_path.read_text())["files"]

    teacher_cp = Path(prepared_meta["checkpoint"]["path"])
    if sha(teacher_cp / "pilot-boundary.json") != spec_info["boundary_sha256"]:
        raise ValueError("Step-130 checkpoint boundary hash mismatch")
    teacher_dir = teacher_cp / "actor/model_state_dict"
    teacher_weights = teacher_dir / "full_weights.pt"
    teacher_sha = sha(teacher_weights)

    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    started = time.time()

    def interrupted(signum, frame):
        raise TimeoutError("Operational preparation time limit reached; completed shards preserved")

    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGALRM):
        signal.signal(sig, interrupted)
    signal.setitimer(signal.ITIMER_REAL, args.max_seconds)

    runtime = root / "sources/critic-selected-a1229239-r1"
    model = root / "inputs/robocasa365-public-pi05-c484-torch-20261101-r2"
    norm = root / "inputs/robocasa365-public-pi05-c484-75000-20261101-r1/assets/norm_stats.json"
    os.environ.update(
        RLINF_PATH=str(runtime),
        ROBOCASA_NATIVE_OUTPUT=str(out / "native"),
        ROBOCASA_PI05_TORCH_MODEL=str(model),
        ROBOCASA_PI05_NATIVE_NORM=str(norm),
    )
    sys.path[:0] = [str(args.code / "src"), str(runtime)]

    from fluffy_bassoon.meta_rl.rlinf_native_ppo import checked_checkout, tensor_hash
    checked_checkout(args.code, args.source_commit)

    # The parent probes the complete four-GPU allocation before any workers start.
    # NVIDIA_VISIBLE_DEVICES remains the scheduler's complete UUID assignment.
    from fluffy_bassoon.baige_gpu import verify_selected_cuda, selected_device
    import torch
    os.environ["LYCO_BAIGE_GPU_MAP"] = str(args.allocation_probe)
    verify_selected_cuda(torch)
    device = selected_device()
    os.environ["MUJOCO_EGL_DEVICE_ID"] = str(device["allocation_ordinal"])

    import numpy as np
    import torch
    from hydra import compose, initialize_config_dir
    from rlinf.models.embodiment.openpi import get_model
    from fluffy_bassoon.meta_rl.flow_history import (
        DirectActionCodec, FlowDirectMotor, teacher_noise)
    from fluffy_bassoon.meta_rl.joint_steering_ppo_worker import _load_native, _versions
    from fluffy_bassoon.meta_rl.robocasa_residual import combine_observations, public_observation

    torch.set_num_threads(4)
    request = dict(
        model_directory=str(model), norm_path=str(norm),
        source_commit=args.source_commit,
        runtime_commit=subprocess.check_output(["git", "-C", str(runtime), "rev-parse", "HEAD"], text=True).strip(),
    )
    native, _, signature = _load_native(request, args.code)
    with initialize_config_dir(config_dir=str(args.code / "configs/native_ppo"), version_base=None):
        cfg = compose(config_name="robocasa365_pi05_eval")
    cfg.rollout.model.model_path = str(teacher_dir)
    cfg.rollout.model.openpi_data.norm_stats_path = str(norm)
    final = get_model(cfg.rollout.model, torch_dtype=torch.bfloat16).to("cuda").eval().requires_grad_(False)
    stats = json.loads(norm.read_text())
    stats = stats.get("norm_stats", stats)
    codec = DirectActionCodec.from_stats(stats)
    motor = FlowDirectMotor(native, codec)
    frozen_versions = tensor_hash(_versions(native.model))

    num_streams, num_versions, chunks_per_ep = 32, 130, 12
    rank_start = args.shard_index * 2
    total_chunks = num_versions * chunks_per_ep
    total_members = 2 * num_versions * chunks_per_ep

    states = np.zeros((num_streams, total_chunks, 25), np.float32)
    actions = np.zeros((num_streams, total_chunks, 50, 12), np.float32)
    rewards = np.zeros((num_streams, total_chunks, 50), np.float32)
    terminations = np.zeros((num_streams, total_chunks, 50), bool)
    truncations = np.zeros((num_streams, total_chunks, 50), bool)
    seeds = np.zeros((num_streams, total_chunks), np.int64)
    features = np.zeros((num_streams, total_chunks, 2073), np.float32)
    labels = np.zeros((num_streams, total_chunks, 50, 12), np.float32)
    done = np.zeros((num_streams, total_chunks), bool)
    tar_index = []

    def process_pair(batch_items):
        obs_list = [item["obs"] for item in batch_items]
        combined = combine_observations(obs_list)
        with torch.no_grad():
            feat, prepared = motor.prepare_batch(combined)
            noise = np.stack([
                teacher_noise(item["sha256"], row)
                for item in batch_items for row in range(16)
            ])
            orig_builder = final.build_prefix_cache
            try:
                final.build_prefix_cache = lambda *a, **kw: prepared["prefix"]
                pred, _ = final.predict_action_batch(
                    prepared["observation"], mode="eval",
                    noise=torch.as_tensor(noise, device=final.device),
                )
                pred_phys = codec.mapped_physical(pred.detach().float().cpu().numpy())
            finally:
                final.build_prefix_cache = orig_builder
                native._prepared = None
        cursor = 0
        for item in batch_items:
            rank, version, chunk = item["rank"], item["version"], item["chunk"]
            t = version * chunks_per_ep + (chunk - 1)
            for row in range(16):
                slot = (rank - rank_start) * 16 + row
                if done[slot, t]:
                    raise ValueError(f"Duplicate slot/time ({slot}, {t})")
                states[slot, t] = item["states"][row]
                actions[slot, t] = item["actions"][row]
                rewards[slot, t] = item["rewards"][row]
                terminations[slot, t] = item["terminations"][row]
                truncations[slot, t] = item["truncations"][row]
                seeds[slot, t] = item["seeds"][row]
                features[slot, t] = feat[cursor]
                labels[slot, t] = pred_phys[cursor]
                done[slot, t] = True
                cursor += 1

    pending = []
    seen_members = 0
    with tarfile.open(archive_path, mode="r:") as tf:
        for member in tf:
            parsed = valid_env_member(args.task, member.name)
            if parsed is None:
                if seen_members == total_members:
                    break
                continue
            rank = parsed["rank"]
            if rank not in (rank_start, rank_start + 1):
                continue
            version = parsed["version"]
            chunk = parsed["chunk"]
            raw_bytes = tf.extractfile(member).read()
            digest = hashlib.sha256(raw_bytes).hexdigest()
            expected = manifest_files[member.name]
            if len(raw_bytes) != expected["bytes"] or digest != expected["sha256"]:
                raise ValueError(f"Hash/size mismatch on {member.name}")
            payload = torch.load(io.BytesIO(gzip.decompress(raw_bytes)), map_location="cpu", weights_only=False)
            obs = payload["current_observation"]
            pending.append(dict(
                name=member.name, sha256=digest, rank=rank, version=version, chunk=chunk,
                obs=public_observation(obs),
                states=np.asarray(obs["states"], np.float32),
                actions=np.asarray(payload["decoded_actions"], np.float32),
                rewards=np.asarray(payload["rewards"], np.float32),
                terminations=np.asarray(payload["terminations"], bool),
                truncations=np.asarray(payload["truncations"], bool),
                seeds=np.asarray(payload["seeds"], np.int64),
            ))
            tar_index.append(dict(
                name=member.name, offset_data=int(member.offset_data),
                bytes=len(raw_bytes), sha256=digest,
                stage=parsed["stage"], rank=rank, version=version, chunk=chunk,
            ))
            seen_members += 1
            if len(pending) == args.files_per_batch:
                process_pair(pending)
                pending = []
                if seen_members % 200 == 0 or seen_members == total_members:
                    write_json(out / "PROGRESS.json", dict(
                        task=args.task, seen_members=seen_members, total_members=total_members,
                        completed_decisions=int(done.sum()), elapsed_seconds=time.time() - started,
                    ))
            if seen_members == total_members:
                break

    if pending:
        process_pair(pending)
    if not done.all() or seen_members != total_members or tensor_hash(_versions(native.model)) != frozen_versions:
        raise ValueError("Incomplete stream coverage or frozen VLM mutation")

    raw_corpus = dict(
        public_states=states, physical_actions=actions, primitive_rewards=rewards,
        native_seeds=seeds, native_terminations=terminations, native_truncations=truncations,
        producer_versions=np.broadcast_to(np.repeat(np.arange(num_versions), chunks_per_ep), (num_streams, total_chunks)),
        episode_starts=np.broadcast_to(np.arange(total_chunks) % chunks_per_ep == 0, (num_streams, total_chunks)),
        physical_episode_ends=np.broadcast_to(np.arange(total_chunks) % chunks_per_ep == chunks_per_ep - 1, (num_streams, total_chunks)),
    )
    raw_npz = out / "RAW-PUBLIC-CORPUS.npz"
    np.savez(raw_npz, **raw_corpus)
    np.savez(out / "FEATURES-TEACHER-130.npz", features=features, mapped_teacher_actions=labels)
    write_json(out / "TAR-INDEX.json", tar_index)

    provenance = dict(
        task=args.task, motor_sha256=signature["model_sha256"],
        raw_corpus_sha256=sha(raw_npz),
        bos_archive=str(archive_path), bos_archive_sha256=spec_info["archive_sha256"],
        source_commit=args.source_commit, runtime_commit=request["runtime_commit"],
        teacher=dict(
            path=str(teacher_weights), sha256=teacher_sha, iteration=130,
            boundary_sha256=spec_info["boundary_sha256"],
            sampling="native eval ODE10; deterministic per-query PCG64 initial Gaussian",
        ),
    )
    files = {
        name: dict(path=str(out / name), sha256=sha(out / name))
        for name in ("RAW-PUBLIC-CORPUS.npz", "FEATURES-TEACHER-130.npz", "TAR-INDEX.json")
    }
    shard = dict(
        completed=True, task=args.task, shard_index=args.shard_index,
        global_slots=list(range(rank_start * 16, (rank_start + 2) * 16)),
        files=files, codec=codec.record(), provenance=provenance,
        teacher_queries=int(done.sum()), expected_members=total_members,
        scheduler_uuid=device["uuid"], elapsed_seconds=time.time() - started,
    )
    write_json(out / "SHARD.json", shard)
    print(json.dumps(shard), flush=True)


if __name__ == "__main__":
    main()
