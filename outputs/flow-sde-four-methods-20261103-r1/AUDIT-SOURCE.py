#!/usr/bin/env python3
"""Read-only full physical-corpus audit; no models, simulator or GPU calls."""
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import re
import socket
import sys
import time

import numpy as np
import torch

torch.set_num_threads(2)
assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
assert socket.gethostname() == os.environ["TASK_EXPECTED_CPU_HOST"]
base = Path("/mowenjie/fluffy-bassoon-2/outputs")
out = Path(sys.argv[1])
out.mkdir(exist_ok=False)
storage = os.statvfs(out)
assert storage.f_bavail * storage.f_frsize > 32 * 2**30
started = time.time()
episodes = {}
chunks = []
excluded = []
for suffix, ranks, batch, versions in [("r3", 2, 16, range(6)), ("r4", 8, 4, range(6, 18))]:
    root = base / f"robocasa-opendrawer-flow-sde-20261102-{suffix}"
    for rank in range(ranks):
        directory = root / "environment" / f"rank-{rank:02d}"
        for path in sorted(directory.glob("training-*.pt.gz")):
            match = re.fullmatch(r"training-i(\d+)-round(\d+)-chunk(\d+)\.pt\.gz", path.name)
            assert match, path
            version, round_index, chunk = map(int, match.groups())
            if version not in versions:
                assert suffix == "r3" and version == 6
                excluded.append(str(path))
                continue
            raw = path.read_bytes()
            payload = torch.load(io.BytesIO(gzip.decompress(raw)), map_location="cpu", weights_only=False)
            assert payload["phase"] == "training" and payload["iteration"] == version
            assert payload["round"] == round_index and 0 <= round_index < 4 and 1 <= chunk <= 8
            obs = payload["current_observation"]
            state = np.asarray(obs["states"])
            action = np.asarray(payload["decoded_actions"])
            reward = np.asarray(payload["rewards"])
            terminated = np.asarray(payload["terminations"])
            truncated = np.asarray(payload["truncations"])
            assert state.shape == (batch, 25) and action.shape == (batch, 50, 12)
            assert reward.shape == (batch, 50)
            assert terminated.shape == truncated.shape == (batch, 50)
            assert np.isfinite(state).all() and np.isfinite(action).all() and np.isfinite(reward).all()
            assert np.isin(action[:, :, [6, 11]], [-1., 1.]).all()
            assert np.isin(reward, [0., 1.]).all()
            for name in ["main_images", "wrist_images"]:
                image = np.asarray(obs[name])
                assert image.shape == (batch, 256, 256, 3) and image.dtype == np.uint8
            extra = np.asarray(obs["extra_view_images"])
            assert extra.shape == (batch, 1, 256, 256, 3) and extra.dtype == np.uint8
            ref = dict(path=str(path), sha256=hashlib.sha256(raw).hexdigest(), rows=batch,
                producer_version=version, round=round_index, environment_rank=rank, chunk=chunk)
            reference_index = len(chunks)
            chunks.append(ref)
            for slot in range(batch):
                global_slot = round_index * 32 + rank * batch + slot
                key = (version, global_slot)
                seed = int(payload["seeds"][slot])
                metadata = obs["task_metadata"][slot]
                assert metadata["task_name"] == "OpenDrawer"
                episode = episodes.setdefault(key, dict(seed=seed, rows={}))
                assert episode["seed"] == seed and chunk not in episode["rows"]
                episode["rows"][chunk] = dict(reference_index=reference_index, row=slot,
                    public_state=state[slot].copy(), action=action[slot].copy(),
                    primitive_rewards=reward[slot].copy(), terminated=terminated[slot].copy(),
                    truncated=truncated[slot].copy())
        (out / "PROGRESS.json").write_text(json.dumps(dict(last_directory=str(directory),
            checked_files=len(chunks), episodes_seen=len(episodes), checked_unix=time.time())) + "\n")
assert set(episodes) == {(v, s) for v in range(18) for s in range(128)}
assert len(chunks) == 3456 and len(excluded) == 2
states = np.zeros((128, 144, 25), np.float32)
actions = np.zeros((128, 144, 50, 12), np.float32)
rewards = np.zeros((128, 144, 50), np.float32)
terminations = np.zeros((128, 144, 50), bool)
truncations = np.zeros((128, 144, 50), bool)
seeds = np.zeros((128, 144), np.int64)
refs = np.zeros((128, 144, 2), np.int64)
index = []
for (version, slot), episode in sorted(episodes.items()):
    assert set(episode["rows"]) == set(range(1, 9))
    assert sum(float(row["primitive_rewards"].sum()) for row in episode["rows"].values()) <= 1
    for chunk, row in episode["rows"].items():
        t = version * 8 + chunk - 1
        states[slot, t], actions[slot, t], rewards[slot, t] = row["public_state"], row["action"], row["primitive_rewards"]
        terminations[slot, t], truncations[slot, t] = row["terminated"], row["truncated"]
        seeds[slot, t] = episode["seed"]
        refs[slot, t] = row["reference_index"], row["row"]
    index.append(dict(producer_version=version, stream_slot=slot, native_seed=episode["seed"],
        chunks=[dict(chunk=n, reference_index=episode["rows"][n]["reference_index"],
            row=episode["rows"][n]["row"]) for n in range(1, 9)]))
assert int(rewards.sum()) == 1336
np.savez(out / "RAW-PUBLIC-CORPUS.npz", public_states=states, physical_actions=actions,
    primitive_rewards=rewards, native_seeds=seeds, raw_chunk_references=refs,
    native_terminations=terminations, native_truncations=truncations,
    producer_versions=np.broadcast_to(np.repeat(np.arange(18), 8), (128, 144)),
    episode_starts=np.broadcast_to(np.arange(144) % 8 == 0, (128, 144)),
    physical_episode_ends=np.broadcast_to(np.arange(144) % 8 == 7, (128, 144)))
(out / "SOURCE-CHUNKS.json").write_text(json.dumps(chunks, indent=2) + "\n")
(out / "EPISODES.json").write_text(json.dumps(index, indent=2) + "\n")
result = dict(completed=True, hostname=socket.gethostname(), CPU_only=True,
    model_simulator_or_GPU_calls=0, source_complete_episodes=len(episodes),
    source_H50_decisions=18432, source_TRAIN_controls=921600, source_policy_versions=list(range(18)),
    chronological_streams=128, decisions_per_stream=144, audited_chunk_files=len(chunks),
    positive_first_success_rewards=int(rewards.sum()), excluded_partial_paths=excluded,
    excluded_resolved_controls=1600, separately_charged_unresolved_controls=800,
    terminal400_raw_RGB_available=False, frozen_visual_features_available=False,
    final_teacher_labels_available=False, student_fit_ready=False,
    state_dimensions=25, physical_action_chunk_shape=[50, 12],
    corpus_sha256=hashlib.sha256((out / "RAW-PUBLIC-CORPUS.npz").read_bytes()).hexdigest(),
    elapsed_seconds=time.time()-started)
(out / "RESULT.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result), flush=True)
