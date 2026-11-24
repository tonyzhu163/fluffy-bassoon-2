"""Inventory R2 Sawyer teacher data and recover one complete public observation tape."""
import argparse
from collections import Counter
import csv
import hashlib
import json
import os
from pathlib import Path
from zipfile import ZipFile

import numpy as np

PROJECT = Path("/mowenjie/fluffy-bassoon-2")
ARCHIVE = PROJECT / "outputs/libero-paligemma-dualarm-20261120-r2"
MANIFEST = ARCHIVE / "scale3fit01/fit/manifest.json"
CAMERAS = ("agentview_image", "robot0_eye_in_hand_image")
PROPRIO = ("robot0_eef_pos", "robot0_eef_quat_xyzw", "robot0_gripper_qpos")
PROMPT = "push the plate to the front of the stove"
SEED = 300000


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def inventory(out):
    manifest = json.loads(MANIFEST.read_text())
    episodes = manifest["contract"]["episodes"]
    indexes = {}
    rows = []
    kinds = Counter()
    for e in episodes:
        if (e["body"], e["split"]) != ("Sawyer", "train"):
            continue
        root = Path(manifest["demos"][e["bank"]])
        if root not in indexes:
            idx = json.loads((root / "index.json").read_text())
            assert (idx["suite"], idx["task_id"], idx["case"]) == ("libero_goal", 5, "BASE")
            indexes[root] = {r["file"]: r for r in idx["attempts"] if r.get("file")}
        r = indexes[root][e["file"]]
        if r.get("route") == "recovery_paddle":
            kinds["expert_recovery_suffix_excluded"] += 1
            continue
        generated = e.get("source_kind") == "native_bddl_sample"
        assert generated or 0 <= e["pose"] < 28
        if generated:
            assert r["split"] == "train" and r["seed"] in range(300000, 400000)
            assert e["generated_row_fingerprint"] not in e["public_row_fingerprints"]
            assert r["same_initial_state"] and r["search_steps"] == r["steps"]
        assert r["accepted"] and r["done"] and r["native_success"] and r["terminal_reward"] == 1
        assert r["body"] == "Sawyer" and e["prompt"] == PROMPT
        assert r.get("route") in (None, "rendered_commands", "whole_gripper_push")
        identity = r["reset"]["model_identity"]
        assert (identity["model_class"], identity["joint_count"], identity["controller"],
                identity["gripper"], identity["control_freq"]) == ("MountedSawyer", 7, "OSC_POSE", "PandaGripper", 20)
        path = root / e["file"]
        n = e["frames"]
        assert n == r["steps"]
        with ZipFile(path) as archive:
            for key in ("rgb", "wrist"):
                with archive.open(key + ".npy") as member:
                    assert np.lib.format.read_magic(member) == (1, 0)
                    shape, _, dtype = np.lib.format.read_array_header_1_0(member)
                    assert shape == (n, 256, 256, 3) and dtype == np.uint8
        with np.load(path, allow_pickle=False) as tape:
            assert tape["body"].item() == "Sawyer" and tape["prompt"].item() == PROMPT
            assert tape["native_success"].item() and tape["pose"] == e["pose"] and tape["seed"] == e["seed"]
            action, state, reward = (tape[k] for k in ("actions", "state", "rewards"))
            assert action.shape == (n, 7) and state.shape == (n, 8) and reward.shape == (n,)
            assert np.isfinite(state).all() and np.isfinite(action).all() and np.abs(action).max() <= 1
            assert not reward[:-1].any() and reward[-1] == 1
        kind = "generated" if generated else "released"
        kinds[kind] += 1
        rows.append(dict(kind=kind, pose=e["pose"], seed=e["seed"], controls=n, terminal_reward=1,
                         bank=e["bank"], path=str(path.relative_to(ARCHIVE)), collector_sha256=e["sha256"],
                         trajectory_sha256=e["trajectory_sha256"], scene_row=e.get("generated_row_fingerprint")
                         or e["released_row_fingerprint"]))
    assert len({r["trajectory_sha256"] for r in rows}) == len(rows)
    with (out / "training-tapes.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0], lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    counts = {}
    for kind in ("generated", "released"):
        group = [r for r in rows if r["kind"] == kind]
        panda = {e.get("generated_row_fingerprint") or e["released_row_fingerprint"]
                 for e in episodes if (e["body"], e["split"]) == ("Panda", "train")
                 and e.get("route") != "recovery_paddle"
                 and (e.get("source_kind") == "native_bddl_sample") == (kind == "generated")}
        layouts = {r["scene_row"] for r in group}
        counts[kind] = dict(tapes=len(group), layouts=len(layouts), controls=sum(r["controls"] for r in group),
                            ten_action_windows=sum(max(0, r["controls"]-9) for r in group),
                            shared_panda_layouts=len(layouts & panda))
    save_json(out / "coverage.json", dict(manifest=str(MANIFEST), sha256=sha(MANIFEST), counts=counts,
              excluded_recovery=kinds["expert_recovery_suffix_excluded"], excluded_validation=sum(
                  (e["body"], e["split"]) == ("Sawyer", "validation") for e in episodes),
              index_sha256={str(p.relative_to(ARCHIVE)): sha(p/"index.json") for p in indexes}))
    print(counts, flush=True)


def replay(out):
    import h5py
    from arm_env import ArmEnv

    root = ARCHIVE / "scale3fit01/data/bank1"
    idx = json.loads((root / "index.json").read_text())
    r, = (r for r in idx["attempts"] if r.get("file") and r["body"] == "Sawyer" and r["seed"] == SEED)
    layout, = (l for l in idx["layouts"] if l["layout_seed"] == SEED)
    assert idx["split"] == r["split"] == layout["split"] == "train" and r["route"] == "rendered_commands"
    assert layout["initial_state_sha256"] not in idx["public_row_fingerprints"]
    path, draft, initial = (root / name for name in (r["file"], r["search_file"], layout["state_file"]))
    assert (sha(path), sha(draft), sha(initial)) == (r["sha256"], r["search_sha256"], layout["state_sha256"])
    obs = {key: [] for key in (*CAMERAS, *PROPRIO)}
    rewards, issued = [], []
    with np.load(path, allow_pickle=False) as tape, ArmEnv("Sawyer", "libero_goal", 5) as arm:
        actions = tape["actions"]
        assert np.array_equal(actions, np.load(draft, allow_pickle=False))
        # The bank's slot 0 is the new sampled training placement, not released row 0.
        arm._states = (np.load(initial, allow_pickle=False),)
        packet = arm.reset(0, SEED)
        identity = packet["model_identity"]
        assert identity == r["reset"]["model_identity"] and packet["language_instruction"] == PROMPT
        physics = packet["environment_reproducibility"]
        assert physics["released_row_fingerprint"] == layout["initial_state_sha256"]
        assert physics["scene_fingerprint"] == layout["scene_hash"]
        assert physics["bddl_sha256"] == layout["bddl_sha256"] and physics["source_pin"] == layout["source_pin"]
        for action in actions:
            for key in obs:
                obs[key].append(packet["observation"][key].copy())
            packet = arm.step(action)
            issued.append(packet["issued_action"])
            rewards.append(packet["reward"])
            if packet["terminated"] or packet["truncated"]:
                break
        for key in obs:
            obs[key].append(packet["observation"][key].copy())
            obs[key] = np.stack(obs[key])
        n = len(issued)
        assert np.array_equal(actions[:n], issued)
        fingers = obs[PROPRIO[2]]
        measured = np.column_stack((obs[PROPRIO[0]], obs[PROPRIO[1]], fingers[:, 0] - fingers[:, 1]))
        original = np.vstack((tape["state"], tape["terminal_state"]))
        delta = measured - original[:n+1]
        result = dict(archive=str(path), archive_sha256=r["sha256"], draft_sha256=r["search_sha256"],
                      sampler_state_sha256=layout["state_sha256"], seed=SEED, group=r["pose"],
                      source_id=layout["source_pin"], bddl_sha256=layout["bddl_sha256"],
                      scene_sha256=layout["scene_hash"], model_identity=identity,
                      replay_settled=physics["initial_state_fingerprint"], archive_settled=r["reset"][
                          "environment_reproducibility"]["initial_state_fingerprint"],
                      prompt=PROMPT, native_success=packet["success"], terminated=packet["terminated"],
                      truncated=packet["truncated"], terminal_reward=rewards[-1], native_return=sum(rewards),
                      archived_controls=len(actions), executed_controls=n, exact_issued_prefix=True,
                      rewards_match=np.array_equal(tape["rewards"], rewards),
                      max_xyz_error_m=float(np.linalg.norm(delta[:, :3], axis=1).max()),
                      max_quat_component_error=float(abs(delta[:, 3:7]).max()),
                      max_signed_jaw_error_m=float(abs(delta[:, -1]).max()),
                      max_error_if_symmetric_fingers_m=float(abs(fingers.sum(axis=1)).max()/2),
                      distinct_preaction_frames={k:len({a.tobytes() for a in obs[k][:-1]}) for k in CAMERAS},
                      cpu_affinity=sorted(os.sched_getaffinity(0)), cuda_visible=os.environ["CUDA_VISIBLE_DEVICES"])
    valid = bool(packet["success"] and packet["terminated"] and rewards[-1] == 1 and not any(rewards[:-1]))
    if valid:
        output = out / "sawyer-300000.h5"
        with h5py.File(output, "x") as f:
            f.attrs.update(arm="Sawyer", prompt=PROMPT, source_sha256=r["sha256"], seed=SEED,
                           success=True, terminated=packet["terminated"], truncated=packet["truncated"])
            for key, arr in obs.items():
                f.create_dataset(key, data=arr, compression="lzf", chunks=(1, *arr.shape[1:]))
            f.create_dataset("actions", data=np.asarray(issued))
            f.create_dataset("rewards", data=np.asarray(rewards))
        with h5py.File(output, "r") as f:
            for key in obs:
                assert np.array_equal(f[key][:], obs[key])
            assert np.array_equal(f["actions"][:], actions[:n]) and np.array_equal(f["rewards"][:], rewards)
        result.update(file=str(output), file_sha256=sha(output), bytes=output.stat().st_size,
                      observation_count=n+1, ten_action_windows=max(0,n-9))
    save_json(out/"replay.json", result)
    print("Sawyer", n, rewards[-1], valid, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("inventory", "replay"))
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    assert 1 <= len(os.sched_getaffinity(0)) <= 2 and os.environ["CUDA_VISIBLE_DEVICES"] == ""
    {"inventory": inventory, "replay": replay}[args.mode](args.output)
