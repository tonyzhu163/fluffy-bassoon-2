"""Replay archived generated plate teachers on physical Panda and Sawyer, with native sensors."""
import argparse
import hashlib
import json
import os
from pathlib import Path

import h5py
import numpy as np

from source_data import FINGERS, HORIZON, IMAGE_KEYS, POS, QUAT, PublicHistory

BASE = Path("/mowenjie/fluffy-bassoon-2/outputs")
ARCHIVE = BASE / "libero-paligemma-dualarm-20261120-r2"
SOURCE = ARCHIVE / "scale3fit01/fit/manifest.json"
PRIOR = Path(__file__).resolve().parents[2] / "frozen-vla-cotraining-20261124-r1"
PROMPT = "push the plate to the front of the stove"
BODIES = ("Panda", "Sawyer")
KEYS = (*IMAGE_KEYS, POS, QUAT, FINGERS)
RESET_REJECTIONS = ("Released grip-site pose unreachable", "Settling moved the paired task scene",
                    "Task already successful")


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def check(condition, message):
    if not condition:
        raise ValueError(message)


def pool(out):
    coverage = json.loads((PRIOR / "source-diversity/evidence/coverage.json").read_text())
    check(sha(SOURCE) == coverage["sha256"], "Source archive manifest digest")
    manifest = json.loads(SOURCE.read_text())
    pairs = {}
    for ep in manifest["contract"]["episodes"]:
        if (ep["body"] not in BODIES or ep["split"] != "train" or
                ep.get("source_kind") != "native_bddl_sample" or ep.get("route") != "rendered_commands"):
            continue
        origin = ep["layout_origin"]
        scene = origin["scene_hash"]
        check((ep["suite"], ep["task_id"], ep["case"], ep["prompt"]) == ("libero_goal", 5, "BASE", PROMPT)
              and ep["seed"] == ep["layout_seed"] == origin["native_sampler_seed"] == origin["layout_seed"]
              and ep["native_terminal_reward"] == 1 and origin["released_rows_checked"] == 50
              and ep["generated_row_fingerprint"] == origin["initial_state_sha256"]
              and ep["generated_row_fingerprint"] not in ep["public_row_fingerprints"]
              and scene == ep["scene_fingerprint"] and origin["collision_free"], "Invalid generated source")
        if scene not in pairs:
            pairs[scene] = {}
        check(ep["body"] not in pairs[scene], "Ambiguous archive teacher per body/scene")
        pairs[scene][ep["body"]] = ep
    candidates = []
    for scene, pair in pairs.items():
        if set(pair) != set(BODIES):
            continue
        a, b = (pair[k] for k in BODIES)
        check(a["layout_origin"] == b["layout_origin"] and a["state_sha256"] == b["state_sha256"],
              "Archive source arms disagree on physical layout")
        key = hashlib.sha256(("plate-transfer-v1/" + scene).encode()).hexdigest()
        candidates.append((key, scene, pair))
    candidates.sort(key=lambda item: item[0])
    check(len(candidates) == coverage["counts"]["generated"]["shared_panda_layouts"], "Incomplete matched archive")
    for key in ("layout_seed", "movable_xy_hash", "initial_state_sha256"):
        check(len({p["Panda"]["layout_origin"][key] for _, _, p in candidates}) == len(candidates), f"Duplicate {key}")
    selection = [dict(rank=i, order_hash=key, scene_hash=scene, layout_seed=p["Panda"]["layout_seed"],
                      movable_xy_hash=p["Panda"]["layout_origin"]["movable_xy_hash"],
                      generated_row_fingerprint=p["Panda"]["generated_row_fingerprint"],
                      tapes={body: p[body]["sha256"] for body in BODIES})
                 for i, (key, scene, p) in enumerate(candidates)]
    write(out / "selection.json", dict(archive=str(SOURCE), archive_sha256=coverage["sha256"],
          order="ascending SHA256(ASCII plate-transfer-v1/ + scene_hash); slots 0:16 train, 16:20 heldout, reserves 20:",
          candidates=selection))
    return candidates, manifest["demos"], coverage["index_sha256"]


def replay(out, ep, bank, indexes, expected_hashes, gpu):
    from arm_env import ArmEnv
    from OpenGL import GL

    body, seed = ep["body"], ep["layout_seed"]
    if bank not in indexes:
        check(sha(bank / "index.json") == expected_hashes[str(bank.relative_to(ARCHIVE))], "Collector index hash")
        idx = json.loads((bank / "index.json").read_text())
        check(Path(idx["command"][0]).name == "generated_collect.py", "Generated expert collector family")
        indexes[bank] = ({r["file"]: r for r in idx["attempts"] if r.get("file")}, idx["revision"])
    records, teacher_revision = indexes[bank]
    record = records[ep["file"]]
    check(record["accepted"] and record["done"] and record["native_success"] and record["terminal_reward"] == 1
          and record["same_initial_state"] and record["route"] == "rendered_commands"
          and record["steps"] == record["search_steps"] == ep["frames"]
          and record["layout_origin"] == ep["layout_origin"], "Invalid positive teacher receipt")
    tape, draft, state = (bank / ep[k] for k in ("file", "search_file", "state_file"))
    check(sha(tape) == ep["sha256"] == record["sha256"] and sha(draft) == ep["search_sha256"] == record["search_sha256"]
          and sha(state) == ep["state_sha256"], "Teacher input bytes changed")
    actions = np.load(draft, allow_pickle=False)
    with np.load(tape, allow_pickle=False) as source:
        check(np.array_equal(source["actions"], actions) and source["native_success"].item()
              and source["body"].item() == body and source["prompt"].item() == PROMPT
              and source["seed"].item() == seed and not source["rewards"][:-1].any() and source["rewards"][-1] == 1
              and actions.shape == (ep["frames"], 7), "Rendered source/search controls or reward mismatch")

    result = dict(body=body, layout_seed=seed, scene_hash=ep["scene_fingerprint"], archive=str(tape),
                  archive_sha256=ep["sha256"], command_sha256=ep["search_sha256"], state_file_sha256=ep["state_sha256"],
                  generated_row_fingerprint=ep["generated_row_fingerprint"], movable_xy_hash=ep["layout_origin"]["movable_xy_hash"],
                  teacher_revision=teacher_revision, recipe=record["recipe"], archive_steps=len(actions), success=False)
    with ArmEnv(body, "libero_goal", 5) as arm:
        arm._states = (np.load(state, allow_pickle=False),)
        try:
            packet = arm.reset(0, seed)
        except ValueError as exc:
            if str(exc).startswith(RESET_REJECTIONS):
                result["failure"] = str(exc)
                return result
            raise
        phy = packet["environment_reproducibility"]
        origin = ep["layout_origin"]
        check(phy["released_row_fingerprint"] == origin["initial_state_sha256"] and phy["scene_fingerprint"] == origin["scene_hash"]
              and all(phy[k] == origin[k] for k in ("suite", "task_id", "source_pin", "bddl_sha256", "manifest_sha256"))
              and phy["perturbation_plan"]["case_id"] == "BASE" and phy["seed"] == seed and packet["max_steps"] == 300
              and packet["model_identity"] == record["reset"]["model_identity"] and packet["language_instruction"] == PROMPT,
              "Native generated start, model or task mismatch")
        renderer = [GL.glGetString(k).decode() for k in (GL.GL_VENDOR, GL.GL_RENDERER)]
        if gpu:
            check("NVIDIA" in renderer[0] and "4090" in renderer[1], f"Wrong native renderer {renderer}")
        else:
            check("llvmpipe" in renderer[1].lower(), f"Wrong CPU renderer {renderer}")
        result.update(renderer=renderer, source_pin=phy["source_pin"], bddl_sha256=phy["bddl_sha256"],
                      archive_settled=ep["initial_state_fingerprint"], replay_settled=phy["initial_state_fingerprint"],
                      post_settle_scene=phy["post_settle_scene_fingerprint"], model_xml=packet["model_identity"]["xml_sha256"])
        observations = {k: [] for k in KEYS}
        issued, rewards, histories = [], [], []
        history = PublicHistory()
        obs = packet["observation"]
        for action in actions:
            for k in KEYS:
                observations[k].append(obs[k].copy())
            histories.append(history.vector())
            packet = arm.step(action)
            actual = packet["issued_action"]
            check(np.array_equal(action, actual) and packet["language_instruction"] == PROMPT, "Issued command/task mismatch")
            issued.append(actual)
            rewards.append(packet["reward"])
            after = packet["observation"]
            history.append(actual, obs, after)
            obs = after
            if packet["terminated"] or packet["truncated"]:
                break
        for k in KEYS:
            observations[k].append(obs[k].copy())
        histories.append(history.vector())
    n = len(issued)
    result.update(steps=n, terminal_reward=rewards[-1], native_return=sum(rewards), native_success=packet["success"],
                  terminated=packet["terminated"], truncated=packet["truncated"], exact_issued_prefix=True)
    if not (packet["success"] and packet["terminated"] and not packet["truncated"] and
            rewards[-1] == 1 and not any(rewards[:-1]) and n >= HORIZON + 11):
        result["failure"] = "Native terminal criterion or number of distinct full windows"
        return result
    path = out / f"{seed}-{body}.h5"
    with h5py.File(path, "x") as f:
        f.attrs.update(body=body, prompt=PROMPT, seed=seed, scene_hash=result["scene_hash"],
                       success=True, terminated=True, archive_sha256=ep["sha256"])
        for k, samples in observations.items():
            arr = np.stack(samples)
            if k in IMAGE_KEYS:
                check(arr.shape == (n + 1, 256, 256, 3) and arr.dtype == np.uint8 and np.any(arr[0] != arr[-1]),
                      f"Native camera invalid: {k}")
            else:
                check(arr.shape == (n + 1, {POS: 3, QUAT: 4, FINGERS: 2}[k]) and np.isfinite(arr).all(), k)
            f.create_dataset(k, data=arr, compression="lzf", chunks=(1, *arr.shape[1:]))
        f.create_dataset("history", data=histories)
        f.create_dataset("actions", data=issued)
        f.create_dataset("rewards", data=rewards)
    result.update(success=True, file=path.name, sequence_sha256=sha(path))
    return result


def main():
    from pack import publish

    parser = argparse.ArgumentParser()
    parser.add_argument("out", type=Path)
    parser.add_argument("pin")
    parser.add_argument("--cpu-smoke", action="store_true")
    args = parser.parse_args()
    out = args.out
    check(0 < len(os.sched_getaffinity(0)) <= 2, "CPU allocation")
    candidates, banks, hashes = pool(out)
    (out / "tapes").mkdir()
    enrolled = []
    indices = {}
    reserve = 20
    slots = range(1 if args.cpu_smoke else 20)
    for slot in slots:
        rank = slot
        while rank < len(candidates):
            _, scene, pair = candidates[rank]
            episodes = {}
            for body in BODIES:
                ep = pair[body]
                result = replay(out / "tapes", ep, Path(banks[ep["bank"]]), indices, hashes, not args.cpu_smoke)
                episodes[body] = result
                write(out / "tapes" / f"{ep['layout_seed']}-{body}.json", result)
                print(slot, rank, body, ep["seed"], result["success"], result.get("steps"), flush=True)
            ok = all(r["success"] for r in episodes.values())
            audit = dict(slot=slot, rank=rank, layout_seed=ep["layout_seed"], scene_hash=scene, accepted=ok)
            with (out / "attempts.jsonl").open("a") as ledger:
                ledger.write(json.dumps(audit) + "\n")
            if ok:
                enrolled.append(dict(**audit, split="train" if slot < 16 else "heldout", episodes=episodes,
                    movable_xy_hash=ep["layout_origin"]["movable_xy_hash"], generated_row_fingerprint=ep["generated_row_fingerprint"]))
                break
            if args.cpu_smoke:
                raise RuntimeError("CPU smoke did not reproduce the initial matched pair; see receipts")
            rank, reserve = reserve, reserve + 1
        else:
            raise RuntimeError("Matched source reserve exhausted")
    write(out / "enrollment.json", enrolled)
    if args.cpu_smoke:
        from pack import write_split
        write_split(out, "train", enrolled, "mixed", out / "tapes")
    else:
        publish(out, enrolled, args.pin)


if __name__ == "__main__":
    main()
