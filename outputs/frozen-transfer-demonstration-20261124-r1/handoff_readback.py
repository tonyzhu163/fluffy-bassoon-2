"""Independent CPU-only record reader. No simulator, OpenPI or author scoring imports."""
import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import tarfile
import time

import numpy as np


ROOT = Path("/mowenjie/fluffy-bassoon-2")
ROWS = tuple(range(40, 48))
DONORS = ("IIWA34", "IIWA35")
MODES = ("raw", "tuned", *DONORS)
EXPECTED_CLOSE = {"IIWA34": 106, "IIWA35": 119}
TASK = "open the top layer of the drawer and put the bowl inside"
RELEASE = "e5500c690c1498eca1b7050350654964190cdf70"
PHYSICS = "a8a997e7d06bb131d3f9710204c58e106f0f0904"
MANIFEST = "71a6e4e25a2a4a162844d77136e7bff8e6f69c839d2d8194620c4afeabde2bba"
BDDL = "695545a33399e12e978b1b5682b6ea2181f3289608cad3f063f5bf90f0b87e67"
BANK = "1cd5354ade9ab9f323a518515f961df96123b7daeca017df1f62d3e9ff18794f"
PRED = "c8420c2cb7d0cb38e6f0d73b3a20ce47997dbef7657b652d45eaaa8988eb6c0b"
WEIGHTS = "ec064cce32f5003e2e3979c5712d7d6385018f6bfcc59cb2161105881e70d170"
STUDY = "frozen-transfer-demonstration-20261124-r1"
IDENTITY = dict(yaw_deg=0.0, xy_scale=1.0, gripper_sign=1)
ARCHIVES = {
    "frozen-transfer-cue-20261124-r1/evidence/native01/native-tapes.tar.gz":
        "637383c46eabaf3f8aaf776a9534e84aef152a277ab3ca9ac191c5a68c94fd5c",
    "frozen-transfer-donor-20261124-r1/inputs/raw-tapes.tar.gz":
        "5b8b787f7715a5c5ba5eb3a1f16a25d4463222589ca99cac1fd17901d37ee750",
}
MEMBERS = {"IIWA34": "a610024dc57b1f8783d3dfe000454414db48b1915f858f3759abb9d700b754c9",
           "IIWA35": "c5d8b4ba1a858b1f1c2b245cb1ec447d09439580ff3a1a404724042d4fe37d34"}
PUBLIC = {"agentview_image", "robot0_eye_in_hand_image", "robot0_joint_pos", "robot0_joint_vel",
          "robot0_eef_pos", "robot0_eef_quat_xyzw", "robot0_gripper_qpos"}
INPUTS = {"observation/image", "observation/wrist_image", "observation/state"}


def check(ok, message):
    if not ok:
        raise ValueError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def verified(path, digest):
    data = path.read_bytes()
    check(sha(data) == digest, f"checksum for {path}")
    return data


def read(path):
    return json.loads(path.read_bytes())


def fingerprint(array):
    a = np.ascontiguousarray(array)
    return sha(repr((a.shape, a.dtype.str)).encode() + a.tobytes())


def git_pin(directory):
    return subprocess.check_output(["git", "-C", str(directory), "rev-parse", "HEAD"], text=True).strip()


def source_tapes(outputs):
    blobs = {}
    for name, digest in ARCHIVES.items():
        path = outputs / name
        verified(path, digest)
        with tarfile.open(path) as archive:
            for donor, expected in MEMBERS.items():
                member = f"gpu01/episodes/IIWA-{donor[-2:]}-raw.json"
                raw = archive.extractfile(member).read()
                check(sha(raw) == expected and blobs.setdefault(donor, raw) == raw, member)
    tapes, info = {}, {}
    for donor, data in blobs.items():
        obj = json.loads(data)
        rec = obj["native"]
        tape = np.asarray([step for block in obj["calls"] for step in block["sent"]], dtype="<f4")
        check(obj["body"]["arm"] == "IIWA" and obj["body"]["gripper"] == "PandaGripper"
              and obj["mode"] == "raw" and rec["inferred"] == IDENTITY and rec["success"]
              and obj["query"] == dict(arm="IIWA", suite="libero_goal", task_id=3, variant="BASE",
                                       init_state_id=int(donor[-2:]), seed=42710300 + int(donor[-2:]))
              and rec["released_prompt"] == TASK and not obj["fixture"]
              and np.isfinite(tape).all() and np.abs(tape).max() <= 1 and tape.shape == (rec["controls"], 7),
              f"archive source {donor}")
        first = next(i for i, v in enumerate(tape[:, 6]) if v > 0)
        cut = 5 * (first // 5)
        check(first == EXPECTED_CLOSE[donor] and np.all(tape[:first, 6] < 0), f"grip {donor}")
        tapes[donor] = tape
        info[donor] = dict(member_sha256=sha(data), source_controls=len(tape), initial_grip=float(tape[0, 6]),
                           previous_grip=float(tape[first-1, 6]), first_positive_grip=float(tape[first, 6]),
                           first_positive_index=first, cut=cut, global_call=cut//5,
                           prefix_le_f32_sha256=sha(tape[:cut].tobytes()), full_le_f32_sha256=sha(tape.tobytes()))
    return tapes, info


def references(outputs):
    import torch

    release = ROOT / "sources/libero-pro2-e5500c69"
    shim = ROOT / "outputs/frozen-vla-native-transfer-20261123-r1" / f"src-{PHYSICS}"
    check(git_pin(release) == RELEASE and git_pin(shim) == PHYSICS, "release or native code pin")
    manifest = json.loads(verified(release/"assets/task_manifest.json", MANIFEST))
    task, = (e for e in manifest["tasks"] if (e["suite"], e["base_task_id"], e["case_id"]) == ("libero_goal", 3, "BASE"))
    bddl = verified(release/"assets"/task["bddl_path"], BDDL)
    check(task["max_steps"] == 300 and task["bddl_sha256"] == BDDL and
          b"(In akita_black_bowl_1 wooden_cabinet_1_top_region)" in bddl, "task reward predicate")
    bank = release/"simulator-source/libero/libero/init_files/libero_goal/open_the_top_drawer_and_put_the_bowl_inside.pruned_init"
    verified(bank, BANK)
    bank_data = torch.load(bank, map_location="cpu", weights_only=False)
    check(bank_data.shape == (50, 79), "layout bank")
    rows = {r: sha(b'{"dtype":"<f8","shape":[79]}\0' + np.asarray(bank_data[r], "<f8").tobytes()) for r in ROWS}
    check(len(set(rows.values())) == 8, "eight distinct release rows")
    pred = json.loads(verified(outputs/"symmeta-perceiver-retry-20261123-r1/evidence/train02pro/predictions.json", PRED))
    result, = (r for r in pred["rows"] if (r["panel"], r["seed"], r["embodiment"], r["suite"], r["task_id"]) ==
               ("gpu06kinova3", 41, "fam_nom", "libero_goal", 3))
    archive = outputs/"symmeta-fk-crossarm-20261123-r1/evidence/gpu06kinova3-cases.json.gz"
    support = json.loads(gzip.decompress(verified(archive, "74587eaaf87ce69d733083b068f27b46fc62619c3a7ffe2ffe8248816949783e")))
    entry, = (r for r in support["rows"] if r["index"] == result["index"])
    check(result["index"] == 24 and entry["support_layout"] == 11 and entry["support_seed"] == 31810311
          and entry["support_sha256"] == result["tape_sha256"], "held-out support identity")
    tape = ROOT/"outputs/xemb-icml-campaign-20261121-r1/gpu06kinova3/live/shard-2/support/00-fam_nom-passive.npz"
    verified(tape, result["tape_sha256"])
    with np.load(tape, allow_pickle=False) as arr:
        check(np.array_equal(arr["valid"], np.arange(24) < 3), "support block mask")
        if "chunk_steps" in arr:
            check(np.array_equal(arr["chunk_steps"], arr["valid"]*5), "support controls")
    return rows, dict(panel="gpu06kinova3", index=24, reader_seed=41, suite="libero_goal", task_id=3, shift="fam_nom",
                      layout=11, seed=31810311, controls=15, tape_sha256=result["tape_sha256"], physics=entry["physics"],
                      predictions_sha256=PRED, tuned=result["tuned"])



def runtime_sources():
    """Compare the deployed simulator's action/reward path to the released package."""
    base = ROOT/"setups/libero-meta-learning-20261117-r1"
    release = ROOT/"sources/libero-pro2-e5500c69"
    files = {}
    for name in ("env_wrapper.py","bddl_base_domain.py","problems/libero_tabletop_manipulation.py"):
        relative = Path("simulator-source/libero/libero/envs")/name
        files[name] = sha((release/relative).read_bytes())
        verified(base/relative, files[name])
    model_path = base/"env/lib/python3.11/site-packages/robosuite/models/grippers/panda_gripper.py"
    files["panda_gripper.py"] = "a3760a8fc4599fa6f79914304b82466f3c1bda1a945943ca693bb57f00dc59eb"
    verified(model_path, files["panda_gripper.py"])
    openpi = Path("/mowenjie/fluffy-bassoon/third_party/openpi-215abfb")
    check(git_pin(openpi) == "215abfb217dbac7d5f1273282331b9b1866c0479", "OpenPI checkout")
    for name in ("policies/policy.py","models/pi0.py"):
        files[name] = sha((openpi/"src/openpi"/name).read_bytes())
    return files

def full_tape_condition(outputs, tapes):
    evidence = outputs/"frozen-transfer-donor-validation-20261124-r1/evidence"
    manifest = verified(evidence/"BAIGE-SHA256SUMS", "3c0f788bcd79ef9a70e392fc96f2620c7369b7836f68cfcb902048864239a3a1")
    hashes = {p: s for s, p in (l.split() for l in manifest.decode().splitlines())}
    output = {}
    for row in ROWS:
        key = f"cases/start-{row}.json"
        start = json.loads(verified(evidence/key, hashes[key]))
        modes = {}
        for donor in DONORS:
            name = f"cases/case-{row}-{donor}.json"
            obj = json.loads(verified(evidence/name, hashes[name]))
            n, length = obj["controls"], len(tapes[donor])
            reward, success, term, trunc = obj["final_reward_success_terminated_truncated"]
            check(obj["start_sha256"] == hashes[key] and obj["row"] == row and obj["branch"] == donor
                  and obj["seed"] == 42710300+row and obj["tape_length"] == length
                  and obj["action_sha256"] == fingerprint(tapes[donor])
                  and obj["submitted_sha256"] == fingerprint(tapes[donor][:n])
                  and reward == obj["reward_sum"] == int(success) and term == success and not trunc
                  and 0 < n <= length and obj["preceding_zero_nonterminal_controls"] == n-1
                  and (obj["stop"],obj["first_success"]) == (("success", n) if success else ("tape_end", None))
                  and (success or n == length), f"CPU full-tape {row}/{donor}")
            modes[donor] = dict(success=int(success), controls=n, stop=obj["stop"])
        output[row] = dict(environment=start["environment"], modes=modes)
    return output


def contrasts(rows):
    """Exact paired empirical bootstrap by multinomial enumeration of win/loss/tie draws."""
    result = {}
    n = len(rows)
    for a, b in [(a,b) for a in DONORS for b in ("tuned","raw")] + [("tuned","raw")]:
        deltas = {r: v["episodes"][a]["success"]-v["episodes"][b]["success"] for r,v in rows.items()}
        counts = Counter(deltas.values())
        hist = Counter()
        for gain in range(n+1):
            for loss in range(n+1-gain):
                ties = n-gain-loss
                ways = math.comb(n,gain)*math.comb(n-gain,loss)
                hist[gain-loss] += ways * counts[1]**gain * counts[-1]**loss * counts[0]**ties
        check(sum(hist.values()) == n**n, "bootstrap count")
        cuts = []
        for numerator in (1,39):
            cumulative = 0
            for delta, frequency in sorted(hist.items()):
                cumulative += frequency
                if 40*cumulative >= n**n*numerator:
                    cuts.append(100*delta/n)
                    break
        result[a+"-"+b] = dict(rescues=[r for r,d in deltas.items() if d==1], losses=[r for r,d in deltas.items() if d==-1],
                                 ties=counts[0], effect_pp=100*sum(deltas.values())/n, paired_layout_bootstrap_95_pp=cuts)
    return result


def tuned_actions(five, correction):
    # Always convert the five proposed rows. Native termination can shorten the executed block.
    a = np.clip(five, -1, 1)
    theta = math.radians(-correction["yaw_deg"])
    c, s = np.float32(math.cos(theta)), np.float32(math.sin(theta))
    inv = np.float32(1 / np.clip(correction["xy_scale"], .25, 4))
    matrix = np.array([[c, s], [-s, c]], np.float32)
    products = a[:, 0, None]*matrix[0] + a[:, 1, None]*matrix[1]
    variants = {}
    for name, xy in (("two_float32_products", products), ("reader_blas", a[:, :2] @ matrix)):
        v = a.copy()
        v[:, :2] = xy*inv
        v[:, 2] *= inv
        v[:, 6] *= np.float32(correction["gripper_sign"])
        variants[name] = np.clip(v, -1, 1)
    return variants


def checked_episode(obj, row, mode, fixture, pin, support, row_hash, noise, tapes, info, kernels):
    name = f"{row}/{mode}"
    env, body, rec, calls = (obj[k] for k in ("env", "body", "native", "calls"))
    reset = rec["reset"]
    case = dict(suite="libero_goal", task_id=3, variant="BASE", arm="Kinova3", init_state_id=row, seed=42710300+row)
    check(obj["query"] == case and obj["pin"] == pin and obj["fixture"] is fixture and obj["mode"] == mode
          and obj["accepted_reset"] in (1, 2, 3) and obj["support"] == support and obj["shape"] == [10,32]
          and obj["weight_manifest"] == WEIGHTS and obj["release_row"] == env["released_row_fingerprint"] == row_hash,
          name+" identity")
    check(all(env[k] == case[k] for k in ("suite", "task_id", "init_state_id", "seed"))
          and env["source_pin"] == RELEASE and env["manifest_sha256"] == MANIFEST and env["bddl_sha256"] == BDDL
          and env["perturbation_plan"]["case_id"] == "BASE" and env["perturbation_plan"]["operator"] == "none"
          and reset["scene_fingerprint"] == env["scene_fingerprint"] and reset["physics"] == env["initial_state_fingerprint"]
          and reset["perturbation_plan"] == env["perturbation_plan_fingerprint"] and reset["physics"] != support["physics"]
          and reset["worker_step_index"] == 0 and reset["horizon"] == 300 and reset["prompt"] == rec["released_prompt"] == TASK,
          name+" nominal drawer reset")
    schema = reset["action_schema"]
    check(body["arm"] == reset["arm"] == "Kinova3" and body["model_class"] == reset["model_class"] == "MountedKinova3"
          and body["joint_count"] == reset["joint_count"] == 7 and body["controller"] == schema["controller"] == "OSC_POSE"
          and body["control_freq"] == schema["control_freq"] == 20 and body["gripper"] == "PandaGripper"
          and body["robosuite"] == "1.4.0" and body["mujoco"] == "3.2.3" and
          body["gripper_xml_sha256"] == "d9d68123909d6910488310d0fae4e358ed4ac831c6d0957f3e30893a5412baac"
          and body["cameras"] == ["agentview", "robot0_eye_in_hand"] and schema["control_mode"] == "relative"
          and schema["dtype"] == "float32" and schema["shape"] == [7] and schema["low"] == [-1]*7 and schema["high"] == [1]*7,
          name+" physics")
    check("llvmpipe" in reset["renderer"].lower() if fixture else reset["renderer_vendor"] == "NVIDIA Corporation",
          name+" renderer")
    correction = IDENTITY if mode == "raw" else support["tuned"]
    check(rec["inferred"] == correction and reset["noise_seed"] == case["seed"]+817117 and reset["noise_sha256"] == fingerprint(noise),
          name+" calibration/noise")
    cut = info[mode]["cut"] if mode in DONORS else 0
    n = rec["controls"]
    check(1 <= n <= 300 and len(calls) == (n+4)//5, name+" record length")
    public = reset["public"]
    rewards, prefix = [], bytearray()
    action_digest = hashlib.sha256()
    handback = None
    converted, changed = 0, 0
    for index, call in enumerate(calls):
        t = 5*index
        count = min(5, n-t)
        ctx = f"{name} t={t}"
        check(len(rewards) == t == call["t"] and call["prefix"] is (t < cut) and call["public"] == public
              and set(public) == set(call["after_public"]) == PUBLIC and call["prompt"] == TASK
              and set(call["inputs"]) == INPUTS and call["inputs"] == call["expected_inputs"]
              and call["noise"] == fingerprint(noise[index]), ctx+" input lineage and global noise")
        expected_motor = "fixture" if fixture and (mode not in DONORS or t != cut) else "pi05_libero"
        check(call["motor"] == expected_motor, ctx+" frozen motor")
        if index == 0:
            check(call["inputs"] == reset["motor_frame0"] and call["proposal"] == rec["opening_proposal_sha256"],
                  ctx+" opening input/proposal")
        plan = np.asarray(call["first_five"], dtype="<f4")
        steps = call["native_steps"]
        sent = np.asarray([s["action"] for s in steps], dtype="<f4")
        check(plan.shape == (5,7) and sent.shape == (count,7) and np.isfinite(plan).all()
              and np.isfinite(sent).all() and np.abs(sent).max() <= 1, ctx+" command shape/bounds")
        sent_bytes = sent.tobytes()
        action_digest.update(sent_bytes)
        raw_bytes = np.clip(plan[:count], -1, 1).tobytes()
        if t < cut:
            check(sent_bytes == tapes[mode][t:t+count].tobytes(), ctx+" literal source bytes")
            prefix.extend(sent_bytes)
        elif mode == "raw":
            check(sent_bytes == raw_bytes, ctx+" raw outgoing controls")
        else:
            options = tuned_actions(plan, correction)
            kernels.intersection_update(k for k, a in options.items() if a[:count].tobytes() == sent_bytes)
            check(kernels, ctx+" whole-five tuned outgoing controls")
            converted += count
            changed += sum(x.tobytes() != y.tobytes() for x, y in zip(sent,np.clip(plan[:count],-1,1)))
        for j, s in enumerate(steps, start=1):
            terminal = t+j == n
            check(s["index"] == t+j and type(s["success"]) is bool and s["reward"] in (0,1)
                  and s["reward"] == int(s["success"]) and s["terminated"] is s["success"]
                  and s["truncated"] is (t+j == 300 and not s["success"])
                  and (terminal or not (s["terminated"] or s["truncated"])), ctx+" native steps/reward")
            rewards.append(s["reward"])
        if mode in DONORS and t == cut:
            changed_keys = [k for k in sorted(INPUTS) if call["inputs"][k] != calls[index-1]["inputs"][k]]
            check(call["inputs"] != calls[0]["inputs"] and bool(changed_keys) and public != calls[index-1]["public"],
                  ctx+" stale handback")
            handback = dict(global_call=index, first_native_control=t+1, last_native_control=t+count,
                            public_sha256=sha(json.dumps(public,sort_keys=True).encode()), inputs=call["inputs"],
                            inputs_changed_since_previous_query=changed_keys,
                            noise=call["noise"], proposal=call["proposal"], outgoing_le_f32_sha256=sha(sent_bytes),
                            differs_from_uncorrected=sent_bytes != raw_bytes)
        public = call["after_public"]
    last = calls[-1]["native_steps"][-1]
    check(all(rec[k] is last[k] for k in ("success","terminated","truncated")) and rewards == rec["native_rewards"]
          and rewards[:-1] == [0]*(n-1) and sum(rewards) == rec["native_return"] == int(rec["success"]), name+" endpoint")
    if fixture:
        check(n == cut+5 and not (last["terminated"] or last["truncated"]), name+" CPU fixture endpoint")
    else:
        check(last["terminated"] != last["truncated"] and (last["success"] or n == 300), name+" incomplete episode")
    return dict(success=int(last["success"]), controls=n, terminal="fixture_cap" if fixture else
                ("success" if last["success"] else "horizon"), calls=len(calls), reset_attempt=obj["accepted_reset"],
                donor_prefix_controls=len(prefix)//28, prefix_le_f32_sha256=sha(prefix) if prefix else None,
                handback=handback, tuned_controls=converted, tuned_controls_different_from_raw=changed,
                all_actions_le_f32_sha256=action_digest.hexdigest())



def challenge_fixture(obj, support, row_hash, bank, tapes, info):
    """Change one recorded link at a time on copies of a genuine qualification trace."""
    labels = {}
    for error in ("source_lsb", "stale_input", "restart_noise", "bypass_tuned", "fake_reward"):
        trial = deepcopy(obj)
        row = trial["query"]["init_state_id"]
        mode = trial["mode"]
        cells = trial["calls"]
        index = info[mode]["global_call"]
        hand = cells[index]
        if error == "source_lsb":
            action = cells[0]["native_steps"][0]["action"]
            bits = np.asarray(action[:1],dtype="<f4").view("<u4")
            bits[0] ^= np.uint32(1)
            action[0] = float(bits.view("<f4")[0])
        elif error == "stale_input":
            hand["inputs"] = hand["expected_inputs"] = cells[index-1]["inputs"]
        elif error == "restart_noise":
            hand["noise"] = cells[0]["noise"]
        elif error == "bypass_tuned":
            for step,a in zip(hand["native_steps"],hand["first_five"]):
                step["action"] = np.clip(a,-1,1).tolist()
        else:
            hand["native_steps"][0]["reward"] = 1.0
        try:
            checked_episode(trial,row,mode,True,trial["pin"],support,row_hash,bank,tapes,info,
                            {"two_float32_products","reader_blas"})
        except ValueError as exc:
            labels[error] = str(exc)
        else:
            raise ValueError(f"failed to detect {mode} {error}")
    return labels

def audit(args):
    started = time.monotonic()
    affinity = sorted(os.sched_getaffinity(0))
    check(os.uname().sysname == "Linux" and set(affinity) == set(map(int,args.cores.split(",")))
          and os.environ["CUDA_VISIBLE_DEVICES"] == "" and os.environ["JAX_PLATFORMS"] == "cpu", "CPU-only reader affinity")
    folder, src, fixture = args.folder, args.source, args.fixture
    check(folder.parent == ROOT/"outputs"/STUDY, "shared outputs directory")
    receipt = (folder/"exit.txt").read_text().split()
    check(receipt[0] == "0", "successful terminal receipt required")
    pin = git_pin(src)
    check(pin == args.pin and subprocess.run(["git","-C",str(src),"symbolic-ref","-q","HEAD"],capture_output=True).returncode == 1,
          "pinned detached checkout")
    common = [folder/"exit.txt", folder/"origin.json", folder/("qualified.json" if fixture else "result.json")]
    origin, official = read(common[1]), read(common[2])
    check(origin["pin"] == official["pin"] == pin and origin["fixture"] is fixture and official["fixture"] is fixture
          and official["status"] == "passed" and origin["shape"] == [10,32]
          and origin["release_bank_sha256"] == BANK and origin["policy"].endswith("/inputs/pi05-libero-openpi-20261027-r1"),
          "registered motor, release and episode summaries")
    if fixture:
        check(receipt == ["0"] and origin["cpu_affinity"] == [38,39], "fixture provenance")
        runtime = dict(host=origin["host"], qualification_affinity=origin["cpu_affinity"])
    else:
        launch = dict(x.split("=",1) for x in (folder/"launch.txt").read_text().split())
        probe = read(folder/"gpu.json")
        check(args.job and len(receipt) == 3 and launch["start"] == receipt[1] and launch["pin"] == pin
              and launch["host"] == origin["host"] == probe["hostname"] == args.job+"-master-0" and
              probe["allocation_type"] == "baige_container" and probe["status"] == "passed" and
              len(probe["devices"]) == 1 and probe["devices"][0]["cuda_name"] == "NVIDIA GeForce RTX 4090",
              "dedicated one-4090 job identity")
        wall = int(receipt[2])-int(receipt[1])
        check(0 < wall < 3600, "registered launcher budget")
        qual_dir = Path(launch["cpu"])
        qual = read(qual_dir/"qualified.json")
        check(qual_dir.parent == folder.parent and (qual_dir/"exit.txt").read_text().strip()=="0" and
              qual["fixture"] is True and qual["pin"] == pin and qual["rows"][0]["handoffs"] == dict.fromkeys(DONORS, True),
              "same-pin qualification")
        common.extend([folder/"launch.txt",folder/"gpu.json"])
        runtime = dict(host=origin["host"], job=args.job, device=probe["devices"][0]["cuda_name"], qualification=str(qual_dir),
                       start_epoch=int(receipt[1]), end_epoch=int(receipt[2]), elapsed_seconds=wall)
    outputs = src/"outputs"
    tapes, info = source_tapes(outputs)
    row_hashes, support = references(outputs)
    check(origin["support"] == official["support"] == support and origin["release_rows"] == {str(r):h for r,h in row_hashes.items()},
          "source of support and layout fingerprints")
    cpu_full = full_tape_condition(outputs,tapes)
    rows_to_read = ROWS[:1] if fixture else ROWS
    files = {p.name for p in (folder/"episodes").glob("*.json")}
    check(files == {f"{r}-{m}.json" for r in rows_to_read for m in MODES}, "exact episode inventory")
    check(official["episodes"] == len(files) and official["paired_starts"] == len(rows_to_read) and
          [x["row"] for x in official["rows"]] == list(rows_to_read), "writer episode inventory")
    rows, artifacts = {}, {p.name:sha(p.read_bytes()) for p in common}
    kernels = {"two_float32_products", "reader_blas"}
    proposal_by_input = {}
    for r in rows_to_read:
        episodes = {}
        raw_group = {}
        noise = np.random.default_rng(42710300+r+817117).standard_normal((60,10,32)).astype("<f4")
        for m in MODES:
            name = f"episodes/{r}-{m}.json"
            data = (folder/name).read_bytes()
            artifacts[name] = sha(data)
            obj = raw_group[m] = json.loads(data)
            episodes[m] = checked_episode(obj,r,m,fixture,pin,support,row_hashes[r],noise,tapes,info,kernels)
            for cell in obj["calls"]:
                # Closed-loop policies may encounter different states after the first control.
                key = (cell["motor"],cell["prompt"],cell["noise"],tuple(sorted(cell["inputs"].items())))
                value = (cell["proposal"],np.asarray(cell["first_five"],"<f4").tobytes())
                check(proposal_by_input.setdefault(key,value) == value, f"{r}/{m} same input/noise produced different proposal")
        base = raw_group["raw"]
        first = base["calls"][0]
        reset = base["native"]["reset"]
        for m,obj in raw_group.items():
            check(all(obj[k] == base[k] for k in ("env","body","query","release_row","shape","support"))
                  and obj["native"]["reset"] == reset and all(obj["calls"][0][k] == first[k]
                  for k in ("public","prompt","noise","inputs","expected_inputs","proposal","first_five")), f"unpaired {r}/{m}")
        cpu_env = cpu_full[r]["environment"]
        check(all(cpu_env[k] == base["env"][k] for k in
                  ("suite","task_id","init_state_id","seed","source_pin","manifest_sha256","bddl_sha256","released_row_fingerprint",
                   "scene_fingerprint","perturbation_plan_fingerprint")), f"{r} CPU registered layout identity")
        for m in DONORS:
            hand = episodes[m]["handback"]
            if hand and not fixture:
                i = hand["global_call"]
                hand["same_index_comparators"] = {}
                for comparator in ("raw","tuned"):
                    calls = raw_group[comparator]["calls"]
                    if i < len(calls):
                        hand["same_index_comparators"][comparator] = dict(
                            noise_equal=calls[i]["noise"] == hand["noise"], inputs_equal=calls[i]["inputs"] == hand["inputs"],
                            proposal_equal=calls[i]["proposal"] == hand["proposal"])
        author_row, = (x for x in official["rows"] if x["row"] == r)
        check(author_row["physics"] == reset["physics"] and author_row["release_row"] == row_hashes[r] and
              author_row["success"] == {m:v["success"] for m,v in episodes.items()} and
              author_row["controls"] == {m:v["controls"] for m,v in episodes.items()} and
              author_row["handoffs"] == {m: episodes[m]["handback"] is not None for m in DONORS}, "writer row disagreement")
        rows[r] = dict(seed=42710300+r, release_row=row_hashes[r], scene=base["env"]["scene_fingerprint"],
                       physics=reset["physics"], post_settle_scene=base["env"]["post_settle_scene_fingerprint"],
                       noise_bank=reset["noise_sha256"], opening_proposal=first["proposal"],
                       cpu_full_tape_same_physics=reset["physics"] == cpu_env["initial_state_fingerprint"],
                       cpu_full_tape_same_post_settle=base["env"]["post_settle_scene_fingerprint"] == cpu_env["post_settle_scene_fingerprint"],
                       episodes=episodes)
    check(len({artifacts["episodes/"+f] for f in files}) == len(files), "duplicate complete episode bytes")
    results = {}
    if fixture:
        results["rejected_fixture_edits"] = {m: challenge_fixture(raw_group[m], support, row_hashes[r], noise, tapes, info) for m in DONORS}
    else:
        successes = {m:sum(row["episodes"][m]["success"] for row in rows.values()) for m in MODES}
        paired = contrasts(rows)
        check(successes == official["successes"] and paired == official["contrasts"], "paired endpoint disagreement")
        results = dict(successes=successes, contrasts=paired, bootstrap_ordered_layout_draws=len(ROWS)**len(ROWS))
    native_code = ROOT/"outputs/frozen-vla-native-transfer-20261123-r1"/f"src-{PHYSICS}"
    reviewed = [outputs/STUDY/x for x in ("PLAN.md","run.py","study.py","audited_worker.py","runtime.sh","job.sh","qualify.sh","launch.sh")]
    reviewed += [native_code/"outputs"/f for f in ("libero-xemb-ur5e-sysid-20261120-r1/ur5e_motor.py",
                 "libero-xemb-ur5e-sysid-20261120-r1/ur5e_sim_worker.py","libero-crossarm-20261120-r1/arm_env.py",
                 "libero-crossarm-20261120-r1/sim_reset.py")] + [native_code/"src/fluffy_bassoon/xemb/embodiment.py"]
    code_hashes = {str(p.relative_to(src if src in p.parents else native_code)):sha(p.read_bytes()) for p in reviewed}
    return dict(verified_utc=datetime.now(timezone.utc).isoformat(), fixture=fixture, native_pin=PHYSICS, release_pin=RELEASE,
                study_pin=pin, source=str(src), shared_attempt=str(folder), reader_host=os.uname().nodename, cpu_cores=affinity,
                audit_wall_seconds=round(time.monotonic()-started,3), audit_cpu_seconds=round(time.process_time(),3),
                runtime=runtime, records=len(files), queries=sum(v["calls"] for r in rows.values() for v in r["episodes"].values()),
                native_controls=sum(v["controls"] for r in rows.values() for v in r["episodes"].values()), rows=rows, **results,
                transform_arithmetic_matching_every_tuned_control=sorted(kernels), registered_support=support,
                donor_reference=info, donor_archives_sha256=ARCHIVES, episode_and_receipt_sha256=artifacts, code_sha256=code_hashes, simulator_openpi_sha256=runtime_sources(),
                separate_cpu_full_tape={r:v["modes"] for r,v in cpu_full.items()})


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("folder", type=Path)
    p.add_argument("--source", type=Path, required=True, help="detached checkout that wrote the attempt")
    p.add_argument("--pin", required=True)
    p.add_argument("--cores", default="46,47")
    p.add_argument("--job", help="scheduler job ID from the parent's receipt")
    p.add_argument("--fixture", action="store_true")
    print(json.dumps(audit(p.parse_args()), indent=2, allow_nan=False))
