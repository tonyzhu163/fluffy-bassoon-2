"""Read completed drawer-cue tapes on Linux CPU 38/39. No policy or simulator imports."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import gzip
import hashlib
from itertools import product
import json
import math
import os
from pathlib import Path
import subprocess
import time

import numpy as np


ARMS = ("Kinova3", "IIWA")
LAYOUTS = (34, 35, 36, 37)
MODES = ("raw", "tuned", "clock", "visual")
BASELINES = ("tuned", "clock", "raw")
TASK = "open the top layer of the drawer and put the bowl inside"
CUE = "put the bowl in the top drawer"
PIN = "e15eefb3f76e475acdf0795e2e17ba0e1ee46fae"
RELEASE = "e5500c690c1498eca1b7050350654964190cdf70"
MANIFEST = "71a6e4e25a2a4a162844d77136e7bff8e6f69c839d2d8194620c4afeabde2bba"
MODEL = "ca8906c86514d8917c1743726929c1e70aaf1f45e07de8892574c58a9ddf7008"
WEIGHTS = "ec064cce32f5003e2e3979c5712d7d6385018f6bfcc59cb2161105881e70d170"
PREDICTIONS = "c8420c2cb7d0cb38e6f0d73b3a20ce47997dbef7657b652d45eaaa8988eb6c0b"
BANK = "1cd5354ade9ab9f323a518515f961df96123b7daeca017df1f62d3e9ff18794f"
ROOT = Path("/mowenjie/fluffy-bassoon-2")
STUDY = "frozen-transfer-cue-20261124-r1"


def check(condition, message):
    if not condition:
        raise ValueError(message)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def fingerprint(a):
    return digest(repr((a.shape, a.dtype.str)).encode() + a.tobytes(order="C"))


def read(path):
    return json.loads(path.read_bytes())


def references(src):
    import torch

    check(subprocess.check_output(["git", "-C", src, "rev-parse", "HEAD"], text=True).strip() == PIN,
          "deployed revision")
    bank = (ROOT / "sources/libero-pro2-e5500c69/simulator-source/libero/libero/init_files/libero_goal"
            / "open_the_top_drawer_and_put_the_bowl_inside.pruned_init")
    check(digest(bank.read_bytes()) == BANK, "release bank checksum")
    states = torch.load(bank, map_location="cpu", weights_only=False)
    check(states.shape == (50, 79), "released state matrix")
    row_hashes = {i: digest(b'{"dtype":"<f8","shape":[79]}\0' + np.asarray(states[i], "<f8").tobytes())
                  for i in LAYOUTS}
    check(len(set(row_hashes.values())) == 4, "four distinct release rows")
    outputs = src / "outputs"
    model = outputs / "frozen-vla-phase-transfer-20261123-r1/evidence/model.json"
    pred = outputs / "symmeta-perceiver-retry-20261123-r1/evidence/train02pro/predictions.json"
    check(digest(model.read_bytes()) == MODEL and digest(pred.read_bytes()) == PREDICTIONS, "archived models")
    supports = {}
    for arm, panel in zip(ARMS, ("gpu06kinova3", "gpu09iiwa"), strict=True):
        case, = [x for x in read(pred)["rows"] if (x["panel"], x["seed"], x["suite"], x["task_id"], x["embodiment"])
                 == (panel, 41, "libero_goal", 3, "fam_nom")]
        archive = outputs / f"symmeta-fk-crossarm-20261123-r1/evidence/{panel}-cases.json.gz"
        entry, = [x for x in json.loads(gzip.decompress(archive.read_bytes()))["rows"] if x["index"] == case["index"]]
        check(entry["support_sha256"] == case["tape_sha256"] and entry["support_layout"] == 11
              and entry["support_seed"] == 31810311 and entry["index"] == 24, "archived support")
        supports[arm] = dict(panel=panel, controls=15, shift="fam_nom", suite="libero_goal", task_id=3,
                             reader_seed=41, index=24, layout=11, seed=31810311, physics=entry["physics"],
                             predictions_sha256=PREDICTIONS, tape_sha256=entry["support_sha256"], tuned=case["tuned"])
    return row_hashes, supports, read(model)["clock"]


def issued(plan, mode, correction, cpu_fixture):
    commands = np.clip(np.array(plan, dtype=np.float32), -1, 1)
    if mode != "raw":
        theta = math.radians(-correction["yaw_deg"])
        c, s = math.cos(theta), math.sin(theta)
        axes = np.array([[c, s], [-s, c]], dtype=np.float32)
        inverse = np.float32(1 / correction["xy_scale"])
        # The qualifier contracts the two-term dot on the dev host. The 4090
        # job host rounded both float32 products separately. Match each tape
        # exactly without an action tolerance.
        if cpu_fixture:
            xy = commands[:, :2] @ axes
        else:
            xy = commands[:, 0, None]*axes[0] + commands[:, 1, None]*axes[1]
        commands[:, :2] = xy * inverse
        commands[:, 2] *= inverse
        commands[:, 6] *= correction["gripper_sign"]
    return np.clip(commands, -1, 1)


def episode(obj, arm, layout, mode, row_hash, support, clock, cpu):
    context = f"{arm}/{layout}/{mode}"
    n, env, body, calls = obj["native"], obj["env"], obj["body"], obj["calls"]
    reset = n["reset"]
    q = dict(arm=arm, suite="libero_goal", task_id=3, variant="BASE", init_state_id=layout, seed=42710300+layout)
    check(obj["query"] == q and obj["pin"] == PIN and obj["mode"] == mode and obj["fixture"] is cpu
          and obj["support"] == support and obj["accepted_reset"] in (1, 2, 3), context + " provenance")
    check(all(env[k] == q[k] for k in ("suite", "task_id", "init_state_id", "seed"))
          and env["source_pin"] == RELEASE and env["manifest_sha256"] == MANIFEST
          and env["perturbation_plan"]["case_id"] == "BASE" and env["released_row_fingerprint"] == row_hash
          and reset["scene_fingerprint"] == env["scene_fingerprint"] and reset["physics"] == env["initial_state_fingerprint"]
          and reset["perturbation_plan"] == env["perturbation_plan_fingerprint"]
          and reset["physics"] != support["physics"] and reset["worker_step_index"] == 0, context + " released reset")
    schema = reset["action_schema"]
    check(body["arm"] == reset["arm"] == arm and body["model_class"] == reset["model_class"] == "Mounted"+arm
          and body["joint_count"] == reset["joint_count"] == 7 and body["gripper"] == "PandaGripper"
          and body["controller"] == schema["controller"] == "OSC_POSE" and body["control_freq"] == schema["control_freq"] == 20
          and schema["control_mode"] == "relative" and schema["shape"] == [7] and schema["low"] == [-1]*7
          and schema["high"] == [1]*7 and body["cameras"] == ["agentview", "robot0_eye_in_hand"], context + " native body")
    check("llvmpipe" in reset["renderer"].lower() if cpu else reset["renderer_vendor"] == "NVIDIA Corporation",
          context + " renderer")
    identity = dict(yaw_deg=0, xy_scale=1, gripper_sign=1)
    check(n["inferred"] == (identity if mode == "raw" else support["tuned"]) and obj["model_sha"] == MODEL
          and obj["weight_manifest_sha"] == WEIGHTS and reset["prompt"] == n["released_prompt"] == TASK
          and reset["horizon"] == 300 and calls[0]["public"] == reset["public"] and calls[0]["inputs"] == reset["motor_frame0"]
          and calls[0]["proposal"] == n["opening_proposal_sha256"], context + " motor and initial input")
    seed = q["seed"] + 817117
    bank = np.random.default_rng(seed).standard_normal((60, *obj["shape"])).astype(np.float32)
    check(reset["noise_seed"] == seed and reset["noise_sha256"] == fingerprint(bank), context + " full noise bank")
    rewards, first_cross = [], {"clock": None, "visual": None}
    pulse = None
    for index, row in enumerate(calls):
        t = len(rewards)
        size = len(row["rewards"])
        check(0 < size <= 5 and row["t"] == index*5 == t and row["noise"] == fingerprint(bank[index])
              and row["clock"] == clock[min(index, len(clock)-1)] and 0 <= row["visual"] <= 1,
              f"{context} cadence, score or noise at {t}")
        check(set(row["inputs"]) == {"observation/image", "observation/state", "observation/wrist_image"}
              and set(row["public"]) == set(reset["public"]), f"{context} input schema at {t}")
        for gate in first_cross:
            if first_cross[gate] is None and row[gate] >= .5:
                first_cross[gate] = t
        fire = first_cross.get(mode) == t
        check(row["fire"] is fire and row["prompt"] == (CUE if fire else TASK), f"{context} policy text at {t}")
        if fire:
            check(pulse is None, context + " repeated cue")
            pulse = t
        plan = np.asarray(row["first_five"], np.float32)
        check(plan.shape == (5, 7) and np.isfinite(plan).all()
              and np.array_equal(issued(plan, mode, n["inferred"], cpu)[:size], row["sent"]), f"{context} issued controls at {t}")
        rewards.extend(row["rewards"])
        if index < len(calls)-1:
            check(size == 5 and not (row["terminated"] or row["truncated"]), context + " episode continued after end")
    length = len(rewards)
    success = rewards[-1]
    check(0 < length <= 300 and rewards[:-1] == [0]*(length-1) and success in (0, 1)
          and n["native_rewards"] == rewards and n["native_return"] == sum(rewards) == int(n["success"])
          and length == n["controls"] and n["terminated"] is calls[-1]["terminated"]
          and n["truncated"] is calls[-1]["truncated"], context + " native rewards")
    if cpu:
        check(length == 10 and not n["terminated"] and not n["truncated"], context + " qualification fixture")
    else:
        check(n["terminated"] == bool(success) and n["truncated"] != n["terminated"]
              and (success or length == 300), context + " native terminal")
    return dict(reward=int(success), controls=length, calls=len(calls), seconds=n["seconds"],
                reset_attempt=obj["accepted_reset"], cue_t=pulse, crossings=first_cross,
                max_visual=max(r["visual"] for r in calls), terminal_reward_control=length if success else None,
                restored_at=pulse+5 if pulse is not None and calls[-1]["t"] > pulse else None)


def compare(left, right):
    """Follow only the common history through the first changed issued control."""
    xs, ys = left["calls"], right["calls"]
    text_t = proposal_t = first_action = None
    common_calls = 0
    for x, y in zip(xs, ys):
        t = x["t"]
        check(all(x[k] == y[k] for k in ("t", "public", "inputs", "noise", "clock", "visual")),
              f"pre-divergence observation at {t}")
        common_calls += 1
        changed_text = x["prompt"] != y["prompt"]
        if changed_text and text_t is None:
            text_t = t
        if x["proposal"] != y["proposal"]:
            check(changed_text, f"unexplained proposal change at {t}")
            if proposal_t is None:
                proposal_t = t
        else:
            check(x["first_five"] == y["first_five"], "proposal hash/prefix mismatch")
        for j, (a, b) in enumerate(zip(x["sent"], y["sent"])):
            if a != b:
                check(changed_text or "raw" in (left["mode"], right["mode"]), "unexplained issued control")
                first_action = dict(query_t=t, control=t+j+1, left=a, right=b, noise=x["noise"], inputs=x["inputs"],
                                    left_prompt=x["prompt"], right_prompt=y["prompt"])
                break
            check(x["rewards"][j] == y["rewards"][j], "reward divergence on common issued prefix")
        if first_action is not None:
            break
        check(all(x[k] == y[k] for k in ("sent", "rewards", "terminated", "truncated")),
              "terminal divergence on common history")
    if first_action is None:
        check(len(xs) == len(ys), "length divergence without any changed action")
    return dict(first_prompt_difference_t=text_t, common_input_queries=common_calls,
                first_different_proposal_query_t=proposal_t, first_different_issued_action=first_action)


def timing(visual, clock):
    if visual == clock:
        order = "neither" if visual is None else "tied"
    elif visual is None or clock is None:
        order = "clock_only" if visual is None else "visual_only"
    else:
        order = "visual_before" if visual < clock else "visual_after"
    return dict(visual=visual, clock=clock, order=order)


def reductions(rows):
    """Enumerate all 256 ordered draws once per group, reusing each draw for all three contrasts."""
    endpoints, histograms = {}, {}
    for name, arms in (("Kinova3", ARMS[:1]), ("IIWA", ARMS[1:]), ("both", ARMS)):
        group = [r for r in rows if r["arm"] in arms]
        totals = {m: sum(r["episodes"][m]["reward"] for r in group) for m in MODES}
        n = len(group)
        deltas = {b: [(r["layout"], r["episodes"]["visual"]["reward"]-r["episodes"][b]["reward"]) for r in group]
                  for b in BASELINES}
        by_layout = {b: {l: sum(d for layout, d in ds if layout == l) for l in LAYOUTS} for b, ds in deltas.items()}
        draws = [tuple(sum(by_layout[b][l] for l in draw) for b in BASELINES) for draw in product(LAYOUTS, repeat=4)]
        contrasts = {}
        for i, b in enumerate(BASELINES):
            ds = [d for _, d in deltas[b]]
            ordered = sorted(draw[i]*100/n for draw in draws)
            contrasts["visual-minus-"+b] = dict(rescues=ds.count(1), losses=ds.count(-1), ties=ds.count(0),
                rescue_denominator=n-totals[b], loss_denominator=totals[b],
                layout_differences=[by_layout[b][l]/len(arms) for l in LAYOUTS], effect_pp=100*sum(ds)/n,
                layout_bootstrap_95_pp=[ordered[math.ceil(len(draws)*q)-1] for q in (.025, .975)])
        endpoints[name] = dict(n=n, successes=totals, contrasts=contrasts)
        histograms[name] = [dict(net_native_successes=list(key), draws=freq)
                            for key, freq in sorted(Counter(draws).items())]
    return endpoints, histograms


def fixture_challenges(raw, rows):
    """Exercise changed prompts/actions and coupled contrasts absent in ten-control physical fixtures."""
    from copy import deepcopy

    before = deepcopy(raw["calls"])
    after = deepcopy(before)
    after[1]["prompt"] = CUE
    after[1]["proposal"] = "fixture-different-proposal"
    after[1]["sent"][2][0] += .1
    change = compare(dict(mode="visual", calls=after), dict(mode="clock", calls=before))
    check(change["first_prompt_difference_t"] == change["first_different_proposal_query_t"] == 5
          and change["first_different_issued_action"]["control"] == 8, "chronological action fixture")
    after[1]["inputs"]["observation/state"] = "fixture-unmatched-input"
    try:
        compare(dict(mode="visual", calls=after), dict(mode="clock", calls=before))
    except ValueError as exc:
        check("pre-divergence observation" in str(exc), "wrong matched-input rejection")
    else:
        raise ValueError("accepted an unexplained pre-prompt observation difference")

    # The two arms have opposite effects on every layout. Breaking their pairing adds spurious spread.
    synthetic = []
    for r in rows:
        v = int((r["layout"] % 2 == 0) == (r["arm"] == "IIWA"))
        values = dict(visual=v, clock=v, tuned=1-v, raw=0)
        synthetic.append(dict(arm=r["arm"], layout=r["layout"], episodes={k: {"reward": x} for k, x in values.items()}))
    ends, hist = reductions(synthetic)
    joint, = hist["both"]
    check(joint == {"net_native_successes": [0, 0, 4], "draws": 256}, "joint layout fixture")
    for b, expected in zip(BASELINES, ((4, 4, 0), (0, 0, 8), (4, 0, 4)), strict=True):
        obs = ends["both"]["contrasts"]["visual-minus-"+b]
        check(tuple(obs[k] for k in ("rescues", "losses", "ties")) == expected, "paired count fixture")
    check(ends["IIWA"]["contrasts"]["visual-minus-tuned"]["layout_bootstrap_95_pp"] == [-100, 100],
          "arm-specific bootstrap tails")
    return dict(first_differing_query_t=5, first_differing_control=8, mismatched_pre_cue_state_rejected=True,
                opposed_arm_fixture_joint_counts=joint)


def audit(args):
    start = time.monotonic()
    cpu = args.fixture
    folder = args.folder
    check(os.sched_getaffinity(0) == {38, 39} and os.environ.get("CUDA_VISIBLE_DEVICES") == "", "CPU 38/39 only")
    receipt = list(map(int, (folder / "exit.txt").read_text().split()))
    check(receipt[0] == 0, "requires successful terminal receipt before opening tapes")
    runtime = {"exit": receipt[0]}
    if not cpu:
        launch = dict(item.split("=", 1) for item in (folder / "launch.txt").read_text().split())
        probe = read(folder / "gpu.json")
        check(len(receipt) == 3 and str(receipt[1]) == launch["start"] and receipt[2] >= receipt[1]
              and launch["pin"] == PIN and launch["host"] == probe["hostname"] == args.job+"-master-0"
              and Path(launch["cpu"]).name == "cpu03" and probe["status"] == "passed"
              and probe["allocation_type"] == "baige_container" and len(probe["devices"]) == 1
              and probe["devices"][0]["cuda_name"] == "NVIDIA GeForce RTX 4090", "registered launcher and device")
        runtime.update(job=args.job, host=launch["host"], gpu=probe["devices"][0]["cuda_name"],
                       start_epoch=receipt[1], end_epoch=receipt[2], launcher_seconds=receipt[2]-receipt[1])
    row_hashes, supports, clock = references(args.source)
    files = set((folder/"episodes").iterdir())
    check({p.name for p in files} == {f"{a}-{l}-{m}.json" for a, l, m in product(ARMS, LAYOUTS, MODES)},
          "exact registered set of 32 episode tapes")
    official = read(folder/("qualified.json" if cpu else "result.json"))
    check(official["episodes"] == 32 and official["paired_starts"] == 8 and official["pin"] == PIN
          and official["fixture"] is cpu and official["status"] == "passed" and len(official["rows"]) == 8,
          "official panel identity")
    rows, scene = [], {}
    for arm, layout in product(ARMS, LAYOUTS):
        group = {m: read(folder/f"episodes/{arm}-{layout}-{m}.json") for m in MODES}
        episodes = {m: episode(obj, arm, layout, m, row_hashes[layout], supports[arm], clock, cpu)
                    for m, obj in group.items()}
        base = group["raw"]
        reset = base["native"]["reset"]
        env = base["env"]
        shared = [env[k] for k in ("released_row_fingerprint", "scene_fingerprint", "bddl_sha256")]
        shared += [reset[k] for k in ("noise_sha256", "prompt", "horizon")]
        check(shared == scene.setdefault(layout, shared), "cross-body scene/noise")
        for mode, obj in group.items():
            check(obj["native"]["reset"] == reset and all(obj[k] == base[k] for k in ("env", "body", "support", "shape", "query")),
                  f"{arm}/{layout} physical pairing")
            check(all(obj["calls"][0][k] == base["calls"][0][k] for k in
                      ("prompt", "noise", "public", "inputs", "proposal", "first_five", "clock", "visual")), "paired opening draw")
            if mode != "raw":
                check(obj["calls"][0]["sent"] == group["tuned"]["calls"][0]["sent"], "paired tuned opening")
        changes = {a+"-"+b: compare(group[a], group[b]) for a, b in
                   (("visual", "tuned"), ("visual", "clock"), ("clock", "tuned"), ("visual", "raw"))}
        actual = timing(episodes["visual"]["cue_t"], episodes["clock"]["cue_t"])
        same = timing(**episodes["tuned"]["crossings"])
        if actual["visual"] == actual["clock"]:
            check(group["visual"]["calls"] == group["clock"]["calls"], "matched cue times changed trajectory")
        ref, = [r for r in official["rows"] if (r["arm"], r["layout"]) == (arm, layout)]
        check(ref["success"] == {m: v["reward"] for m, v in episodes.items()} and
              ref["controls"] == {m: v["controls"] for m, v in episodes.items()} and ref["actual"] == actual
              and ref["same_tuned_observations"] == same and ref["released_row_sha"] == row_hashes[layout],
              f"{arm}/{layout} official episode results or timing")
        for key in ("visual-tuned", "clock-tuned", "visual-clock"):
            for field in ("first_different_proposal_query_t", "first_different_issued_action"):
                check(changes[key][field] == ref["changes"][key][field], f"{arm}/{layout}/{key} official first change")
        rows.append(dict(arm=arm, layout=layout, seed=42710300+layout, episodes=episodes, actual=actual,
                         same_tuned_observations=same, first_changes=changes,
                         paired_start=dict(release_row=row_hashes[layout], scene=env["scene_fingerprint"],
                                           settled_scene=env["post_settle_scene_fingerprint"], physics=reset["physics"],
                                           full_noise=reset["noise_sha256"], opening_proposal=base["calls"][0]["proposal"])))
    all_eps = [e for r in rows for e in r["episodes"].values()]
    controls = sum(e["controls"] for e in all_eps)
    check(controls == official["controls"], "official control count")
    counts = {k: dict(Counter(r[k]["order"] for r in rows)) for k in ("actual", "same_tuned_observations")}
    endpoints, histograms = reductions(rows)
    if not cpu:
        check(endpoints == official["endpoints"] and all(official[k+"_timing_counts"] == v for k, v in counts.items()),
              "official paired endpoints or exposure counts")
    runtime["summed_episode_seconds"] = round(sum(e["seconds"] for e in all_eps), 3)
    challenge = {"synthetic_checks": fixture_challenges(group["raw"], rows)} if cpu else {}
    return dict(**challenge, pin=PIN, fixture=cpu, run_directory=str(folder), runtime=runtime, episodes=len(all_eps),
                controls=controls, queries=sum(e["calls"] for e in all_eps), rows=rows, timing_counts=counts,
                endpoints=endpoints, bootstrap=dict(unit="release layout, both arms travel together in pooled draw",
                    ordered_draws=256, contrast_order=["visual-minus-"+b for b in BASELINES],
                    percentile_ranks_one_based=[7,250], joint_frequency=histograms), checks_agree_with_official=True,
                cpu_affinity=sorted(os.sched_getaffinity(0)), readback_seconds=round(time.monotonic()-start, 3),
                verified_utc=datetime.now(timezone.utc).isoformat())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--job", default="job-wnafkdyqfdu5")
    parser.add_argument("--fixture", action="store_true")
    args = parser.parse_args()
    print(json.dumps(audit(args), allow_nan=False, separators=(",", ":")))
