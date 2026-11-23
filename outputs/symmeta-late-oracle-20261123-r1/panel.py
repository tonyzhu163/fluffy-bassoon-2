"""Recover and qualify the official seed-41 panel from the original Baige shards."""
from pathlib import Path
import ast
import inspect
import os

import numpy as np
from paired_inputs import archived_panel, native, ROOT, CAMPAIGN
from paired_records import SETTINGS, atomic, checked, clustered_interval, digest, encode, parse, require

HELPERS = Path(os.environ["HELPER_TREE"])
HISTORY = HELPERS / "outputs/xemb-live-wave-20261121-r1/evidence"
DATA = ROOT / "outputs" / CAMPAIGN
CHECKPOINT = DATA / "gpu03v2/stage1/checkpoints/equiv_v2_dis_dz-passive_attempt1-seed41.msgpack"
CHECKPOINT_SHA = "c12bace5cb391499d6abd7cc5767f94b1519de7d77940efb0cfcb71ca0f686a4"
PAPER = {"stream11": "ours_equiv_v2_online", "retry": "ours_equiv_v2_passive", "exact1": "privileged_oracle"}
ARMS = ("exact1", "exact11", "stream11")
IDENTITY = (0., 1., 1)


def differences(rows, left, right):
    grouped = {}
    deltas = []
    for row in rows:
        delta = int(row["outcomes"][left]) - int(row["outcomes"][right])
        grouped.setdefault(f"{row['suite']}/{row['task_id']}", []).append(delta)
        deltas.append(delta)
    require(len(grouped) == 8 and all(len(g)*8 == len(rows) for g in grouped.values()), "Unbalanced tasks")
    totals = [sum(g) for g in grouped.values()]
    return dict(left=left, right=right, n=len(rows), left_only=deltas.count(1), right_only=deltas.count(-1),
                ties=deltas.count(0), difference_pp=100*sum(deltas)/len(rows),
                task_ci95_pp=clustered_interval(totals, len(rows)),
                task_differences_pp={k: 100*sum(g)/len(g) for k, g in grouped.items()})


def noise_bank(seed, horizon):
    return np.random.default_rng(seed).standard_normal(((horizon+4)//5, 10, 32)).astype(np.float32)


def prepare(out):
    ev = native()
    tasks, scheduled = ev.build_test_plan()
    support_root, locked, refs, _, runtime = archived_panel("gpu09test", tasks)
    checked(CHECKPOINT, CHECKPOINT_SHA)
    source_shas = {}
    original = Path(os.environ["PAIRED_DEPS"]).parent / "source-s2ab-v2"
    # Compare the executable streaming function on both original rollout revisions.
    old = (original / "outputs" / CAMPAIGN / "eval_stage2a_live_ur5e.py").read_text()
    functions = {node.name: node for node in ast.parse(old).body if isinstance(node, ast.FunctionDef)}
    for name in ("run_online_streaming_episode", "_get_streaming_step_fn"):
        text = inspect.getsource(getattr(ev, name))
        require(text.rstrip() == ast.get_source_segment(old, functions[name]), f"Original implementation differs for {name}")
        source_shas[name] = digest(text.encode())
    for path in ("outputs/libero-xemb-ur5e-sysid-20261120-r1/ur5e_motor.py",
                 "outputs/libero-xemb-ur5e-sysid-20261120-r1/ur5e_sim_worker.py",
                 "src/fluffy_bassoon/xemb/equiv_v2.py", "src/fluffy_bassoon/xemb/embodiment.py"):
        raw = (original / path).read_bytes()
        require(raw == (Path(os.environ["PAIRED_DEPS"]) / path).read_bytes(), f"Historical dependency drift: {path}")
        source_shas[path] = digest(raw)
    prov = parse(checked(HISTORY/"pi05-provenance.json",
                         "c0301415a7efd149359dd2ad50bd4d0ebc049c5b3429e0f25cc380bf5a8ca389"))
    paper = [parse(line) for line in checked(HISTORY/"pi05-rows.jsonl",
                          "1863e924cebc57bbac19ed65195f85171c2d3757018b752a631491a140105185").splitlines()]
    summary = parse(checked(HELPERS/"outputs"/CAMPAIGN/"evidence/ur5e-pi05-summary.json",
                          "1608ebb1b4eac24fbed1c108302a226c6657f12569f3fea050352642c97c56ea"))
    rows, conditions, archives = [], [], []
    require(len(tasks) == len(set(tasks)) == len(refs) == 8 and len(scheduled) == len(paper) == len(locked) == 96,
            "Incomplete schedule")
    suites = sorted({s for s, t in tasks})
    for shard, ((suite, tid), ref) in enumerate(zip(tasks, refs, strict=True)):
        p = prov["shards"][str(shard)]
        hist = parse(checked(Path(p["path"]), p["sha256"]))
        launch = (DATA/p["attempt"]/"launch.txt").read_text().split()
        require(launch[:2] == [p["pin"], p["job"]+"-master-0"] and hist["shard_idx"] == shard
                and len(hist["rows"]) == 12, "Shard launch/provenance")
        archives.append(dict(**p, shard=shard))
        # Matched-flow was a separate native job. These are reset references only.
        flow_file = ROOT/"outputs/symmeta-matched-flow-20261123-r1/gpu01"/f"shard-{shard}/{12*shard:02d}.json"
        flow_reset = parse(flow_file.read_bytes())["arms"]["learned"]["reset"]
        require(flow_reset == ref, f"Independent historical reset mismatch on task {shard}")
        if shard == 3:
            log = (DATA/p["attempt"]/"logs/shard-3.log").read_text().splitlines()
            goal_reset, = [parse(line[line.index("{"):]) for line in log if line.startswith("Paired reset reference ")]
            require(goal_reset == ref, "Original Goal 6 recovery reset")
        expected_reset = {k:v for k,v in ref.items() if k != "renderer"}
        ref = {k: ref[k] for k in SETTINGS}
        query = dict(arm="UR5e", suite=suite, task_id=tid, variant="BASE", init_state_id=13,
                     seed=41800000+10000*suites.index(suite)+100*tid+13)
        bank = noise_bank(query["seed"] + ev.NOISE_SEED_OFFSET, ref["horizon"])
        require(ref["noise_seed"] == query["seed"]+817117 and ev.array_digest(bank) == ref["noise_sha256"],
                f"Historical query noise {shard}")
        conditions.append(dict(reset=ref, gpu_reset=expected_reset, noise_draws=[ev.array_digest(x) for x in bank]))
        manifest = support_root / f"shard-{shard}/support-meta.json"
        meta = parse(checked(manifest,locked[12*shard]["manifest_sha256"]))
        require(len(meta) == 12, "Incomplete original support manifest")
        for slot, record in enumerate(hist["rows"]):
            i = 12*shard + slot
            e, r = scheduled[i], locked[i]
            spec = ev.get_embodiment_spec(record["embodiment"])
            truth = [spec.yaw_deg, spec.xy_scale, spec.gripper_sign]
            require(i == record["index"] and record == paper[i] and all(record[k] == r[k] == e[k]
                    for k in ("index","suite","task_id","embodiment","regime")) and r["query"] == query
                    and e["query_init_state_id"] == 13 and r["init_state_id"] == e["support_init_state_id"] == 11
                    and [record[k] for k in ("true_yaw","true_scale","true_grip")] == truth, f"Case {i}")
            tape = Path(r["path"])
            require(str(tape) == meta[slot]["passive_npz"] and r["physics"] == meta[slot]["physics"] and
                    r["embodiment"] == meta[slot]["embodiment"] and r["seed"] == e["seed"] and
                    r["manifest"] == str(manifest), f"Retry support mapping {i}")
            checked(tape,r["sha256"])
            with np.load(tape,allow_pickle=False) as archive:
                require(np.array_equal(np.flatnonzero(archive["valid"]), [0,1,2]) and ev.CHUNK_SIZE == 5,
                        f"Original 15-control passive support {i}")
            ev.validate_arm_results(record)
            rows.append(dict(index=i, suite=suite, task_id=tid, shard=shard, query=query,
                             embodiment=spec.name, regime=spec.regime, truth=truth,
                             support_physics=r["physics"], support_sha256=r["sha256"], archived={k: record["arms"][v] for k,v in PAPER.items()}))
    scores = [dict(**r, outcomes={k: a["success"] for k,a in r["archived"].items()}) for r in rows]
    counts = {k: sum(r["outcomes"][k] for r in scores) for k in PAPER}
    require(counts == {"stream11":20,"retry":76,"exact1":82} and
            all(summary["by_regime"]["overall"]["successes"][v] == counts[k] for k,v in PAPER.items()), "Paper counts")
    fields = {k: sorted(set().union(*(r["archived"][k].keys() for r in rows))) for k in PAPER}
    provenance = dict(original_shards=archives, frozen_reader=str(CHECKPOINT), reader_sha256=CHECKPOINT_SHA,
                      support_layout=11, query_layout=13, archived_counts=counts, archived_fields=fields,
                      archived_retry_minus_stream=differences(scores,"retry","stream11"),
                      original_function_hashes=source_shas, runtime=runtime, archived_arms_reused=0)
    atomic(out/"panel.json", encode(dict(provenance=provenance, rows=rows, conditions=conditions)))
    print(f"Verified all original rows and eight noise banks, historical scores {counts}", flush=True)
