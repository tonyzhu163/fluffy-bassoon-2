"""Read each panel from its archived passive tapes and pinned native evaluator."""
import gzip
import importlib.util
import os
from pathlib import Path
import subprocess
import sys

from paired_records import HERE, HASH, OOD, atomic, checked, digest, encode, join, load_predictions, parse, require

OUTPUTS = HERE.parent
ROOT = Path("/mowenjie/fluffy-bassoon-2")
CAMPAIGN = "xemb-icml-campaign-20261121-r1"
SYSID = "libero-xemb-ur5e-sysid-20261120-r1"
SCALE = "libero-xemb-ur5e-scale-20261120-r1"
CROSS = OUTPUTS / "symmeta-fk-crossarm-20261123-r1"
FLOW = OUTPUTS / "symmeta-matched-flow-20261123-r1"
GOAL_PIN = "ff2ddf0ab03e6fbe243979556d8d8f2ebb9bc29c"
KEYS = ("index", "suite", "task_id", "embodiment")
PANEL_UR = "gpu09test"


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def check_tree(path, pin):
    require(subprocess.check_output(["git", "-C", path, "rev-parse", "HEAD"], text=True).strip() == pin
            and not subprocess.check_output(["git", "--no-optional-locks", "-C", path, "status", "--porcelain"]),
            f"Wrong or dirty source tree {path}")


def native():
    import eval_stage2a_live_ur5e as ev
    import ur5e_motor
    import ur5e_sim_worker

    src = Path(os.environ["PAIRED_DEPS"])
    for m, directory in ((ev, CAMPAIGN), (ur5e_motor, SYSID), (ur5e_sim_worker, SYSID)):
        require(Path(m.__file__).resolve() == src/"outputs"/directory/(m.__name__+".py"), f"Wrong source {m}")
    require(ev.run_episode is ur5e_motor.run_episode and ev.UR5eSimulator is ur5e_sim_worker.UR5eSimulator,
            "Historical motor/simulator identity")
    return ev


def archived_panel(panel, tasks):
    src = Path(os.environ["PAIRED_DEPS"])
    if panel == PANEL_UR:
        flow = module("matched_flow", FLOW/"live.py")
        fk = flow.load_fk()
        check_tree(src, GOAL_PIN)
        history = fk.DATA/flow.FK_RUN/"gpu01"
        locked = parse(checked(history/"panel.json", flow.FK_PANEL))
        stable = flow.task_conditions(fk)
        refs = []
        for shard in range(8):
            rec = parse((history/f"shard-{shard}/{12*shard:02d}.json").read_bytes())
            ref = rec["arms"]["symmeta"]["reset"]
            require({k: ref[k] for k in flow.RESET_FIELDS} == stable[shard], "Goal6 task conditions")
            refs.append(ref)
        rows = [dict(**{k: r[k] for k in (*KEYS, "regime", "query")}, **r["support"]) for r in locked["rows"]]
        return (fk.ARCHIVE, rows, refs, [None]*8,
                dict(arm="UR5e", policy="pi05", source=str(src), pin=GOAL_PIN, weights=flow.qualify_runtime(fk),
                     reference_panel_sha256=flow.FK_PANEL, support_set_sha256=fk.SUPPORT_SET))

    sys.path.insert(0, str(CROSS))
    cross = module("fk_crossarm", CROSS/"live.py")
    archive, historical_src, info = cross.sources(panel)
    require(historical_src == src, f"Wrong runtime for {panel}")
    q = parse((CROSS/"evidence/summary.json").read_bytes())[panel]
    cases = parse(gzip.decompress(checked(CROSS/"evidence"/q["artifact"], q["sha256"])))
    require(all(cases[k] == v for k, v in info.items()) and cases["native_reader_qualified"] is True,
            f"Wrong archived cases {panel}")
    hist, meta, refs = cross.historical(archive, panel, tasks)
    rows = []
    for i, r in enumerate(cases["rows"]):
        shard, slot = divmod(i, 12)
        h = hist[shard]["rows"][slot]
        m = meta[shard][slot]
        reset = cases["resets"][shard]
        require(all(r[k] == h[k] for k in (*KEYS, "regime")) and r["index"] == i and
                r["physics"] == m["physics"] == reset["archived_physics"] and
                r["embodiment"] == m["embodiment"] and r["reader_available"] and r["native_cpu_identity"],
                f"Historical cross-arm case {panel}/{i}")
        rows.append(dict(**{k: r[k] for k in (*KEYS, "regime")}, path=m["passive_npz"], physics=r["physics"],
                         sha256=r["support_sha256"], seed=r["support_seed"], init_state_id=r["support_layout"],
                         query=dict(arm=info["arm"], suite=r["suite"], task_id=r["task_id"], variant="BASE",
                                    init_state_id=r["query_layout"], seed=r["query_seed"])))
    return (archive/"live", rows, refs, [r["identity"] for r in cases["resets"]],
            dict(**info, weights=cross.verify_policy(panel, info), archive_sha256=cross.ARCHIVE_SHA[panel],
                 reference_cases_sha256=q["sha256"], support_set_sha256=q["support_set_sha256"]))


def prepare(panel, out, predictions_sha, *, allow_fixture=False):
    import numpy as np

    ev = native()
    tasks, entries = ev.build_test_plan()
    base, rows, refs, identities, runtime = archived_panel(panel, tasks)
    suites = sorted({s for s, _ in tasks})
    require(len(tasks) == len(refs) == 8 and len(rows) == len(entries) == 96, "Historical schedule")
    arm, policy = runtime["arm"], runtime["policy"]
    cases = []
    for shard, (suite, tid) in enumerate(tasks):
        ref = refs[shard]
        sup_seed = 31800000 + 10000*suites.index(suite) + 100*tid + 11
        query = dict(arm=arm, suite=suite, task_id=tid, variant="BASE", init_state_id=13,
                     seed=41800000 + 10000*suites.index(suite) + 100*tid + 13)
        noise_seed = query["seed"] + ev.NOISE_SEED_OFFSET
        shape = ((ref["horizon"]+4)//5, 50 if policy == "pi0" else 10, 32)
        noise = np.random.default_rng(noise_seed).standard_normal(shape).astype(np.float32)
        require(ref["arm"] == arm and (identities[shard] is None or
                ref["model_class"] == identities[shard]["model_class"]) and ref["worker_step_index"] == 0
                and ref["joint_count"] == (6 if arm == "UR5e" else 7) and ref["noise_seed"] == noise_seed
                and ev.array_digest(noise) == ref["noise_sha256"] and ev.CHUNK_SIZE == 5, f"Query schedule {panel}/{shard}")
        manifest = base/f"shard-{shard}/support-meta.json"
        raw = manifest.read_bytes()
        meta = parse(raw)
        require(len(meta) == 12 and len({m["physics"] for m in meta}) == 1, manifest)
        for slot, m in enumerate(meta):
            i = shard*12 + slot
            e, r = entries[i], rows[i]
            tape = base/f"shard-{shard}/support/{slot:02d}-{e['embodiment']}-passive.npz"
            require(all(r[k] == e[k] for k in (*KEYS, "regime")) and e["index"] == i
                    and (e["suite"], e["task_id"]) == (suite, tid) and e["embodiment"] == m["embodiment"]
                    and e["support_init_state_id"] == r["init_state_id"] == 11 and e["query_init_state_id"] == 13
                    and r["seed"] == e["seed"] == sup_seed and r["query"] == query
                    and str(tape) == r["path"] == m["passive_npz"] and HASH.fullmatch(r["physics"])
                    and r["physics"] == m["physics"] != ref["physics"], f"Tape/query identity {panel}/{i}")
            if panel == PANEL_UR:
                require(str(manifest) == r["manifest"] and digest(raw) == r["manifest_sha256"], manifest)
            checked(tape, r["sha256"])
            cases.append(dict(panel=panel, **{k: e[k] for k in (*KEYS, "regime")}, shard=shard,
                              tape_sha256=r["sha256"], tape=str(tape), support_physics=r["physics"],
                              support_seed=sup_seed, support_layout=11, manifest_sha256=digest(raw), query=query))
    ordered = "".join(c["tape_sha256"]+"\n" for c in cases).encode()
    require(digest(ordered) == runtime["support_set_sha256"] and sum(c["regime"] in OOD for c in cases) == 64,
            f"Wrong tape set or held-out split {panel}")
    pred = load_predictions(out/"predictions.json", predictions_sha, allow_fixture=allow_fixture)
    result = dict(panel=panel, runtime=runtime, refs=refs, model_identities=identities,
                  predictions_sha256=predictions_sha, rows=join(panel, cases, pred))
    atomic(out/panel/"panel.json", encode(result))
    print(f"{panel}: 96 archive cases and 288 prediction keys verified", flush=True)
