"""Yaw-only intervention on the official UR5e pi0.5 query panel."""
import argparse
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace

OUTPUTS = Path(__file__).resolve().parents[1]
DATA = Path("/mowenjie/fluffy-bassoon-2/outputs")
CAMERA = OUTPUTS / "symmeta-camera-command-20261123-r1/evidence"
PUBLISHED = OUTPUTS / "xemb-live-wave-20261121-r1/evidence/pi05-rows.jsonl"
ARCHIVE = DATA / "libero-xemb-ur5e-scale-20261120-r1/gpu09test/test_live"
WORLD = "96d9de3131348abed6afd6271dd5197c77cbb7a9159e6559fe5d3833587a3449"
ROLL = "be4190952f62bc352e92ec99e00cb407b04acb2e8eb0597964cc964f9d0b8850"
OFFICIAL = "1863e924cebc57bbac19ed65195f85171c2d3757018b752a631491a140105185"
CHECKPOINT = "c12bace5cb391499d6abd7cc5767f94b1519de7d77940efb0cfcb71ca0f686a4"
SUPPORT_SET = "8729fa68343e2a7a0cc6471ab25e7c707555a866e682a3dc999079a7b9dc3554"
READER = "b740827bcb8bf8d1f90cc86648772e8a6b9dbce5dd6e7fffeaa88a75a7deb654"
KEYS = ("index", "suite", "task_id", "embodiment")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def checked(path, expected):
    raw = path.read_bytes()
    require(digest(raw) == expected, f"SHA-256 mismatch: {path}")
    return raw


def official_rows():
    return [json.loads(line) for line in checked(PUBLISHED, OFFICIAL).splitlines()]


def identity(row):
    return tuple(row[k] for k in KEYS)


def panel():
    world = json.loads(gzip.decompress(checked(CAMERA / "world.json.gz", WORLD)))
    roll = json.loads(gzip.decompress(checked(CAMERA / "roll.json.gz", ROLL)))
    upright = [r for r in roll["rows"] if r["roll_deg"] == 0]
    reader_path = Path(__file__).parent / "evidence/reader.json"
    reader = json.loads(checked(reader_path, READER))
    require(reader["checkpoint_sha256"] == CHECKPOINT and reader["backend"] == "cpu", "Wrong reader")
    official = official_rows()
    require(roll["support_set_sha256"] == SUPPORT_SET and
            roll["sources"]["checkpoint"]["sha256"] == CHECKPOINT, "Wrong frozen inputs")
    panel_rows = []
    joined = zip(world["rows"], upright, roll["supports"], official, reader["rows"], strict=True)
    for i, (w, v, s, o, learned) in enumerate(joined):
        c = s["case"]
        require(i == c["index"] and identity(c) == identity(w) == identity(v) == identity(o),
                f"Identity mismatch at {i}")
        sha = s["npz"]["sha256"]
        shard, slot = divmod(i, 12)
        path = ARCHIVE / f"shard-{shard}/support/{slot:02d}-{c['embodiment']}-passive.npz"
        require(sha == w["archive_sha256"] == v["support_sha256"] and
                v["checkpoint_sha256"] == CHECKPOINT and
                str(path) == w["archive_npz"] == c["passive_npz"] == s["npz"]["path"] and
                shard == w["shard"] == c["shard"] and
                c["support_init_state_id"] == w["init_state_id"] == 11 and
                c["query_init_state_id"] == 13 and c["support_seed"] == w["seed"] and
                s["selected_chunk_indices"] == [0, 1, 2] and s["issued_mean"] == w["issued_mean"],
                f"Support provenance mismatch at {i}")
        # Reconstruct the planar fit from commands and measured world deltas alone.
        xy = list(zip(w["issued_mean"], w["delta_world_m"], strict=True))
        dot = sum(a[0]*b[0] + a[1]*b[1] for a, b in xy)
        cross = sum(a[0]*b[1] - a[1]*b[0] for a, b in xy)
        fk = w["yaw_hat_deg"]
        require(len(xy) == 3 and math.hypot(dot, cross) > 0 and
                math.isclose(math.degrees(math.atan2(cross, dot)), fk, abs_tol=1e-10, rel_tol=0),
                f"FK alignment mismatch at {i}")
        require(learned["index"] == i and learned["id"] == c["id"] == v["id"] and
                learned["support_sha256"] == sha and learned["yaw_hat"] == v["yaw_deg"]["model_native"],
                f"Frozen reader mismatch at {i}")
        yaw = v["yaw_deg"]["model_native"]
        scale, grip = learned["scale_hat"], learned["grip_hat"]
        require(math.isfinite(yaw) and math.isfinite(scale) and scale > 0 and grip in (-1, 1),
                f"Invalid prediction at {i}")
        panel_rows.append(dict(**{k: c[k] for k in KEYS}, shard=shard,
                               support=dict(path=str(path), sha256=sha, physics=c["physics"],
                                            seed=w["seed"], init_state_id=11),
                               symmeta_yaw=yaw, fk_yaw=fk, scale=scale, grip=grip))
    ordered = "".join(r["support"]["sha256"] + "\n" for r in panel_rows)
    require(len(panel_rows) == 96 and digest(ordered.encode()) == SUPPORT_SET, "Incomplete panel")
    return panel_rows


def prepare(out, runner):
    tasks, entries = runner.build_test_plan()
    require(len(tasks) == 8 and len(entries) == 96, "Unexpected official schedule")
    suites = sorted({suite for suite, _ in tasks})
    rows = panel()
    for shard, task in enumerate(tasks):
        meta_path = ARCHIVE / f"shard-{shard}/support-meta.json"
        meta_bytes = meta_path.read_bytes()
        group = rows[12*shard:12*(shard+1)]
        for r, m in zip(group, json.loads(meta_bytes), strict=True):
            e = entries[r["index"]]
            support = r["support"]
            require(identity(e) == identity(r) and task == (r["suite"], r["task_id"]) and
                    e["arm"] == "UR5e" and e["seed"] == support["seed"] and
                    e["support_init_state_id"] == 11 and e["query_init_state_id"] == 13 and
                    (m["embodiment"], m["physics"], m["passive_npz"]) ==
                    (r["embodiment"], support["physics"], support["path"]), f"Wrong source for {r['index']}")
            checked(Path(support["path"]), support["sha256"])
            support.update(manifest=str(meta_path), manifest_sha256=digest(meta_bytes))
            seed = 41800000 + 10000 * suites.index(e["suite"]) + 100 * e["task_id"] + 13
            r.update(regime=e["regime"], query=dict(arm="UR5e", suite=e["suite"], task_id=e["task_id"],
                                                   variant="BASE", init_state_id=13, seed=seed))
    runner.write_json(out / "panel.json", dict(rows=rows, world_sha256=WORLD, roll_sha256=ROLL,
                                              official_sha256=OFFICIAL, reader_sha256=READER, checkpoint_sha256=CHECKPOINT,
                                              support_set_sha256=SUPPORT_SET))


def validate_pair(r, arms, runner):
    require(set(arms) == {"symmeta", "fk"}, "Incomplete arm set")
    a, b = arms["symmeta"], arms["fk"]
    reset = a["reset"]
    require(reset == b["reset"] and reset["physics"] != r["support"]["physics"] and
            reset["arm"] == "UR5e" and reset["joint_count"] == 6 and reset["worker_step_index"] == 0 and
            reset["noise_seed"] == r["query"]["seed"] + runner.NOISE_SEED_OFFSET and
            len(reset["noise_sha256"]) == 64 and a["opening_proposal_sha256"] == b["opening_proposal_sha256"]
            and len(a["opening_proposal_sha256"]) == 64, f"Pairing failed at {r['index']}")
    for name, rec in arms.items():
        require(rec["inferred"] == dict(yaw_deg=r[name + "_yaw"], xy_scale=r["scale"], gripper_sign=r["grip"])
                and type(rec["success"]) is bool and 0 < rec["controls"] <= reset["horizon"]
                and len(rec["native_rewards"]) == rec["controls"]
                and sum(rec["native_rewards"]) == rec["native_return"], f"Invalid native record at {r['index']}")


def run_shard(out, shard, runner):
    rows = json.loads((out / "panel.json").read_text())["rows"][12*shard:12*(shard+1)]
    directory = out / f"shard-{shard}"
    directory.mkdir()
    policy = runner.load_frozen_policy("pi05")
    resets = {}
    with runner.UR5eSimulator() as sim:
        for slot, r in enumerate(rows):
            case = r["query"]
            # The simulator receives the interface identity. The motor only needs its name.
            spec = SimpleNamespace(name=r["embodiment"])
            if slot == 0:
                sim.call("create", dict(case, embodiment=spec.name))
            arms = {}
            for name in ("symmeta", "fk"):
                key = (case["suite"], case["task_id"], case["init_state_id"])
                correction = dict(yaw_hat=r[name + "_yaw"], scale_hat=r["scale"],
                                  grip_hat=r["grip"])
                arms[name] = runner.run_paired_query(
                    runner.run_episode, sim, policy, case, spec, arm_name=name, resets=resets,
                    key=key, support_physics=r["support"]["physics"], expected_prompt=None,
                    create_attempts=3, **correction)
            validate_pair(r, arms, runner)
            runner.write_json(directory / f"{r['index']:02d}.json", dict(**r, arms=arms))
            print(f"Paired {r['index']}: SymMeta={arms['symmeta']['success']} FK={arms['fk']['success']}", flush=True)


def merge(out, runner):
    rows = []
    historical = official_rows()
    for r in json.loads((out / "panel.json").read_text())["rows"]:
        row = json.loads((out / f"shard-{r['shard']}/{r['index']:02d}.json").read_text())
        require({k: v for k, v in row.items() if k != "arms"} == r, "Unexpected output identity")
        validate_pair(r, row["arms"], runner)
        if rows and r["shard"] == rows[-1]["shard"]:
            require(row["arms"]["fk"]["reset"] == rows[-1]["arms"]["fk"]["reset"], "Task reset changed")
        rows.append(row)
    require([r["index"] for r in rows] == list(range(96)), "Incomplete paired panel")

    def stats(group):
        return dict(n=len(group), successes={a: sum(r["arms"][a]["success"] for r in group)
                                             for a in ("symmeta", "fk")},
                    fk_minus_symmeta=runner.task_clustered_bootstrap(group, "fk", "symmeta", seed=20261121))

    reference = [r["arms"]["ours_equiv_v2_passive"]["success"] for r in historical]
    changed = [r["index"] for r in rows if r["arms"]["symmeta"]["success"] != reference[r["index"]]]
    ood = [r for r in rows if r["regime"] in ("compositional_ood", "continuous_angle_ood")]
    require(len(ood) == 64, "Incomplete OOD panel")
    runner.write_json(out / "summary.json", dict(
        overall=stats(rows), ood=stats(ood),
        tasks={f"{s}/{t}": stats([r for r in rows if (r["suite"], r["task_id"]) == (s, t)])
               for s, t in sorted({(r["suite"], r["task_id"]) for r in rows})},
        historical_symmeta=dict(successes=sum(reference), changed_indices=changed)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "shard", "merge"))
    parser.add_argument("output", type=Path)
    parser.add_argument("--shard", type=int, choices=range(8))
    args = parser.parse_args()
    import eval_stage2a_live_ur5e as official

    import ur5e_motor
    import ur5e_sim_worker

    dependencies = Path(os.environ["PAIRED_DEPS"]) / "outputs"
    for module, directory in ((official, "xemb-icml-campaign-20261121-r1"),
                              (ur5e_motor, "libero-xemb-ur5e-sysid-20261120-r1"),
                              (ur5e_sim_worker, "libero-xemb-ur5e-sysid-20261120-r1")):
        expected = dependencies / directory / (module.__name__ + ".py")
        require(Path(module.__file__).resolve() == expected.resolve(), f"Wrong module origin: {module}")

    if args.mode == "prepare":
        prepare(args.output, official)
    elif args.mode == "merge":
        merge(args.output, official)
    elif args.shard is None:
        parser.error("shard mode requires --shard")
    else:
        run_shard(args.output, args.shard, official)
