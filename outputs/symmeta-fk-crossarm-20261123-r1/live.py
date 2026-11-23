"""Run one historical arm/policy per process, with a qualified yaw-only intervention."""
import argparse
import hashlib
import json
import math
import re
import sys
from pathlib import Path
from types import SimpleNamespace

from qualify import CAMPAIGN, CHECKPOINT, HERE, PANELS, ROOT, require, sources

sys.path.insert(0, str(HERE.parent / "symmeta-fk-ur5e-20261123-r1"))
from paired import checked, identity

ARCHIVE_SHA = {
    "gpu09iiwa": "57d1117436b678cf63965a95fb01f218cd0d6a36ad042afbf12ad960d59f4d5d",
    "gpu06kinova3": "e408b758bff9eef476483dfae6d6de15cbc7e2aff4c7450d96bc2cf4b978e4f5",
    "gpu08pi0fix": "a75b8fc227f19def06cf8b5ebfbd8082783a9a65264261ead91a85512898c3ca",
}
# Stable native settings. Cross-host physics and image fingerprints are recorded
# independently, then compared in full within the fresh paired experiment.
SETTINGS = ("arm", "model_class", "joint_count", "action_schema", "worker_step_index", "horizon",
            "noise_seed", "noise_sha256", "prompt", "renderer", "renderer_vendor")


def historical(archive, panel, tasks):
    h = hashlib.sha256()

    def read(path):
        raw = path.read_bytes()
        h.update(raw)
        return raw

    read(archive / "launch.txt")
    read(archive / "config.txt")
    shards = [archive / "live" / f"shard-{i}" for i in range(8)]
    native = [json.loads(read(s / "live-shard.json")) for s in shards]
    support = [json.loads(read(s / "support-meta.json")) for s in shards]
    resets = []
    for i, (suite, tid) in enumerate(tasks):
        prefix = f"Paired reset reference {(suite, tid, 13)}: ".encode()
        with (archive / "logs" / f"shard-{i}.log").open("rb") as f:
            line, = [line for line in f if line.startswith(prefix)]
        h.update(line)
        resets.append(json.loads(line[len(prefix):]))
    require(h.hexdigest() == ARCHIVE_SHA[panel], f"Changed historical sources for {panel}")
    return native, support, resets


def verify_policy(panel, info):
    import ur5e_motor
    import eval_stage2a_live_ur5e as ev

    cfg = json.loads((HERE / "QUERY_CONFIG.json").read_text())["panels"][panel]
    for k in ("launch", "config", "source"):
        require(cfg[k] == info[k], k)
    require(cfg["source_pin"] == info["pin"]
            and cfg["query_canonicalize_quat"] is (info["policy"] == "pi0"), panel)
    for file, sha in cfg["sources"].items():
        checked(Path(info["source"]) / file, sha)
    path = ev.PI0_CKPT if info["policy"] == "pi0" else ur5e_motor.OPENPI_CKPT
    require(path == Path(cfg["checkpoint"]), path)
    checked(path / "transport-manifest.json", cfg["weight_transport_manifest_sha256"])
    ready_sha = {"pi0": "220a78caa83dd4d3ed762d14f63d981a9162a348edb777a109b8940f4ecfdbc3",
                 "pi05": "ec064cce32f5003e2e3979c5712d7d6385018f6bfcc59cb2161105881e70d170"}
    ready = json.loads(checked(path / "READY.json", ready_sha[info["policy"]]))
    require(ready["status"] == "verified" and ready["checkpoint_path"] == str(path), path)
    for file in ready["files"]:
        h = hashlib.sha256()
        with (path / file["name"]).open("rb") as stream:
            for chunk in iter(lambda: stream.read(8 << 20), b""):
                h.update(chunk)
        require(h.hexdigest() == file["sha256"], path / file["name"])
    return dict(path=str(path), ready_sha256=ready_sha[info["policy"]],
                transport_sha256=cfg["weight_transport_manifest_sha256"])


def prepare(out, manifest, manifest_sha, panel, info, archive, ev):
    pins = json.loads(checked(manifest, manifest_sha), object_pairs_hook=ev.unique_json_keys)
    require(pins.keys() == PANELS.keys(), "Manifest must pin the three panels")
    entry = pins[panel]
    # Accept the qualifier's own evidence/summary.json directly.
    if "json_sha256" in entry:
        run = ROOT / "outputs" / HERE.name / entry["run_attempt"] / panel
        require((run / "launch.txt").read_text().split()[0] == entry["run_pin"], run)
        entry = dict(path=str(run / "cases.json"), sha256=entry["json_sha256"],
                     qualification=entry)
    path = Path(entry["path"])
    require(path.is_absolute(), path)
    c = json.loads(checked(path, entry["sha256"]), object_pairs_hook=ev.unique_json_keys)
    require(all(c[k] == v for k, v in info.items()) and c["checkpoint_sha256"] == CHECKPOINT, path)
    withheld = [r["index"] for r in c["rows"] if r["fk_available"] is False]
    if "qualification" in entry:
        q = entry["qualification"]
        require(q["cases"] == q["native_reader_identity"] == 96 and q["fk_available"] == 96 - len(withheld), path)
        for key, sha_key in (("active_sha256", "active_set_sha256"), ("support_sha256", "support_set_sha256")):
            raw = "".join(r[key] + "\n" for r in c["rows"])
            require(hashlib.sha256(raw.encode()).hexdigest() == q[sha_key], sha_key)
    else:
        require(withheld == entry.get("withheld_indices", []), f"Unexpected unavailable indices: {withheld}")
    require(c["qualified"] is (not withheld)
            and c["fk_qualified"] is (not withheld) and c["native_reader_qualified"] is True,
            f"Offline availability differs from manifest for {panel}: {withheld}")
    policy_weights = verify_policy(panel, info)
    ckpt = archive.parent / "gpu03v2/stage1/checkpoints/equiv_v2_dis_dz-passive_attempt1-seed41.msgpack"
    checked(ckpt, CHECKPOINT)
    tasks, plan = ev.build_test_plan()
    suites = sorted({s for s, _ in tasks})
    require(len(plan) == len(c["rows"]) == 96 and len(tasks) == len(c["resets"]) == 8, path)
    native, support, refs = historical(archive, panel, tasks)
    cases = []
    for shard, (suite, tid) in enumerate(tasks):
        reset, ref = c["resets"][shard], refs[shard]
        require((reset["shard"], reset["suite"], reset["task_id"]) == (shard, suite, tid)
                and native[shard]["shard_idx"] == shard and len(native[shard]["rows"]) == len(support[shard]) == 12
                and reset["native_horizon"] == ref["horizon"]
                and reset["identity"]["arm"] == info["arm"] == ref["arm"]
                and all(reset["identity"][k] == ref[k] for k in ("model_class", "joint_count")), reset)
        for slot, (m, hist) in enumerate(zip(support[shard], native[shard]["rows"], strict=True)):
            i = 12 * shard + slot
            r, e = c["rows"][i], plan[i]
            qseed = 41800000 + 10000 * suites.index(suite) + 100 * tid + 13
            require(identity(e) == identity(r) == identity(hist) and r["index"] == i
                    and (r["suite"], r["task_id"], r["shard"]) == (suite, tid, shard)
                    and r["regime"] == hist["regime"] == e["regime"]
                    and r["embodiment"] == m["embodiment"]
                    and r["physics"] == m["physics"] == reset["archived_physics"]
                    and r["support_seed"] == e["seed"] == reset["support_seed"]
                    and r["support_layout"] == e["support_init_state_id"] == reset["support_layout"] == 11
                    and r["query_layout"] == e["query_init_state_id"] == 13 and r["query_seed"] == qseed
                    and r["noise_seed"] == qseed + ev.NOISE_SEED_OFFSET == ref["noise_seed"], (panel, i))
            for mode, key in (("active", "active_sha256"), ("passive", "support_sha256")):
                tape = archive / "live" / f"shard-{shard}/support/{slot:02d}-{r['embodiment']}-{mode}.npz"
                require(str(tape) == m[mode + "_npz"], tape)
                checked(tape, r[key])
            baseline = hist["arms"]["ours_equiv_v2_passive"]
            require(r["historical"] == baseline and all(r[k] is True for k in
                    ("reader_available", "native_cpu_identity", "native_reader_qualified")), (panel, i))
            y, g, sign = r["yaw_hat"], r["scale_hat"], r["grip_hat"]
            require(math.isfinite(y) and math.isfinite(g) and g > 0 and type(sign) is int
                    and sign in (-1, 1) and r["yaw_residual_deg"] == (y - baseline["yaw_hat"] + 180) % 360 - 180
                    and r["scale_residual"] == g - baseline["scale_hat"], f"Reader identity at {panel}/{i}")
            if i in withheld:
                require(r["fk_yaw_hat"] is None and r["reason"], f"Malformed withheld FK at {panel}/{i}")
            else:
                require(r["fk_available"] is True and reset["active_replay"].get("matched", True) is True,
                        f"Unqualified FK at {panel}/{i}")
                xy = list(zip(r["issued_xy"], r["world_delta_xy_m"], strict=True))
                dot = sum(a*x+b*z for (a, b), (x, z) in xy)
                cross = sum(a*z-b*x for (a, b), (x, z) in xy)
                require(len(xy) == 3 and math.hypot(dot, cross) > 0 and
                        math.isclose(math.degrees(math.atan2(cross, dot)), r["fk_yaw_hat"], abs_tol=1e-9, rel_tol=0),
                        f"FK identity at {panel}/{i}")
            cases.append(dict(panel=panel, index=i, shard=shard, suite=suite, task_id=tid, regime=r["regime"],
                              embodiment=r["embodiment"], support=m, passive_sha256=r["support_sha256"],
                              active_sha256=r["active_sha256"], support_seed=r["support_seed"], support_layout=11,
                              query=dict(arm=info["arm"], suite=suite, task_id=tid, variant="BASE",
                                         init_state_id=13, seed=qseed),
                              withheld_reason=r.get("reason") if i in withheld else None,
                              symmeta_yaw=y, fk_yaw=r["fk_yaw_hat"], scale=g, grip=sign,
                              historical_success=baseline["success"]))
    require(all(len({r[k] for r in cases}) == 96 for k in ("active_sha256", "passive_sha256")),
            "Duplicated support tape in panel")
    ev.write_json(out / "panel.json", dict(**info, policy_weights=policy_weights, cases_file=entry, manifest_sha256=manifest_sha,
                  archive_sha256=ARCHIVE_SHA[panel], checkpoint_sha256=CHECKPOINT, rows=cases,
                  resets=refs, model_identities=[r["identity"] for r in c["resets"]]))


def validate(r, arms, ref):
    require(arms.keys() == {"symmeta", "fk"}, r["index"])
    a, b = arms["symmeta"], arms["fk"]
    reset = a["reset"]
    require(reset == b["reset"] and all(reset[k] == ref[k] for k in SETTINGS)
            and reset["physics"] != r["support"]["physics"]
            and re.fullmatch("[0-9a-f]{64}", a["opening_proposal_sha256"])
            and a["opening_proposal_sha256"] == b["opening_proposal_sha256"], f"Pair mismatch at {r['index']}")
    for name, rec in arms.items():
        require(rec["inferred"] == dict(yaw_deg=r[name + "_yaw"], xy_scale=r["scale"], gripper_sign=r["grip"])
                and rec["released_prompt"] == ref["prompt"] and type(rec["success"]) is bool
                and 0 < rec["controls"] <= reset["horizon"] and len(rec["native_rewards"]) == rec["controls"]
                and math.isfinite(rec["native_return"]) and sum(rec["native_rewards"]) == rec["native_return"],
                f"Invalid native outcome at {r['index']}")


def run_shard(out, shard, data, ev):
    dest = out / f"shard-{shard}"
    dest.mkdir()
    ref = data["resets"][shard]
    group = [r for r in data["rows"][12*shard:12*(shard+1)] if r["withheld_reason"] is None]
    if not group:
        print("All FK estimates withheld by qualification", flush=True)
        return
    policy = ev.load_frozen_policy(data["policy"])
    resets = {}
    with ev.UR5eSimulator() as sim:
        for slot, r in enumerate(group):
            case = r["query"]
            spec = SimpleNamespace(name=r["embodiment"])
            if slot == 0:
                warm = dict(case, embodiment=spec.name)
                created, initial = sim.call("create", warm)
                require(created == dict(warm, max_steps=ref["horizon"])
                        and initial["model_identity"] == data["model_identities"][shard]
                        and initial["language_instruction"] == ref["prompt"]
                        and all(initial[k] == ref[k] for k in ("renderer", "renderer_vendor", "worker_step_index")),
                        f"Model, renderer or reset mismatch for {warm}")
            arms = {}
            for name in ("symmeta", "fk"):
                arms[name] = ev.run_paired_query(ev.run_episode, sim, policy, case, spec, arm_name=name,
                    resets=resets, key=(r["suite"], r["task_id"], 13), support_physics=r["support"]["physics"],
                    expected_prompt=ref["prompt"], create_attempts=3, canonicalize_quat=data["policy"] == "pi0",
                    yaw_hat=r[name + "_yaw"], scale_hat=r["scale"], grip_hat=r["grip"])
            validate(r, arms, ref)
            ev.write_json(dest / f"{r['index']:02d}.json", dict(**r, arms=arms))
            print(r["panel"], r["index"], {k: v["success"] for k, v in arms.items()}, flush=True)


def merge(out, data, ev):
    rows = []
    for r in data["rows"]:
        case_path = out / f"shard-{r['shard']}/{r['index']:02d}.json"
        if r["withheld_reason"] is not None:
            require(not case_path.exists(), f"Unexpected rollout for withheld case {case_path}")
            continue
        row = json.loads(case_path.read_text())
        require({k: v for k, v in row.items() if k != "arms"} == r, r["index"])
        validate(r, row["arms"], data["resets"][r["shard"]])
        if rows and rows[-1]["shard"] == r["shard"]:
            require(row["arms"]["fk"]["reset"] == rows[-1]["arms"]["fk"]["reset"], r["index"])
        rows.append(row)

    def stats(group):
        return dict(n=len(group), successes={a: sum(r["arms"][a]["success"] for r in group)
                                            for a in ("symmeta", "fk")},
                    both_failed=[r["index"] for r in group if not any(a["success"] for a in r["arms"].values())],
                    fk_minus_symmeta=ev.task_clustered_bootstrap(group, "fk", "symmeta", seed=20261121))

    ood = [r for r in rows if r["regime"] in ("compositional_ood", "continuous_angle_ood")]
    withheld = {r["index"]: r["withheld_reason"] for r in data["rows"] if r["withheld_reason"] is not None}
    require(len(rows) + len(withheld) == 96, "Incomplete panel")
    ev.write_json(out / "summary.json", dict(panel=data["panel"], withheld=withheld, scheduled=96,
        overall=stats(rows) if rows else None, ood=stats(ood) if ood else None,
        tasks={f"{s}/{t}": stats([r for r in rows if (r["suite"], r["task_id"]) == (s, t)])
               for s, t in sorted({(r["suite"], r["task_id"]) for r in rows})},
        historical_full_successes=sum(r["historical_success"] for r in data["rows"]),
        historical_paired_successes=sum(r["historical_success"] for r in rows),
        changed_indices=[r["index"] for r in rows if r["arms"]["symmeta"]["success"] != r["historical_success"]]))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("prepare", "shard", "merge"))
    p.add_argument("panel", choices=PANELS)
    p.add_argument("out", type=Path)
    p.add_argument("--manifest", type=Path)
    p.add_argument("--sha256")
    p.add_argument("--shard", type=int, choices=range(8))
    args = p.parse_args()
    archive, src, info = sources(args.panel)
    import eval_stage2a_live_ur5e as ev
    import ur5e_motor
    import ur5e_sim_worker

    for module, folder in ((ev, CAMPAIGN), (ur5e_motor, "libero-xemb-ur5e-sysid-20261120-r1"),
                           (ur5e_sim_worker, "libero-xemb-ur5e-sysid-20261120-r1")):
        require(Path(module.__file__).resolve() == src / "outputs" / folder / (module.__name__ + ".py"), module)
    if args.mode == "prepare":
        prepare(args.out, args.manifest, args.sha256, args.panel, info, archive, ev)
    else:
        data = json.loads((args.out / "panel.json").read_text())
        require(all(data[k] == v for k, v in info.items()), "Prepared panel identity changed")
        if args.mode == "merge":
            merge(args.out, data, ev)
        else:
            require(args.shard is not None, "Missing --shard")
            run_shard(args.out, args.shard, data, ev)
