"""Fresh learned versus linear yaw on the official UR5e pi0.5 second-attempt panel."""
import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import eval_stage2a_live_ur5e as native
import ur5e_motor as motor
import ur5e_sim_worker as worker

HERE = Path(__file__).resolve().parent
FK_RUN = "symmeta-fk-ur5e-20261123-r1"
CAMPAIGN = "xemb-icml-campaign-20261121-r1"
SYSID = "libero-xemb-ur5e-sysid-20261120-r1"
FK_EVALUATOR = "1221f4f9e571fa9a8fd8a8d3de9d2489a41bfeff114ebbb01c7411e0bbe1f7d6"
FK_PANEL = "51a4575ff3105b0e7005befb598f94d8bf69ed0c068324c8833a92b11cfd05d0"
RESET_FIELDS = ("noise_sha256", "noise_seed", "horizon", "prompt", "action_schema",
                "perturbation_plan", "model_class")
FK_TASK_CONDITIONS = "49eec3d35cdc3bbf74e7e14d5fe459141dd8ee311f56634be7a234e706004f30"
PREDICTION_RECEIPT = "699aec14de6470e0dbba3ab71ab7cc70a8ee4cf2d2df3312116f94520862db1e"
POLICY_MANIFEST = "ec064cce32f5003e2e3979c5712d7d6385018f6bfcc59cb2161105881e70d170"
SIM_MANIFEST = "d8a8f31f5da86402927a469b9b4742120c75a54030bcc9b5fba8f5a157fbd915"
ARMS = ("learned", "linear")
HELD_OUT = {"compositional_ood", "continuous_angle_ood"}
BOOTSTRAP_SEED = 20261121
BOOTSTRAP_DRAWS = 2000


def require(ok, message):
    if not ok:
        raise ValueError(message)


def parse(raw):
    return json.loads(raw, object_pairs_hook=native.unique_json_keys)


def checked_files(base, files, key):
    for entry in files:
        path = base / entry[key]
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        require(digest == entry["sha256"], f"File digest mismatch: {path}")


def load_fk():
    deps = Path(os.environ["PAIRED_DEPS"]) / "outputs"
    for module, directory in ((native, CAMPAIGN), (motor, SYSID), (worker, SYSID)):
        require(Path(module.__file__).resolve() == deps / directory / (module.__name__ + ".py"),
                f"Unexpected module origin: {module.__file__}")
    path = HERE.parent / FK_RUN / "paired.py"
    require(hashlib.sha256(path.read_bytes()).hexdigest() == FK_EVALUATOR, "FK evaluator changed")
    spec = importlib.util.spec_from_file_location("verified_fk", path)
    fk = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fk)
    return fk


def qualify_runtime(fk):
    from openpi.policies import policy_config

    openpi = Path("/mowenjie/fluffy-bassoon/third_party/openpi-215abfb")
    require(Path(policy_config.__file__).resolve() == openpi / "src/openpi/policies/policy_config.py",
            "Wrong OpenPI source")
    policy = parse(fk.checked(motor.OPENPI_CKPT / "READY.json", POLICY_MANIFEST))
    require(str(motor.OPENPI_CKPT) == policy["checkpoint_path"] and
            policy["checkpoint"] == "gs://openpi-assets/checkpoints/pi05_libero", "Wrong motor checkpoint")
    checked_files(motor.OPENPI_CKPT, policy["files"], "name")
    sim = Path(os.environ["LIBERO_PRO_ROOT"])
    manifest = parse(fk.checked(sim.parent / "preparation.json", SIM_MANIFEST))
    require(str(sim) == manifest["simulator_package"] and
            os.environ["CROSSARM_RELEASE"] == manifest["source"], "Wrong simulator setup")
    checked_files(sim, manifest["python_sources"], "path")
    fk.checked(Path(os.environ["LIBERO_CONFIG_PATH"]) / "config.yaml",
               "db0ca804a8f8d8fb1792281bb4d3dcf059967bbe600eff6c01bd452288daf75e")
    return dict(policy_manifest_sha256=POLICY_MANIFEST, policy_files=len(policy["files"]),
                policy_bytes=sum(f["bytes"] for f in policy["files"]), sim_manifest_sha256=SIM_MANIFEST,
                sim_sources=len(manifest["python_sources"]))


def task_conditions(fk):
    # These fields are stable across qualified hosts. Full reset fingerprints are paired in-job.
    history = fk.DATA / FK_RUN / "gpu01"
    conditions = []
    for shard in range(8):
        row = parse((history / f"shard-{shard}/{12*shard:02d}.json").read_bytes())
        reset = row["arms"]["symmeta"]["reset"]
        conditions.append({k: reset[k] for k in RESET_FIELDS})
    encoded = json.dumps(conditions, sort_keys=True, separators=(",", ":")).encode()
    require(fk.digest(encoded) == FK_TASK_CONDITIONS, "Reference noise, prompt or reset contract changed")
    return conditions


def join_predictions(fk, rows, raw):
    predictions = parse(raw)["rows"]
    require(isinstance(predictions, list) and len(predictions) == 96, "Expected 96 prediction rows")
    by_index = {}
    for p in predictions:
        i = p["index"]
        require(type(i) is int and 0 <= i < 96 and i not in by_index and
                type(p["task_id"]) is int and isinstance(p["embodiment"], str),
                f"Invalid or duplicate prediction identity: {i}")
        by_index[i] = p
    for r in rows:
        p = by_index[r["index"]]
        require(fk.identity(r) == fk.identity(p) and r["support"]["sha256"] == p["support_sha256"],
                f"Prediction identity or tape mismatch at {r['index']}")
        for arm in ARMS:
            angle = p[arm + "_yaw_deg"]
            require(type(angle) in (int, float) and math.isfinite(angle), f"Bad {arm} yaw at {r['index']}")
            r[arm + "_yaw_deg"] = angle
        del r["fk_yaw"], r["symmeta_yaw"]
    return rows


def prepare(out, source, sha, fk):
    receipt = parse(fk.checked(HERE / "evidence/prediction-provenance.json", PREDICTION_RECEIPT))
    require(sha == receipt["predictions_sha256"] and receipt["ordered_support_set_sha256"] == fk.SUPPORT_SET
            and receipt["checkpoint"]["sha256"] == fk.CHECKPOINT, "Wrong frozen offline prediction receipt")
    raw = fk.checked(source, sha)
    # Recompute the original panel from all 96 tape bytes, eight manifests and the official schedule.
    with TemporaryDirectory(dir=out) as directory:
        fk.prepare(Path(directory), native)
        original = parse(fk.checked(Path(directory) / "panel.json", FK_PANEL))
    rows = join_predictions(fk, original.pop("rows"), raw)
    conditions = task_conditions(fk)
    runtime = qualify_runtime(fk)
    (out / "predictions.json").write_bytes(raw)
    native.write_json(out / "panel.json", dict(
        rows=rows, task_conditions=conditions, fk_panel_sha256=FK_PANEL,
        fk_task_conditions_sha256=FK_TASK_CONDITIONS, predictions_sha256=sha,
        prediction_receipt_sha256=PREDICTION_RECEIPT, offline_revision=receipt["revision"],
        source_fit_sha256=receipt["fit"]["sha256"],
        original_inputs=original, runtime=runtime))
    print(f"Prepared {len(rows)} paired queries from predictions {sha}", flush=True)


def validate_pair(r, arms, condition, fk):
    require(set(arms) == set(ARMS), "Incomplete or unexpected arm set")
    # The FK validator has two fixed arm labels. Only its argument names are translated.
    fk.validate_pair(dict(r, symmeta_yaw=r["learned_yaw_deg"], fk_yaw=r["linear_yaw_deg"]),
                     dict(symmeta=arms["learned"], fk=arms["linear"]), native)
    reset = arms["learned"]["reset"]
    require({k: reset[k] for k in RESET_FIELDS} == condition and
            "nvidia" in reset["renderer_vendor"].lower(), f"Unexpected query conditions at {r['index']}")
    for rec in arms.values():
        require(rec["released_prompt"] == reset["prompt"] and
                type(rec["terminated"]) is bool and type(rec["truncated"]) is bool and
                rec["terminated"] != rec["truncated"] and
                (not rec["truncated"] or rec["controls"] == reset["horizon"]) and
                (not rec["success"] or rec["terminated"]) and all(map(math.isfinite, rec["native_rewards"])),
                f"Incomplete native outcome at {r['index']}")


def run_shard(out, panel, shard, fk):
    directory = out / f"shard-{shard}"
    directory.mkdir()
    rows = panel["rows"][12 * shard:12 * (shard + 1)]
    condition = panel["task_conditions"][shard]
    policy = native.load_frozen_policy("pi05")
    resets = {}
    with native.UR5eSimulator() as sim:
        sim.call("create", dict(rows[0]["query"], embodiment=rows[0]["embodiment"]))
        for r in rows:
            case = r["query"]
            spec = SimpleNamespace(name=r["embodiment"])
            arms = {}
            for name in ARMS:
                arms[name] = native.run_paired_query(
                    native.run_episode, sim, policy, case, spec, arm_name=name, resets=resets,
                    key=(case["suite"], case["task_id"], case["init_state_id"]),
                    support_physics=r["support"]["physics"], expected_prompt=condition["prompt"],
                    create_attempts=3, yaw_hat=r[name + "_yaw_deg"], scale_hat=r["scale"], grip_hat=r["grip"])
            validate_pair(r, arms, condition, fk)
            native.write_json(directory / f"{r['index']:02d}.json", dict(**r, arms=arms))
            print(f"Pair {r['index']:02d}: learned={arms['learned']['success']} linear={arms['linear']['success']}",
                  flush=True)


def stats(rows):
    return dict(n=len(rows), successes={a: sum(r["arms"][a]["success"] for r in rows) for a in ARMS},
                learned_minus_linear=native.task_clustered_bootstrap(
                    rows, *ARMS, seed=BOOTSTRAP_SEED, draws=BOOTSTRAP_DRAWS))


def merge(out, panel, panel_sha, fk):
    expected = [out / f"shard-{i // 12}/{i:02d}.json" for i in range(96)]
    require(set(out.glob("shard-*/*.json")) == set(expected), "Missing or unexpected pair files")
    rows = []
    for index, (path, r) in enumerate(zip(expected, panel["rows"], strict=True)):
        row = parse(path.read_bytes())
        require(r["index"] == index and r["shard"] == index // 12 and
                {k: v for k, v in row.items() if k != "arms"} == r, f"Wrong pair provenance at {index}")
        validate_pair(r, row["arms"], panel["task_conditions"][r["shard"]], fk)
        if index % 12:
            require(row["arms"]["learned"]["reset"] == rows[-1]["arms"]["learned"]["reset"],
                    f"Reset changed within shard {r['shard']}")
        rows.append(row)
    held = [r for r in rows if r["regime"] in HELD_OUT]
    tasks = sorted({(r["suite"], r["task_id"]) for r in rows})
    require(len(tasks) == 8 and len(held) == 64, "Incomplete held-out panel")
    result = {}
    for label, group, per_task in (("all", rows, 12), ("held_out", held, 8)):
        by_task = {}
        for suite, task_id in tasks:
            sub = [r for r in group if (r["suite"], r["task_id"]) == (suite, task_id)]
            require(len(sub) == per_task, f"Unbalanced task cluster for {label}/{suite}/{task_id}")
            by_task[f"{suite}/{task_id}"] = stats(sub)
        result[label] = dict(**stats(group), by_task=by_task)
    native.write_json(out / "summary.json", dict(
        **result, pairs=96, native_episodes=192, task_clusters=8, bootstrap_draws=BOOTSTRAP_DRAWS,
        bootstrap_seed=BOOTSTRAP_SEED, predictions_sha256=panel["predictions_sha256"], panel_sha256=panel_sha))


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    modes = cli.add_subparsers(dest="mode", required=True)
    pre = modes.add_parser("prepare")
    pre.add_argument("output", type=Path)
    pre.add_argument("predictions", type=Path)
    pre.add_argument("sha256")
    for mode in ("shard", "merge"):
        command = modes.add_parser(mode)
        command.add_argument("output", type=Path)
        command.add_argument("panel_sha256")
        if mode == "shard":
            command.add_argument("shard", type=int, choices=range(8))
    args = cli.parse_args()
    fk = load_fk()
    if args.mode == "prepare":
        prepare(args.output, args.predictions, args.sha256, fk)
    else:
        panel = parse(fk.checked(args.output / "panel.json", args.panel_sha256))
        if args.mode == "shard":
            run_shard(args.output, panel, args.shard, fk)
        else:
            merge(args.output, panel, args.panel_sha256, fk)
