"""Baige Linux-CPU qualification only. No policy inference and no scored rollouts."""
import argparse
import copy
from contextlib import redirect_stdout
from io import StringIO
import gzip
import os
from unittest.mock import patch
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

from paired_inputs import CROSS, ROOT, native
from paired_records import (PANELS, SEEDS, atomic, checked, contract, counts, describe, digest, encode,
                            join, load_predictions, pair_path, parse, read_pair, reduce_attempt, require,
                            validate_arms, write_pair)
from live import adopt, episode, reusable, simulator, warmup

HISTORY = ROOT/"outputs/symmeta-fk-ur5e-20261123-r1/gpu01"


def fixture(out):
    rows = []
    for p in PANELS:
        if p == "gpu09test":
            archived = parse((HISTORY/"panel.json").read_bytes())["rows"]
        else:
            archived = parse(gzip.decompress((CROSS/f"evidence/{p}-cases.json.gz").read_bytes()))["rows"]
        for r in archived:
            sha = r["support"]["sha256"] if "support" in r else r["support_sha256"]
            for s in SEEDS:
                c = dict(yaw_deg=float(s), xy_scale=1.0, gripper_sign=1)
                rows.append(dict(**{k: r[k] for k in ("index", "suite", "task_id", "embodiment")}, panel=p,
                                 tape_sha256=sha, seed=s, tuned=c, symmeta=c))
    raw = encode(dict(schema=1, qualification_fixture=True, rows=rows[::-1]))
    atomic(out/"predictions.json", raw)
    print(digest(raw))


def dry_reset(ev, sim, out, d, idx, correction):
    import numpy as np

    panel = d["panel"]
    r = dict(d["rows"][3*idx], tuned=correction, symmeta=correction)
    ref = d["refs"][idx//12]
    require(os.environ["LIBGL_ALWAYS_SOFTWARE"] == "1" and os.environ["CUDA_VISIBLE_DEVICES"] == "",
            "CPU-only qualification")
    items, resets = [], {}

    class FirstInput(Exception):
        pass

    class Capture:
        _model = SimpleNamespace(action_horizon=50 if panel == "gpu08pi0fix" else 10, action_dim=32)

        def infer(self, item, *, noise):
            items.append(({k: ev.array_digest(v) for k,v in item.items() if k != "prompt"},
                          item["prompt"], ev.array_digest(noise)))
            raise FirstInput

    first = warmup(sim, r, ref, d["model_identities"][idx//12], cpu=True)
    for reader in ("tuned", "symmeta"):
        try:
            episode(ev, sim, Capture(), r, reader, ref, resets)
        except FirstInput:
            pass
        else:
            raise ValueError("Expected to stop at first motor input")
    reset, = resets.values()
    from paired_records import SETTINGS
    require(all(reset[k] == ref[k] for k in SETTINGS) and len(items) == 2 and items[0] == items[1]
            and items[0][0] == reset["motor_frame0"] and items[0][1] == reset["prompt"]
            and items[0][2] == ev.array_digest(np.random.default_rng(ref["noise_seed"]).standard_normal(
                (Capture._model.action_horizon, 32)).astype(np.float32)), "Original motor input/noise mismatch")
    statement = dict(panel=panel, index=idx, arm=reset["arm"], model_class=reset["model_class"],
                     policy=d["runtime"]["policy"], renderer=first["renderer"], query=r["query"],
                     two_native_resets_equal=True, first_policy_inputs_equal=True,
                     query_noise_sha256=reset["noise_sha256"], first_noise_sha256=items[0][2],
                     cpu_physics=reset["physics"], support_physics=r["support_physics"], scored_episodes=0)
    if panel == "gpu08pi0fix":
        q = first["observation"]["robot0_eef_quat_xyzw"]
        statement["query_quaternion_flipped_at_warmup"] = bool(abs(q[3]) < 0.85 and q[0] < 0)
        # Both signs of this legal orientation must yield the positive-x motor convention.
        u = np.array([-0.8, 0, 0, 0.6])
        states = []
        for sign in (1, -1):
            obs = dict(first["observation"], robot0_eef_quat_xyzw=sign*u)
            states.append(ev.to_openpi_observation(obs, ref["prompt"], canonicalize_quat=True)["observation/state"])
        np.testing.assert_array_equal(*states)
        require(states[0][3] > 0, "pi0 sign correction")
        statement["quaternion_double_cover_control"] = True
    atomic(out/panel/f"cpu-reset-{idx//12}.json", encode(statement))
    print(statement, flush=True)


def all_resets(out, panel):
    ev = native()
    d = parse((out/panel/"panel.json").read_bytes())
    if panel == "gpu09test":
        corrections = [dict(yaw_deg=r["symmeta_yaw"], xy_scale=r["scale"], gripper_sign=r["grip"])
                       for r in parse((HISTORY/"panel.json").read_bytes())["rows"]]
    else:
        archived = parse(gzip.decompress((CROSS/f"evidence/{panel}-cases.json.gz").read_bytes()))
        corrections = [dict(yaw_deg=r["yaw_hat"], xy_scale=r["scale_hat"], gripper_sign=r["grip_hat"])
                       for r in archived["rows"]]
    with simulator(ev, cpu=True) as sim:
        for shard in range(8):
            # Vary embodiment across tasks, including the original Goal6 recovery.
            idx = shard*12 + shard
            dry_reset(ev, sim, out, d, idx, corrections[idx])


def rejected(f):
    try:
        f()
    except (ValueError, KeyError):
        return
    raise AssertionError("Invalid data accepted")


def reduction_fixture(out, scratch):
    # Test the reducer independently of rollout validation. Temporary empty path
    # markers model a live directory. No synthetic trajectories are serialized.
    atomic(scratch/"contract.json", (out/"contract.json").read_bytes())
    paths = []
    for panel in PANELS:
        raw = (out/panel/"panel.json").read_bytes()
        atomic(scratch/panel/"panel.json", raw)
        for r in parse(raw)["rows"]:
            path = pair_path(scratch, r)
            paths.append(path)
            atomic(path, b"")

    def already_validated(directory, row, ref, sha, receipts):
        # Distinct process openings for distinct seeds are valid when each pair matches.
        record = dict(row=row, execution_sha256=str(row["seed"]), arms={a: dict(reset=ref,
                      opening_proposal_sha256=digest(str(row["seed"]).encode()),
                      success=(a == "tuned" and row["seed"] == 43)) for a in ("tuned", "symmeta")})
        return None, record

    with patch("paired_records.read_pair", side_effect=already_validated), redirect_stdout(StringIO()):
        paths[-1].unlink()
        partial = reduce_attempt(scratch, partial=True)
        require(partial["completed_pairs"] == 1151 and not partial["complete"] and
                partial["panels"][PANELS[-1]]["missing_index_seed"] == [[95,43]], "Partial reduction")
        rejected(lambda: reduce_attempt(scratch))
        atomic(paths[-1], b"")
        summary = reduce_attempt(scratch)
        require(summary["complete"] and summary["completed_pairs"] == 1152 and
                all(v["overall"]["successes"]["tuned"] == 96 and v["ood"]["successes"]["tuned"] == 64
                    and all(x["opening"] == 3 for x in v["task_distinct_fingerprints"].values())
                    for v in summary["panels"].values()), "Complete reduction")
        atomic(scratch/"pairs/extra.json", b"")
        rejected(lambda: reduce_attempt(scratch))


def checks(out):
    sha, c = contract(out)
    raw = checked(out/"predictions.json", c["predictions_sha256"])
    rejected(lambda: load_predictions(out/"predictions.json", digest(raw)))
    preds = load_predictions(out/"predictions.json", digest(raw), allow_fixture=True)
    panel = PANELS[0]
    d = parse(checked(out/panel/"panel.json", c["panels"][panel]))
    cases = d["rows"][::3]
    for field, value in (("task_id", 999), ("embodiment", "wrong"), ("tape_sha256", "0"*64), ("suite", "wrong")):
        modified = preds.copy()
        key = panel, 0, 41
        modified[key] = dict(preds[key], **{field: value})
        rejected(lambda: join(panel, cases, modified))
    for i, case in enumerate(cases):
        past = parse((HISTORY/f"shard-{i//12}/{i:02d}.json").read_bytes())["arms"]
        arms = dict(tuned=past["fk"], symmeta=past["symmeta"])
        row = dict(case, **{a: rec["inferred"] for a, rec in arms.items()})
        validate_arms(row, arms, d["refs"][i//12])
    partial = reduce_attempt(out, partial=True)
    require(partial["completed_pairs"] == 0 and not partial["complete"], "Partial prepare reduction")
    rejected(lambda: reduce_attempt(out))
    with TemporaryDirectory(dir=out) as tmp:
        base = Path(tmp)
        for reason in ("missing", "duplicate", "seed", "sign", "scale", "nan"):
            obj = parse(raw)
            row = obj["rows"][0]
            if reason == "missing": obj["rows"].pop()
            if reason == "duplicate": obj["rows"][-1] = row
            if reason == "seed": row["seed"] = 44
            if reason == "sign": row["tuned"]["gripper_sign"] = True
            if reason == "scale": row["symmeta"]["xy_scale"] = -1
            raw_bad = encode(obj)
            if reason == "nan": raw_bad = raw_bad.replace(b'"xy_scale":1.0', b'"xy_scale":NaN', 1)
            atomic(base/reason, raw_bad)
            rejected(lambda: load_predictions(base/reason, digest(raw_bad), allow_fixture=True))
        # Reuse the two complete native records from an earlier experiment, only as a
        # validator fixture. Restore their TRUE corrections before exercising validation.
        hist = parse((HISTORY/"shard-0/00.json").read_bytes())
        arms = dict(tuned=hist["arms"]["fk"], symmeta=hist["arms"]["symmeta"])
        ref = d["refs"][0]
        row = dict(cases[0], **{a:v["inferred"] for a,v in arms.items()})
        validate_arms(row, arms, ref)
        for field, val in (("opening_proposal_sha256", "d"*64), ("success", 1), ("native_return", 1e6)):
            bad = copy.deepcopy(arms)
            bad["tuned"][field] = val
            rejected(lambda: validate_arms(row, bad, ref))
        for field in ("physics", "noise_sha256"):
            bad = copy.deepcopy(arms)
            bad["symmeta"]["reset"][field] = "b"*64
            rejected(lambda: validate_arms(row, bad, ref))
        other = base/"source"; dest = base/"dest"
        atomic(other/"contract.json", (out/"contract.json").read_bytes())
        origin = dict(contract_sha256=sha, hostname="job-cpu-qa", devices=[dict(allocation_ordinal=0,
                      cuda_name="fixture", cuda_total_bytes=1)], software={"role":"qualification fixture"})
        cert = digest(encode(origin))
        atomic(other/"receipts"/(cert+".json"), encode(origin))
        write_pair(other, row, arms, ref, sha, cert, 0)
        available = reusable([other], [row], ref, sha, origin, 0)
        require(len(available) == 1 and not reusable([other], [row], ref, "0"*64, origin, 0)
                and not reusable([other], [row], ref, sha, dict(origin, software={}), 0), "Compatible reuse")
        require(len(adopt(dest, [row], available)) == 1 and
                read_pair(dest,row,ref,sha,{})[0] == pair_path(other,row).read_bytes(), "Pair transfer")
        raw_pair = parse(pair_path(other,row).read_bytes())
        raw_pair["pair"]["arms"]["symmeta"]["success"] ^= True
        atomic(pair_path(other,row), encode(raw_pair), replace=True)
        rejected(lambda: read_pair(other, row, ref, sha, {}))
        require(not reusable([other], [row], ref, sha, origin, 0), "Corrupt prior record accepted")
        atomic(dest/"contract.json", (out/"contract.json").read_bytes())
        changed = copy.deepcopy(arms)
        changed["tuned"]["seconds"] += 1
        pair_path(other, row).unlink()
        write_pair(other, row, changed, ref, sha, cert, 0)
        require(not reusable([other, dest], [row], ref, sha, origin, 0), "Conflicting histories accepted")
        reduction_fixture(out, base/"reducer-only")

    # Pure arithmetic fixture: all cases in a task move together across all seeds.
    sample = [dict(row=r, arms=dict(tuned={"success": r["shard"]%2==0}, symmeta={"success": False})) for r in d["rows"]]
    for split, n in ((sample,96), ([b for b in sample if b["row"]["regime"] != "familiar"],64)):
        s = describe(split,n)
        require(s["n"] == n*3 and s["tuned_minus_symmeta_pp"] == 50 and s["task_cluster_ci95_pp"] == [12.5,87.5]
                and s["seed_sd_pp"]["tuned_minus_symmeta"] == 0 and not describe(split[:-1],n)["complete"], s)
    varied = copy.deepcopy(sample)
    for b in varied:
        r = b["row"]
        b["arms"]["tuned"]["success"] = r["seed"] == 43 or (r["seed"] == 42 and r["shard"]%2 == 0)
    require(describe(varied, 96)["seed_sd_pp"]["tuned_minus_symmeta"] == 50, "Sample SD across seeds")
    for lanes in (4,8):
        assignments = [t for lane in range(lanes) for t in range(lane,32,lanes)]
        require(sorted(assignments) == list(range(32)), "Incomplete lane routing")
    print("PASS: keyed join, strict schema, native record pairing, atomic reuse, corruption, "
          "partial/full gates, task bootstrap, seed statistics and 4/8-lane partition. No scored episodes.")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("fixture", "reset", "checks"))
    p.add_argument("out", type=Path)
    p.add_argument("--panel", choices=PANELS)
    a = p.parse_args()
    if a.mode == "fixture": fixture(a.out)
    elif a.mode == "reset": all_resets(a.out, a.panel)
    else: checks(a.out)
