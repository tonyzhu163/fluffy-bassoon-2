"""Read back archived reward, issued-command, history and noise tapes before scoring pairs."""
import argparse
import json
from pathlib import Path
import numpy as np
from panel import MODES, TASKS, cases, key, require, save
from ur5e_motor import NOISE_SEED_OFFSET, array_digest


def f32(x):
    return np.asarray(x, dtype=np.float32)


def validate(row, fixture):
    case, r, calls = row["case"], row["native"], row["calls"]
    mode, env, reset = row["mode"], row["environment"], r["reset"]
    layout = case["init_state_id"]
    _, horizon, prompt, fingerprints = TASKS[case["suite"], case["task_id"]]
    require(env["released_row_fingerprint"] == fingerprints[layout-46] and reset["horizon"] == horizon
            and reset["prompt"] == prompt and r["released_prompt"] == prompt, "registered release")
    rewards = r["native_rewards"]
    n = r["controls"]
    require(row["fixture"] == fixture and len(rewards) == n and 0 < n <= horizon and
            all(v in (0, 1) for v in rewards) and sum(rewards) == r["native_return"]
            and bool(max(rewards)) == r["success"], "native reward tape")
    if fixture:
        require(n == 15, "short fixture controls")
    else:
        require(r["terminated"] or r["truncated"], "missing terminal flag")
        require(n == horizon or r["terminated"] and r["success"], "failure before released horizon")
    bank = np.random.default_rng(case["seed"] + NOISE_SEED_OFFSET).standard_normal(
        ((horizon + 4)//5, 10, 32)).astype(np.float32)
    require(reset["noise_sha256"] == array_digest(bank), "full bank digest")
    previous = np.zeros(52, np.float32)
    xyz = f32(row["initial_xyz"])
    width = abs(np.diff(f32(row["initial_fingers"]))[0])
    slots = np.zeros((4, 13), np.float32)
    t = 0
    for call in calls:
        current = slots.flatten()
        np.testing.assert_array_equal(current, f32(call["history"]))
        used = {"joint_zero": np.zeros_like(current), "joint_lagged": previous}.get(mode, current)
        require(call["used"] == array_digest(used) and call["noise"] == array_digest(bank[t//5])
                and call["t"] == t, "history routing or noise index")
        if t == 0:
            require(call["obs"] == reset["motor_frame0"] and call["proposal"] == r["opening_proposal_sha256"],
                    "opening observation or proposal")
        issued = f32(call["issued"])
        require(len(issued) == min(5, n-t) and issued.shape[1:] == (7,), "executed prefix length")
        np.testing.assert_array_equal(issued, np.clip(f32(call["proposed_prefix"])[:len(issued)], -1, 1))
        for action, pos, fingers in zip(issued, call["positions"], call["fingers"], strict=True):
            pos = f32(pos)
            next_width = abs(np.diff(f32(fingers))[0])
            slots[:-1] = slots[1:]
            slots[-1] = np.r_[action, pos-xyz, next_width-width, next_width, 1]
            xyz, width = pos, next_width
        previous = current
        t += len(issued)
    require(t == n, "incomplete executed trajectory")
    if mode == "original":
        p = row["parity"]
        require(p["shape"] == [10, 7] and p["max_abs"] == 0 and p["stock_sha256"] == p["noop_sha256"],
                "stock/no-op decoded action parity")
        require(p.get("deployed_noop_sha256", p["noop_sha256"]) == p["stock_sha256"] and
                p.get("deployed_noop_max_abs", p["max_abs"]) == 0, "deployed no-op parity")
    if mode == "joint" and n > 10:
        m = row["mechanism"]
        require(m["at_control"] == 10 and m["proposal"] == calls[2]["proposal"]
                and m["current"] == calls[2]["used"], "identical-input joint check")
        expected = {"zero": array_digest(np.zeros(52, np.float32)), "lagged": calls[1]["used"]}
        require(all(m["alternatives"][s]["history"] == h for s, h in expected.items()), "diagnostic history")
    return dict(success=r["success"], native_return=r["native_return"], controls=n,
                opening_proposal=r["opening_proposal_sha256"])


def first_change(calls, reference):
    """Compare acquired trajectories only up to the first observed difference in a given quantity."""
    proposal_at = None
    issued_at = None
    for a, b in zip(calls, reference):
        require(a["t"] == b["t"] and a["noise"] == b["noise"], "action comparisons require paired noise")
        if proposal_at is None and a["proposal"] != b["proposal"]:
            proposal_at = a["t"]
        aa, bb = f32(a["issued"]), f32(b["issued"])
        count = min(len(aa), len(bb))
        differed = np.flatnonzero(np.any(aa[:count] != bb[:count], axis=1))
        if issued_at is None and len(differed):
            issued_at = a["t"] + int(differed[0])
        if proposal_at is not None and issued_at is not None:
            break
    return dict(first_different_proposal_replan=proposal_at, first_different_issued_control=issued_at,
                index_base=0)


def summarize(out, fixture):
    before = json.loads((out/"frozen-before.json").read_text())
    after = json.loads((out/"frozen-after.json").read_text())
    require(before == after, "deployed parameters, checkpoint or normalization changed")
    rows, kinova = [], {}
    for c in cases():
        casekey = key(c)
        modes = {}
        opening = {}
        tapes = {}
        for mode in MODES:
            v = json.loads((out/"episodes"/f"{casekey}-{mode}.json").read_text())
            require(v["case"] == c and v["mode"] == mode, "incorrect join key")
            modes[mode] = validate(v, fixture)
            if mode == "original":
                reference, env, body = v["native"]["reset"], v["environment"], v["body"]
            else:
                require(v["native"]["reset"] == reference and v["environment"] == env and v["body"] == body,
                        "mismatched within-arm reset/observation/noise")
            opening[mode] = modes[mode]["opening_proposal"]
            tapes[mode] = v["calls"]
            if mode == "joint":
                mechanism = v["mechanism"]
            if mode == "original":
                parity = v["parity"]
        require(len({opening[m] for m in ("joint", "joint_zero", "joint_lagged")}) == 1,
                "joint checkpoints diverged at empty history")
        require(opening["adapter_only"] == opening["original"], "bias-free adapter changed empty-history opening")
        first_changes = {m: first_change(tapes["joint"], tapes[m]) for m in MODES if m != "joint"}
        common = [env[k] for k in ("scene_fingerprint", "released_row_fingerprint", "bddl_sha256",
                                   "perturbation_plan_fingerprint")] + [
            reference[k] for k in ("horizon", "prompt", "noise_sha256", "action_schema")]
        scene = (c["suite"], c["task_id"], c["init_state_id"])
        if c["arm"] == "Panda":
            require(common == kinova[scene], "unmatched Panda subset")
        else:
            kinova[scene] = common
        rows.append(dict(case=c, modes=modes, mechanism=mechanism, parity=parity, first_changes=first_changes,
                         released_row=env["released_row_fingerprint"], physics=reference["physics"],
                         noise=reference["noise_sha256"]))
    require(len(rows) == 18 and len(list((out/"episodes").glob("*.json"))) == 108, "panel completeness")
    result = dict(fixture=fixture, starts=len(rows), episodes=len(rows)*len(MODES), source_lock=before,
                  manifest=json.loads((out/"run.json").read_text()), rows=rows)
    if not fixture:
        result["arms"] = {arm: statistics([r for r in rows if r["case"]["arm"] == arm])
                          for arm in ("Kinova3", "Panda")}
        shared_arms = [r for r in rows if r["case"]["init_state_id"] >= 48]
        result["matched_arms"] = {
            m: [dict(task=[k["case"]["suite"], k["case"]["task_id"]], layout=k["case"]["init_state_id"],
                     kinova_effect=int(k["modes"]["joint"]["success"])-int(k["modes"][m]["success"]),
                     panda_effect=int(p["modes"]["joint"]["success"])-int(p["modes"][m]["success"]))
                for k, p in zip(shared_arms[::2], shared_arms[1::2], strict=True)]
            for m in MODES if m != "joint"}
    return result


def statistics(rows):
    # The selected tasks are fixed. Resample paired rows within each task, keeping all six treatments together.
    tasks = np.array([[[int(r["modes"][m]["success"]) for m in MODES]
                       for r in rows if (r["case"]["suite"], r["case"]["task_id"]) == task]
                      for task in TASKS])
    t, l, _ = tasks.shape
    rng = np.random.default_rng(20261124)
    indices = rng.integers(l, size=(20000, t, l))
    sample_means = tasks[np.arange(t)[None, :, None], indices].mean(axis=(1, 2))
    observed = tasks.mean(axis=(0, 1))
    def interval(x):
        return np.quantile(x, [.025, .975]).tolist()
    j = MODES.index("joint")
    contrasts = {}
    for k, mode in enumerate(MODES):
        if k != j:
            delta = tasks[:, :, j] - tasks[:, :, k]
            contrasts[mode] = dict(joint_minus=float(delta.mean()), rescues=int((delta == 1).sum()),
                                   losses=int((delta == -1).sum()), pairs=int(delta.size),
                                   paired_bootstrap_95=interval(sample_means[:, j]-sample_means[:, k]))
    factorial = np.array([1, -1, -1, 1, 0, 0])
    return dict(starts=len(rows), successes=dict(zip(MODES, tasks.sum(axis=(0, 1)).tolist())), joint_vs=contrasts,
                interaction=float(observed @ factorial), interaction_bootstrap_95=interval(sample_means @ factorial),
                interval_method="Paired percentile bootstrap resampling starts within the three fixed tasks; "
                                "conditional on these tasks, check-point and noise banks. Small panels can yield zero-width intervals.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("out", type=Path)
    p.add_argument("--fixture", action="store_true")
    a = p.parse_args()
    result = summarize(a.out, a.fixture)
    target = a.out/("cpu-summary.json" if a.fixture else "native-summary.json")
    if target.exists():
        require(json.loads(target.read_text()) == result, "stored summary differs from complete readback")
    else:
        save(target, result)
    print(f"Verified {result['episodes']} episodes on {result['starts']} paired starts.")
