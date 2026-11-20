"""Run the receipt-bound five-condition panel with native, paired UR5e resets."""

import json
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory
from time import monotonic

from large_transfer_gate import CHECK, PULSES, authenticate
from large_transfer_native import PHYSICAL, worker
from large_transfer_stats import (
    CONDITIONS,
    aggregate,
    audited_shard,
    canonical,
    fingerprint,
    native_outcome,
    read,
    require,
    sha,
)


def observed(ev, obs):
    return {
        k: v if k == "prompt" else ev.digest_array(v)
        for k, v in ev.public_obs(obs).items()
    }


def admission(ev, call, slot, layout, prompt):
    obs, info = call("reset", slot)
    if obs is None:
        return None, {
            "layout": layout,
            "status": "physically_unadmitted",
            "reason": info,
        }
    require(
        obs["prompt"] == prompt
        and "NVIDIA" in info["renderer"][0].upper()
        and info["renderer"][1]
        and info["max_steps"] == 300,
        "Unexpected task, renderer or horizon",
    )
    return obs, {
        "layout": layout,
        "status": "admitted",
        "reset": json.loads(canonical(info)),
        "observation": observed(ev, obs),
    }


def collect_context(out, cache, plan, report):
    import adapt_eval as ev
    import numpy as np

    from large_panel_protocol import bind, publish
    from transfer import checked_practice

    reserved = {
        r["row"]
        for phase in ("source_development", "source_confirmation")
        for r in plan[phase]["layouts"]
    }
    requested = {(b, p) for b in ("Panda", "UR5e") for p in (*PULSES, CHECK)}
    for p, b in zip(plan["supports"], plan["donors"], strict=True):
        requested.update((("UR5e", p), (b, p)))
    histories = {}
    for body, pose in sorted(requested):
        key = f"{body}-{pose}"
        entry = cache / f"practice-{key}"
        marker = entry / "receipt.json"
        if not marker.exists():
            with TemporaryDirectory(dir=cache) as work:
                work = Path(work) / "entry"
                work.mkdir()
                try:
                    _, rec = checked_practice(
                        work,
                        body,
                        pose,
                        plan["prompt"],
                        reserved,
                        pulses=pose in PULSES,
                    )
                except (ValueError, RuntimeError) as exc:
                    last = (
                        str(exc)
                        .strip()
                        .rsplit("\n", 1)[-1]
                        .removeprefix("ValueError: ")
                    )
                    if not last.startswith(PHYSICAL):
                        raise
                    rec = {
                        "status": "physically_unadmitted",
                        "body": body,
                        "pose": pose,
                        "reason": last,
                    }
                else:
                    rec["status"] = "admitted"
                bind(work / "receipt.json", rec)
                # The setup lock serializes writers. A reader sees both tape and receipt together.
                work.rename(entry)
        rec = read(marker)
        require((rec["body"], rec["pose"]) == (body, pose), "Misassigned practice")
        report["practice"][key] = rec
        if rec["status"] == "physically_unadmitted":
            require(rec["reason"].startswith(PHYSICAL), "Unknown practice rejection")
            continue
        name = f"practice-{key}.npz"
        require(
            rec["status"] == "admitted" and rec["file"] == name,
            "Invalid practice receipt",
        )
        with publish(out / name) as tmp:
            shutil.copyfile(entry / name, tmp)
            require(sha(tmp) == rec["sha256"], "Changed cached practice")
        with np.load(out / name, allow_pickle=False) as tape:
            histories[body, pose] = ev.History(tape["commands"], tape["motion"], name)
    for p, b in zip(plan["supports"], plan["donors"], strict=True):
        if ("UR5e", p) in histories and (b, p) in histories:
            a, z = (report["practice"][f"{x}-{p}"] for x in ("UR5e", b))
            require(
                a["released_row"] == z["released_row"]
                and a["scene"] == z["scene"]
                and np.array_equal(
                    histories["UR5e", p].commands, histories[b, p].commands
                ),
                "Wrong-body control requires identical practice commands and initial scene",
            )
    responses = {}
    for body in ("Panda", "UR5e"):
        if all((body, p) in histories for p in (*PULSES, CHECK)):
            responses[body] = ev.measured_response([histories[body, p] for p in PULSES])
            report["calibration"][body] = responses[body].report(histories[body, CHECK])
    calibration = None
    if len(responses) == 2:
        for p in (*PULSES, CHECK):
            a, z = (report["practice"][f"{b}-{p}"] for b in ("Panda", "UR5e"))
            require(
                all(a[k] == z[k] for k in ("released_row", "scene", "seed")),
                "Unmatched calibration",
            )
        calibration = ev.privileged(responses["Panda"], responses["UR5e"])
    return histories, calibration


def episode(
    ev,
    call,
    motor,
    transform,
    out,
    plan,
    adm,
    slot,
    condition,
    binding,
    context,
    expected,
    timing,
):
    import numpy as np

    from large_panel_protocol import publish

    layout = adm["layout"]
    before = monotonic()
    obs, now = admission(ev, call, slot, layout, plan["prompt"])
    require(
        now == adm,
        f"Target physical state or first public observation changed on slot {slot}",
    )
    timing.update(reset=monotonic() - before, inference=0.0, simulator=0.0)
    chunks, proposals, actions, motions, rewards, dones, successes = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )
    stride = plan["stride"]
    while len(actions) < 300:
        require(obs["prompt"] == plan["prompt"], "Native instruction changed")
        seed = ev.plan_seed(layout["seed"], len(actions))
        before = monotonic()
        chunk = ev.real_array(
            ev.proposal(motor, ev.public_obs(obs), seed),
            (5, 7),
            "raw proposal",
            bounded=True,
        )
        if not actions:
            opening = {
                "admission": adm,
                "chunk": chunk.tolist(),
                "first_noise_seed": seed,
            }
            require(
                expected is None or opening == expected,
                f"Unmatched first raw chunk on {slot}/{condition}",
            )
        chunks.append(chunk)
        raw = chunk[: min(stride, 300 - len(actions))]
        issued = raw.copy() if transform is None else transform(raw.copy())
        issued = ev.real_array(issued, raw.shape, "issued action", bounded=True)
        timing["inference"] += monotonic() - before
        before = monotonic()
        obs, trace = call("advance", issued.tolist())
        timing["simulator"] += monotonic() - before
        require(1 <= len(trace) <= len(raw), "Invalid native prefix length")
        for j, (motion, reward, done, success) in enumerate(trace):
            actions.append(issued[j])
            proposals.append(raw[j])
            motions.append(motion)
            rewards.append(reward)
            dones.append(done)
            successes.append(success)
        if done or success:
            break
        require(len(trace) == len(raw), "Native process returned an incomplete prefix")
    name = f"query-{slot:03d}-{condition}.npz"
    receipt = {
        "body": "UR5e",
        "condition": condition,
        "slot": slot,
        "binding": binding,
        "context_sha256": context,
        "opening": opening,
        "stride": stride,
    }
    timing.update(controls=len(actions), replans=len(chunks))
    with publish(out / name) as tmp:
        with tmp.open("wb") as f:
            np.savez_compressed(
                f,
                receipt=json.dumps(receipt, sort_keys=True, allow_nan=False),
                chunks=chunks,
                actions=actions,
                proposals=proposals,
                motion=motions,
                rewards=rewards,
                done=dones,
                success=successes,
            )
        rec = {"file": name, "sha256": sha(tmp)}
        native_outcome(tmp, rec, binding, slot, condition, adm, stride, context)
    return rec, opening


def query_case(
    ev,
    call,
    policy,
    transform,
    args,
    cache,
    plan,
    adm,
    i,
    condition,
    binding,
    context,
    reference,
    cached,
):
    from large_panel_protocol import bind, lock, publish

    started = monotonic()
    name = f"query-{i:03d}-{condition}.npz"
    tape = cache / name
    marker = tape.with_suffix(".json")
    evidence = (binding, i, condition, adm, plan["stride"], context)
    with lock(cache / f"case-{i:03d}.lock"):
        if cached:
            require(cached["file"] == name, "Misassigned prior native episode")
            with publish(tape) as tmp:
                shutil.copyfile(args.reuse / name, tmp)
                _, opening = native_outcome(tmp, cached, *evidence)
                require(
                    reference is None or reference == opening, "Different prior opening"
                )
        timing = {"reused": tape.exists()}
        if timing["reused"]:
            rec = (
                read(marker) if marker.exists() else {"file": name, "sha256": sha(tape)}
            )
            require(rec["file"] == name, "Misassigned cached episode")
            _, opening = native_outcome(tape, rec, *evidence)
            require(
                reference is None or reference == opening, "Different cached opening"
            )
        else:
            require(not marker.exists(), "Missing published native evidence")
            rec, opening = episode(
                ev,
                call,
                policy,
                transform,
                cache,
                plan,
                adm,
                i,
                condition,
                binding,
                context,
                reference,
                timing,
            )
        bind(marker, rec)
    with publish(args.out / name) as tmp:
        shutil.copyfile(tape, tmp)
        require(sha(tmp) == rec["sha256"], "Changed copied episode")
    timing["elapsed"] = monotonic() - started
    return rec, opening, timing


def evaluate(args):
    started = monotonic()
    plan = authenticate(args)
    count = getattr(args, "shard_count", 1)
    index = getattr(args, "shard_index", 0)
    require(0 <= index < count <= len(plan["layouts"]) == 100, "Invalid target shard")
    auth_seconds = monotonic() - started
    assigned = list(range(index, len(plan["layouts"]), count))

    import adapt_eval as ev

    from large_panel_protocol import bind, lock, publish
    from large_transfer_fit import PanelPolicy, frozen, restore

    checkpoint = args.root / "accepted/checkpoint.pt"
    args.out.mkdir(parents=True, exist_ok=False)
    binding = fingerprint(plan)
    report = {
        "status": "waiting_for_source_fit_verification",
        "binding": binding,
        "slots": [{} for _ in range(100)],
        "practice": {},
        "calibration": {},
        "gate_verified_before_target": False,
        "revision": args.revision,
        "receipt_sha256": args.receipt_sha256,
        "shard": {"index": index, "count": count, "slots": assigned},
        "timing_seconds": {"authenticate": auth_seconds, "queries": {}},
    }
    report_path = args.out / "report.json"
    ev.write_json(report_path, report)
    with frozen(checkpoint, args.weights) as (motor, identity):
        net = restore(
            args.joint,
            identity,
            {
                r[k]
                for phase in ("source_development", "source_confirmation")
                for r in plan[phase]["layouts"]
                for k in ("seed", "row", "xy", "scene", "joint_xy")
            },
        )
        require(
            identity["motor_sha256"] == plan["motor_sha256"]
            and sha(args.joint / "reader-adapter.pt") == plan["reader_sha256"]
            and sha(args.joint / "report.json") == plan["fit_report_sha256"],
            "Source fit changed after review",
        )
        report["timing_seconds"]["fit_verification_and_load"] = (
            monotonic() - started - auth_seconds
        )
        with lock(args.root / "accepted/.large-target.lock"):
            bind(args.root / "accepted/large-target.json", {"binding": binding})
        plan_path = args.out / "target-binding.json"
        bind(plan_path, plan)
        with publish(args.out / "review.json") as tmp:
            shutil.copyfile(args.receipt, tmp)
            require(
                sha(tmp) == args.receipt_sha256 and read(tmp) == plan,
                "Source receipt changed",
            )
        report.update(
            status="admitting_target_layouts", gate_verified_before_target=True
        )
        ev.write_json(report_path, report)
        policy = PanelPolicy(motor)
        with worker(args.root, plan_path) as call:
            cache = args.root / "accepted" / "large-target-cache" / binding
            waiting = monotonic()
            with lock(cache / "setup.lock"):
                setup = monotonic()
                report["timing_seconds"]["setup_lock_wait"] = setup - waiting
                admissions = []
                for i, layout in enumerate(plan["layouts"]):
                    path = cache / f"admission-{i:03d}.json"
                    if not path.exists():
                        _, adm = admission(ev, call, i, layout, plan["prompt"])
                        bind(path, adm)
                    adm = read(path)
                    require(
                        adm["layout"] == layout, "Changed cached admission ordering"
                    )
                    admissions.append(adm)
                bind(args.out / "admission.json", admissions)
                report["admission_sha256"] = sha(args.out / "admission.json")
                bind(
                    args.root / "accepted/large-target-admission.json",
                    {"binding": binding, "admission": report["admission_sha256"]},
                )
                report["timing_seconds"]["admission"] = monotonic() - setup
                context_started = monotonic()
                if any(adm["status"] == "admitted" for adm in admissions):
                    histories, calibrated = collect_context(
                        args.out, cache, plan, report
                    )
                report["context_sha256"] = fingerprint(
                    {
                        "practice": report["practice"],
                        "calibration": report["calibration"],
                    }
                )
                bind(
                    args.root / "accepted/large-target-context.json",
                    {"binding": binding, "context": report["context_sha256"]},
                )
                report["timing_seconds"]["context"] = monotonic() - context_started
            previous = None
            if args.reuse is not None:
                previous = read(args.reuse / "report.json")
                require(
                    all(
                        previous[k] == report[k]
                        for k in (
                            "binding",
                            "receipt_sha256",
                            "admission_sha256",
                            "context_sha256",
                        )
                    )
                    and len(previous["slots"]) == len(admissions),
                    "Prior partial run has different pairing inputs",
                )
            report["status"] = "querying"
            ev.write_json(report_path, report)
            for i in assigned:
                adm = admissions[i]
                if adm["status"] != "admitted":
                    require(
                        adm["status"] == "physically_unadmitted",
                        "Unexpected admission status",
                    )
                    bind(args.out / f"case-{i:03d}.json", {})
                    continue
                p, donor = plan["supports"][i], plan["donors"][i]
                transforms = {
                    "raw": None,
                    "empty": ev.learned(net, None),
                    "calibration": calibrated,
                }
                for c, b in (("correct", "UR5e"), ("wrong", donor)):
                    transforms[c] = (
                        ev.learned(net, histories[b, p])
                        if (b, p) in histories
                        else None
                    )
                reference = None
                for c in CONDITIONS:
                    if c != "raw" and transforms[c] is None:
                        report["slots"][i][c] = {
                            "missing": "predeclared practice physically unadmitted"
                        }
                    else:
                        cached = previous["slots"][i].get(c) if previous else None
                        rec, opening, timing = query_case(
                            ev,
                            call,
                            policy,
                            transforms[c],
                            args,
                            cache,
                            plan,
                            adm,
                            i,
                            c,
                            binding,
                            report["context_sha256"],
                            reference,
                            cached,
                        )
                        report["timing_seconds"]["queries"][rec["file"]] = timing
                        report["slots"][i][c] = rec
                        if c == "raw":
                            reference = opening
                    ev.write_json(report_path, report)
                bind(args.out / f"case-{i:03d}.json", report["slots"][i])
    report["timing_seconds"]["before_aggregate"] = monotonic() - started
    report["status"] = "shard_complete" if count > 1 else "complete"
    ev.write_json(report_path, report)
    audit_started = monotonic()
    try:
        if count > 1:
            audited_shard(args.out)
            summary = {"status": "shard_complete", "slots": assigned}
        else:
            summary = aggregate(args.out)
            bind(args.out / "summary.json", summary)
    except Exception:
        report["status"] = "audit_failed"
        ev.write_json(report_path, report)
        raise
    report["timing_seconds"]["audit"] = monotonic() - audit_started
    report["timing_seconds"]["total"] = monotonic() - started
    ev.write_json(report_path, report)
    return summary
