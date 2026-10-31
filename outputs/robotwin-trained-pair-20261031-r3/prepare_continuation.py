"""Static owner metadata construction; never admits or starts a runtime.

Requires independently audited immutable predecessor and its separately
verified terminal UI receipt. Actual Linux preflight still validates live
ownership, canonical locks, checkpoints, configuration and remaining quotas.
"""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import tempfile


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("previous")
    parser.add_argument("next")
    args = parser.parse_args()
    old = Path("outputs") / args.previous
    new = Path("outputs") / args.next
    assert args.previous.startswith("robotwin-trained-pair-") and args.next.startswith("robotwin-trained-pair-")
    assert not new.exists(), "A new owner identity is required"
    terminal = json.loads((old / "receipts/scheduler-terminal.json").read_text())
    assert terminal["state"] == terminal["pod_state"] == "Succeeded" and terminal["gpus"] == 1
    seconds = terminal["actual_scheduler_gpu_seconds"]
    assert seconds == (datetime.fromisoformat(terminal["end_shanghai"]) - datetime.fromisoformat(terminal["start_shanghai"])).total_seconds()
    assert 0 < seconds <= 2700
    independent = json.loads((old / "receipts/independent-final-audit.json").read_text())
    assert independent["status"] == "passed_one_genuine_trained_N6_round_per_original_arm"
    assert independent["allocation_id"] == terminal["job"]
    audit = json.loads((old / "receipts/audit.json").read_text())
    assert audit["status"] == "passed_one_trained_round_each"
    assert all(0 < arm["completed_iterations"] < 100 for arm in audit["arms"].values())
    new.mkdir()
    remote = Path("/mowenjie/fluffy-bassoon-2/inputs") / args.next

    def save(name, value):
        path = new / name
        path.write_text(json.dumps(value, indent=2) + "\n")
        return dict(path=str(remote / name), sha256=sha(path))

    terminal_ref = save("predecessor-scheduler-terminal.json", terminal)
    methods = {method: dict(released=True, allocation_id=terminal["job"], terminal_state="Succeeded",
        terminal_receipt=terminal_ref, manifest=arm["next_manifest"], checkpoint=arm["checkpoint"],
        canonical_head=arm["final_canonical_head"], owner_boundary_audit=dict(
            path=f"/mowenjie/fluffy-bassoon-2/outputs/{args.previous}/audit.json",
            sha256=sha(old / "receipts/audit.json"))) for method, arm in audit["arms"].items()}
    release = save("owner-release.json", dict(schema="robotwin_trained_pair_owner_release_v1",
        runtime_commit="57016ba08eed79043fd67bf8b085bd6d42e6d766",
        parent_manifest_sha256="1f80c08310dfa4543ec7a748ca6e2bf9aad589cd55978a402210abb0d83bdbc0", methods=methods))
    authority = save("authority.json", dict(schema="robotwin_original_goal_continuation_authority_v1",
        source_goal_rounds=100, original_total_compute_ceiling_seconds=None,
        scope="Continue one original source round per trained arm while the continuous runner is qualified; same570/c353, no new pilot/tests/fresh learners/scientific scope.",
        human_time_ceiling_supersession="wait dont limit time like this，how much are we aiming to collect for either",
        current_protocol="outputs/robotwin-wholetrial-20261030-r1/PROTOCOL.md",
        measured_predecessor=dict(allocation_id=terminal["job"], actual_scheduler_gpu_seconds=seconds,
            new_source_actions=audit["total_new_source_actions"]), qualification_capacity_reused=False))
    previous_cost = json.loads((old / "prior-charge-and-physical-snapshot.json").read_text())
    known = previous_cost["known_terminal_gpu_seconds_including_predecessor_pair"] + seconds
    physical = save("prior-charge-and-physical-snapshot.json", dict(
        schema="robotwin_trained_pair_actual_charges_snapshot_v1", owner_release=release,
        known_terminal_gpu_seconds_including_predecessor_pair=known, full_cumulative_gpu_seconds=None,
        unknowns="Historical optimization630GPU-s reported but not independently timed here; active SAC continuation not terminal charged.",
        original_source_goal_rounds=100, qualification_capacity_reused=False,
        separate_active_sac_allocation="job-gv77bzhyeroo", previous_unused_gpu_seconds_sealed=2700-seconds,
        methods={method: dict(source_rounds=arm["completed_iterations"], source_actions=arm["trained_boundary"]["source_steps"],
            canonical_head=arm["final_canonical_head"], remaining_total_actions=
            {"ppo":1226400,"ppo_reset":1168800}[method]-arm["final_canonical_head"]["charged_actions"],
            remaining_source_safety_actions=1100000-arm["trained_boundary"]["source_steps"]) for method,arm in audit["arms"].items()}))
    resources = dict(gpus=1, cpus=32, memory_gib=192, shared_memory_gib=16, scheduler_cap_seconds=2700)
    compute = save("compute-binding.json", dict(schema="robotwin_trained_pair_compute_v1",
        campaign_id="robotwin-native-wholetrial-20261030-r2", original_total_compute_ceiling_seconds=None,
        source_goal_rounds=100, admitted_scheduler_gpu_seconds=2700, maximum_scheduler_gpu_seconds=2700,
        qualification_capacity_reused=False, resources=resources, owner_release=release,
        authority=authority, prior_charge_and_physical_snapshot=physical))
    plan = json.loads((old / "input-plan.json").read_text())
    plan.update(run_id=args.next, output=f"/mowenjie/fluffy-bassoon-2/outputs/{args.next}",
        owner_release=release, compute_binding=compute)
    for method, arm in audit["arms"].items():
        plan["methods"][method] = dict(manifest=arm["next_manifest"], checkpoint=arm["checkpoint"],
            frozen_motor=dict(path=f"/mowenjie/fluffy-bassoon-2/outputs/{args.previous}/{method}/frozen-motor.json",
                sha256=sha(old / "receipts" / method / "frozen-motor.json")), output=plan["output"]+"/"+method)
    input_ref = save("input-plan.json", plan)
    pin = "c353970a316837b13277ec65587e58e5cb25ce56"
    source = subprocess.check_output(["git", "show", pin+":scripts/diagnostics/robotwin_trained_pair.py"])
    with tempfile.TemporaryDirectory(prefix="robotwin-static-plan-") as directory:
        compiler = Path(directory) / "compile.py"
        compiler.write_bytes(source)
        subprocess.run(["python3", str(compiler), "--launcher-commit", pin, "--plan", str(new/"input-plan.json"),
            "--plan-sha256", input_ref["sha256"], "--compile-plan", str(new/"compiled-plan.json")], check=True)
    compiled_sha = sha(new/"compiled-plan.json")
    command = ["bash", "/mowenjie/fluffy-bassoon-2/worktrees/robotwin-trained-pair-launcher-20261031-r1/scripts/diagnostics/robotwin_trained_pair.sh",
        pin, args.next, str(remote/"compiled-plan.json"), compiled_sha]
    (new/"command.txt").write_text(shlex.join(command)+"\n")
    save("reservation.json", dict(run_id=args.next, status="reserved_not_submitted", **resources,
        maximum_source_actions=21600, maximum_scheduler_gpu_seconds=2700, source_goal_rounds=100,
        starting_rounds={m:a["completed_iterations"] for m,a in audit["arms"].items()},
        runtime_commit=plan["runtime_commit"], launcher_commit=pin, compiled_plan_sha256=compiled_sha,
        scope="Original source goal continuation; no additional pilot or tests."))


if __name__ == "__main__":
    main()
