#!/usr/bin/env python3
"""Compose bounded one-arm ledgers into one immutable RoboTwin campaign.

This stdlib-only preparation utility does not load a model, simulator or CUDA.
Workers retain separate exclusive ledger locks. Their fixed caps, together
with the prior campaign's fixed cap, cannot exceed the parent reservation.
The prior journal is charged once in the combined report, never imported into
each child ledger. A later prior-cost audit is a separate receipt; it must not
rewrite the manifest after workers have started.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re


PROJECT_ROOT = Path("/mowenjie/fluffy-bassoon-2")
METHODS = ("ppo", "sac", "ppo_reset")
SOURCE_TASK_LIMITS = {
    "handover_mic": 600,
    "pick_diverse_bottles": 400,
    "place_a2b_left": 400,
    "place_object_stand": 400,
    "place_phone_stand": 400,
    "scan_object": 500,
}
TARGET_TASK_LIMITS = {"adjust_bottle": 400, "move_stapler_pad": 400}
ITERATIONS = 100
TRIALS_PER_ITERATION = 6
TRAIN_EPISODES = 4
EVAL_EPISODES = 8
EVAL_ITERATIONS = (10, 25, 50, 100)
SOURCE_SAFETY_ACTIONS = 1_100_000
SEED = 20261030
PRIOR_CAMPAIGN_ID = "robotwin-one-million-20261029-r1"
PRIOR_CAMPAIGN_CAP = 1_000_000
CALIBRATION_ROOT = PROJECT_ROOT / "outputs/robotwin-staged-20261030-r2/optimizer-calibration"
CALIBRATION_RESULT_SHA256 = "821995144fe49d5503e07ac05b1932ba729ce19d64e101740e06b0fa11ac3f61"


def validate_identity(value: str) -> str:
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]{2,127}", value):
        raise ValueError("Require a short lowercase campaign identity without path separators")
    return value


def build_manifest(campaign_id: str, source_commit: str, horizon_config_sha256: str):
    validate_identity(campaign_id)
    if not re.fullmatch(r"[0-9a-f]{40}", source_commit):
        raise ValueError("Require an exact 40-hex source commit")
    if not re.fullmatch(r"[0-9a-f]{64}", horizon_config_sha256):
        raise ValueError("Require the pinned native horizon file SHA256")
    root = PROJECT_ROOT / "campaigns" / campaign_id
    source_maximum = ITERATIONS * TRAIN_EPISODES * sum(SOURCE_TASK_LIMITS.values())
    if not source_maximum <= SOURCE_SAFETY_ACTIONS:
        raise ValueError("Source safety reservation cannot fit all complete benchmark trials")
    stages = len(EVAL_ITERATIONS)
    source_validation = stages * TRAIN_EPISODES * sum(SOURCE_TASK_LIMITS.values())
    target_per_condition = stages * EVAL_EPISODES * sum(TARGET_TASK_LIMITS.values())
    native_reference = stages * 2 * sum(TARGET_TASK_LIMITS.values())
    conditions = {"ppo": 3, "sac": 2, "ppo_reset": 1}
    workers = {}
    for method in METHODS:
        native = native_reference if method == "ppo" else 0
        cap = SOURCE_SAFETY_ACTIONS + source_validation + target_per_condition * conditions[method] + native
        workers[method] = dict(
            method=method,
            campaign_id=f"{campaign_id}-{method}",
            ledger_root=str(root / "worker-ledgers" / method),
            source_safety_actions=SOURCE_SAFETY_ACTIONS,
            maximum_complete_source_actions=source_maximum,
            source_validation_actions=source_validation,
            target_conditions=conditions[method],
            target_actions=target_per_condition * conditions[method],
            native_reference_actions=native,
            maximum_actions=cap,
            native_episodes=1 if method == "ppo" else 0,
            resource=dict(gpus=1, cpus=16, memory_gib=128, shared_memory_gib=8,
                          operational_driver_seconds=43_200, scheduler_cap_minutes=780),
        )
    child_total = sum(worker["maximum_actions"] for worker in workers.values())
    return dict(
        format="robotwin_parallel_whole_trial_campaign_v1",
        campaign_id=campaign_id,
        source_commit=source_commit,
        root=str(root),
        source_task_episode_limits=SOURCE_TASK_LIMITS,
        target_task_episode_limits=TARGET_TASK_LIMITS,
        benchmark_horizon_config_sha256=horizon_config_sha256,
        source_iterations=ITERATIONS,
        seed=SEED,
        trials_per_iteration=TRIALS_PER_ITERATION,
        train_episodes=TRAIN_EPISODES,
        eval_episodes=EVAL_EPISODES,
        source_eval_iterations=list(EVAL_ITERATIONS),
        completion="100 rounds of six complete K4 trials per learner; actual actions vary with success",
        source_action_budget="physical safety bound only; never cut an episode or trial to meet it",
        ppo=dict(epochs=5, minibatch_trials=16, learning_rate=1e-6, max_update_kl=.1),
        sac=dict(warmup_chunks=500, batch_size=256, utd=20, learning_rate=3e-4),
        frozen_motor=dict(noise_shape=[32, 32], action_limit=1.5, feature_dim=2062),
        prior_campaign=dict(
            id=PRIOR_CAMPAIGN_ID,
            ledger_root=str(PROJECT_ROOT / "campaigns" / PRIOR_CAMPAIGN_ID),
            maximum_actions=PRIOR_CAMPAIGN_CAP,
            status="prior active campaign held at its immutable cap until final audit",
            accounting="charge the prior journal once; its 13026 earlier actions are already included; no child imports",
            continuation="retain current allocation through controlled stop; no automatic old-recipe restart",
        ),
        calibration=dict(root=str(CALIBRATION_ROOT), result_sha256=CALIBRATION_RESULT_SHA256,
                         learning_rate=1e-6, transfer="qualified scalar learning rate only; all new learners initialize fresh"),
        workers=workers,
        maximum_new_actions=child_total,
        maximum_combined_actions=PRIOR_CAMPAIGN_CAP + child_total,
        unused_capacity="release only in a dated reconciliation; never silently spend it on extra data",
    )


def validate_manifest(manifest, *, source_commit=None):
    expected = build_manifest(manifest["campaign_id"], manifest["source_commit"],
                              manifest["benchmark_horizon_config_sha256"])
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(f"Parallel campaign manifest differs from its fixed recipe: {key}")
    if set(manifest) != set(expected) | {"created_at_utc"}:
        raise ValueError("Unexpected or missing parallel campaign manifest fields")
    if source_commit is not None and manifest["source_commit"] != source_commit:
        raise ValueError("Worker source commit differs from its parent manifest")
    return manifest


def read_manifest(path, *, source_commit=None):
    return validate_manifest(json.loads(Path(path).read_text()), source_commit=source_commit)


def write_exclusive(path, value):
    path = Path(path)
    with path.open("x") as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--campaign-id", required=True)
    prepare.add_argument("--source-commit", required=True)
    prepare.add_argument("--horizon-config-sha256", required=True)
    prepare.add_argument("--output", type=Path, required=True)
    admit = subparsers.add_parser("admit-worker")
    admit.add_argument("--manifest", type=Path, required=True)
    admit.add_argument("--source-commit", required=True)
    admit.add_argument("--method", choices=METHODS, required=True)
    admit.add_argument("--run-id", required=True)
    admit.add_argument("--robotwin-root", type=Path, required=True)
    admit.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "prepare":
        manifest = build_manifest(args.campaign_id, args.source_commit, args.horizon_config_sha256)
        manifest["created_at_utc"] = datetime.now(timezone.utc).isoformat()
        validate_manifest(manifest)
        write_exclusive(args.output, manifest)
        print(json.dumps(dict(manifest=str(args.output), maximum_new_actions=manifest["maximum_new_actions"],
                              maximum_combined_actions=manifest["maximum_combined_actions"])))
        return
    validate_identity(args.run_id)
    manifest = read_manifest(args.manifest, source_commit=args.source_commit)
    horizon_path = args.robotwin_root / "task_config/_eval_step_limit.yml"
    horizon_sha = hashlib.sha256(horizon_path.read_bytes()).hexdigest()
    if horizon_sha != manifest["benchmark_horizon_config_sha256"]:
        raise ValueError("Native benchmark horizon file differs from the frozen parent receipt")
    worker = manifest["workers"][args.method]
    receipt = dict(
        parent_campaign_id=manifest["campaign_id"],
        parent_manifest=str(args.manifest.resolve()),
        parent_manifest_sha256=hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        maximum_combined_actions=manifest["maximum_combined_actions"],
        prior_charged_once=True,
        source_commit=args.source_commit,
        run_id=args.run_id,
        benchmark_horizon_config_sha256=horizon_sha,
        worker=worker,
        source_iterations=manifest["source_iterations"],
        trials_per_iteration=manifest["trials_per_iteration"],
        source_eval_iterations=manifest["source_eval_iterations"],
    )
    write_exclusive(args.output, receipt)
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
