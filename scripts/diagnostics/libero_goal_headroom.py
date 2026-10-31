"""LIBERO-Goal instruction-steering headroom check for a frozen RLinf pi0.5.

Modes:
  sim-check   CPU only: task order, scene/init-state interchangeability, goal checker.
  model-check Load the checkpoint, verify fingerprints, shapes, noise determinism and
              prompt sensitivity on one real observation (CPU or allocated GPU).
  run         Allocated GPU: explicit / vague / swap episodes with all-goal tracing.
  summarize   Recompute summary.json from a saved episodes.jsonl.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import socket
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from fluffy_bassoon.rl import libero_goal as lg  # noqa: E402


def now():
    return datetime.now(timezone.utc).isoformat()


def sim_check(args):
    import robosuite
    import mujoco
    from libero.libero import get_libero_path
    suite, tasks = lg.suite_tasks()
    goal_states = [t["goal_state"] for t in tasks]
    envs = [lg.make_env(t["bddl"], render=False) for t in tasks]
    report = dict(time=now(), host=socket.gethostname(), mujoco=mujoco.__version__,
                  robosuite=robosuite.__version__, libero_root=get_libero_path("benchmark_root"),
                  tasks=[{k: t[k] for k in ("index", "name", "language", "bddl_sha256", "goal_state")} for t in tasks])
    # LIBERO re-samples fixture placement on every hard reset; seeding first must give one
    # identical compiled scene across all ten task BDDLs, and different seeds must differ.
    xml_by_seed, sizes = {}, []
    for seed in (1000, 1101, 1000):
        hashes = []
        for env in envs:
            env.seed(seed)
            env.reset()
            hashes.append(lg.hashlib.sha256(env.sim.model.get_xml().encode()).hexdigest())
            sizes.append(int(env.sim.get_state().flatten().size))
        xml_by_seed.setdefault(str(seed), []).append(hashes)
    xml_ok = (all(len({h for run in runs for h in run}) == 1 for runs in xml_by_seed.values())
              and xml_by_seed["1000"][0][0] != xml_by_seed["1101"][0][0])
    xml_hashes = [h for runs in xml_by_seed.values() for run in runs for h in run]
    report["model_xml_sha256_by_seed"] = xml_by_seed
    report["state_sizes"] = sorted(set(sizes))
    init_sets = [np.asarray(suite.get_task_init_states(i)) for i in range(len(tasks))]
    report["init_state_counts"] = [len(s) for s in init_sets]
    report["init_state_dims"] = [list(s.shape[1:]) for s in init_sets]

    # Cross-load every init state into every task's environment; the sim state must equal
    # the stored state exactly and no goal may hold at the initial state. LIBERO's own
    # success (_check_success of env j) must equal the checker's goal j.
    cross = dict(states_loaded=0, exact_state_mismatch=0, goal_true_at_init=[0] * len(tasks),
                 native_parity_mismatch=0)
    object_xy = {}
    for j, env in enumerate(envs):
        for i, states in enumerate(init_sets):
            for state in states:
                env.set_init_state(state)
                cross["states_loaded"] += 1
                if not np.array_equal(env.sim.get_state().flatten(), state):
                    cross["exact_state_mismatch"] += 1
                goals = lg.goal_vector(env, goal_states)
                for g, value in enumerate(goals):
                    cross["goal_true_at_init"][g] += int(value)
                cross["native_parity_mismatch"] += int(bool(env.env._check_success()) != goals[j])
                if j == 0:
                    for name, obj in env.env.objects_dict.items():
                        pos = env.sim.data.body_xpos[env.sim.model.body_name2id(obj.root_body)][:2]
                        object_xy.setdefault(name, {}).setdefault(i, []).append(pos.tolist())
    report["cross_load"] = cross
    report["object_xy_by_source_task"] = {
        name: {str(i): dict(mean=np.mean(v, 0).round(4).tolist(), std=np.std(v, 0).round(4).tolist())
               for i, v in by_task.items()} for name, by_task in object_xy.items()}

    # Settle physics from two init states of every source task with the gripper open.
    settle = []
    env = envs[0]
    for i, states in enumerate(init_sets):
        for trial in range(2):
            env.seed(lg.env_seed(dict(source_task=i, trial=trial)))
            env.reset()
            env.set_init_state(states[trial])
            action = np.zeros(7)
            action[-1] = -1
            for _ in range(lg.SETTLE):
                env.step(action)
            settle.append(dict(source_task=i, trial=trial, goals=lg.goal_vector(env, goal_states)))
    report["settle_goals_true"] = sum(any(s["goals"]) for s in settle)
    report["settle_samples"] = len(settle)
    ok = (xml_ok and len(set(sizes)) == 1 and cross["exact_state_mismatch"] == 0
          and not any(cross["goal_true_at_init"]) and cross["native_parity_mismatch"] == 0
          and report["settle_goals_true"] == 0)
    report["passed"] = bool(ok)
    for env in envs:
        env.close()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    lg.write_json(out / "sim-check.json", report)
    print(json.dumps(dict(passed=ok, xml_ok=xml_ok, xml_unique=len(set(xml_hashes)), sizes=sorted(set(sizes)),
                          cross=cross, settle_goals_true=report["settle_goals_true"]), indent=1))
    if not ok:
        raise SystemExit("sim-check failed")


def load_model(args, device):
    import torch
    from rlinf.models import get_model
    weights = Path(args.model_path) / "model.safetensors"
    digest = lg.sha256_file(weights)
    if digest != args.expected_model_sha256:
        raise RuntimeError(f"Checkpoint SHA256 {digest} differs from the expected value")
    torch.manual_seed(0)
    model = get_model(lg.model_config(args.model_path, args.norm_stats))
    model = model.to(device).eval().requires_grad_(False)
    return model, digest


def gpu_receipt(output, pids):
    """Record where the renderer workers and this driver hold GPU contexts."""
    try:
        text = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=30).stdout
        apps = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,gpu_uuid,used_memory", "--format=csv,noheader"],
                              capture_output=True, text=True, timeout=30).stdout
        uuids = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid,pci.bus_id", "--format=csv,noheader"],
                               capture_output=True, text=True, timeout=30).stdout
    except (OSError, subprocess.TimeoutExpired) as error:
        text, apps, uuids = repr(error), "", ""
    import re
    import torch
    allocated = "GPU-" + str(torch.cuda.get_device_properties(0).uuid)
    index_of = {row.split(",")[1].strip(): row.split(",")[0].strip()
                for row in uuids.strip().splitlines() if row.count(",") >= 2}
    allowed = index_of.get(allocated)
    ours = set(pids + [os.getpid()])
    processes = [dict(gpu=m.group(1), pid=int(m.group(2)), type=m.group(3)) for m in
                 re.finditer(r"^\|\s+(\d+)\s+\S+\s+\S+\s+(\d+)\s+(C\+G|C|G)\s", text, re.M)]
    listed = [p for p in processes if p["pid"] in ours]
    misplaced = [p for p in listed if p["gpu"] != allowed]
    record = dict(time=now(), worker_pids=pids, driver_pid=os.getpid(), allocated_uuid=allocated,
                  allocated_nvidia_smi_index=allowed, our_gpu_processes=listed, misplaced=misplaced,
                  workers_listed=sorted({p["pid"] for p in listed} & set(pids)),
                  cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
                  mujoco_egl_device_id=os.environ.get("MUJOCO_EGL_DEVICE_ID"),
                  visible_gpus=uuids.strip().splitlines(), compute_apps=apps.strip().splitlines(), nvidia_smi=text)
    lg.write_json(output / "gpu-binding.json", record)
    if misplaced or allowed is None:
        raise RuntimeError(f"GPU binding check failed: {misplaced or 'allocated GPU not listed'}")
    return record


def model_check(args):
    import torch
    device = "cuda" if args.device == "cuda" else "cpu"
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    suite, tasks = lg.suite_tasks()
    model, digest = load_model(args, device)
    fingerprint_before = lg.model_fingerprint(model)
    rows = [dict(source_task=1, trial=0), dict(source_task=1, trial=0)]
    if args.synthetic_obs:  # CPU load/transform check without a renderer
        rng = np.random.default_rng(0)
        fake = dict(obs=dict(full_image=rng.integers(0, 255, (lg.CAMERA, lg.CAMERA, 3), dtype=np.uint8),
                             wrist_image=rng.integers(0, 255, (lg.CAMERA, lg.CAMERA, 3), dtype=np.uint8),
                             state=np.array([-.05, 0, 1.1, 3.1, 0, 0, .04, -.04])),
                    goals=[False] * len(tasks), settled_state_sha256="synthetic")
        pool = None
    else:
        pool = lg.EnvPool(2, [t["goal_state"] for t in tasks])
    try:
        if pool is None:
            states = [fake, fake]
        else:
            states = pool.reset([dict(bddl=tasks[1]["bddl"], task=1, seed=lg.env_seed(rows[0]),
                                      init_state=suite.get_task_init_states(1)[0])] * 2)
        if device == "cuda" and pool is not None:
            gpu_receipt(out, pool.pids)
        langs = [t["language"] for t in tasks]
        noise = lg.chunk_noise(rows, 0, args.base_seed, device)
        same, _ = model.predict_action_batch(lg.model_obs(states, [langs[1], langs[1]]), mode="eval", noise=noise)
        other, _ = model.predict_action_batch(lg.model_obs(states, [langs[1], langs[7]]), mode="eval", noise=noise)
        repeat, _ = model.predict_action_batch(lg.model_obs(states, [langs[1], langs[7]]), mode="eval", noise=noise)
        same, other, repeat = (x.float().cpu() for x in (same, other, repeat))
        report = dict(time=now(), device=device, checkpoint_sha256=digest, fingerprint=fingerprint_before,
                      action_shape=list(same.shape), finite=bool(torch.isfinite(same).all()),
                      paired_rows_identical=bool(torch.equal(same[0], same[1])),
                      repeat_identical=bool(torch.equal(other, repeat)),
                      prompt_change_max_abs_diff=float((other[1] - same[1]).abs().max()),
                      first_actions=same[0, :2].tolist(), initial_goals=states[0]["goals"],
                      paired_start_identical=states[0]["settled_state_sha256"] == states[1]["settled_state_sha256"],
                      fingerprint_after=lg.model_fingerprint(model))
    finally:
        if pool is not None:
            pool.close()
    report["synthetic_obs"] = bool(args.synthetic_obs)
    report["passed"] = bool(report["action_shape"] == [2, lg.CHUNK, 7] and report["finite"]
                            and report["paired_rows_identical"] and report["prompt_change_max_abs_diff"] > 0
                            and report["paired_start_identical"]
                            and report["fingerprint_after"] == fingerprint_before)
    lg.write_json(out / ("model-check-synthetic.json" if args.synthetic_obs else "model-check.json"), report)
    print(json.dumps({k: v for k, v in report.items() if k != "first_actions"}, indent=1))
    if not report["passed"]:
        raise SystemExit("model-check failed")


def run(args):
    import torch
    from omegaconf import OmegaConf
    start = time.monotonic()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=False)
    suite, tasks = lg.suite_tasks()
    plan = lg.build_plan(len(tasks), args.explicit_trials, args.paired_trials)
    if args.limit:
        plan = plan[:args.limit]
    config = dict(time=now(), argv=sys.argv, source_commit=args.source_commit, host=socket.gethostname(),
                  slurm_job_id=os.environ.get("SLURM_JOB_ID"), slurm_step_id=os.environ.get("SLURM_STEP_ID"),
                  python=platform.python_version(), torch=torch.__version__, model_config=OmegaConf.to_container(lg.model_config(
                      args.model_path, args.norm_stats)), norm_stats_sha256=lg.sha256_file(args.norm_stats),
                  tasks=[{k: t[k] for k in ("index", "name", "language", "bddl_sha256", "goal_state")} for t in tasks],
                  vague_prompts=lg.VAGUE_PROMPTS, steps=lg.STEPS, chunk=lg.CHUNK, settle=lg.SETTLE,
                  base_seed=args.base_seed, batch_size=args.batch_size, plan_size=len(plan),
                  plan_counts={c: sum(r["condition"] == c for r in plan) for c in ("explicit", "vague", "swap")})
    lg.write_json(out / "config.json", config)
    lg.write_json(out / "plan.json", plan)
    device = args.device
    if args.smoke_chunks and device == "cuda":
        raise SystemExit("--smoke-chunks is a CPU pipeline smoke test only")
    model, digest = load_model(args, device)
    fingerprint = lg.model_fingerprint(model)
    pool = lg.EnvPool(args.batch_size, [t["goal_state"] for t in tasks])
    records, batches = [], []
    try:
        for begin in range(0, len(plan), args.batch_size):
            elapsed = time.monotonic() - start
            longest = max((b["seconds"] for b in batches), default=0)
            if elapsed + 1.2 * longest > args.max_elapsed_seconds:
                print(f"Stopping before batch {begin // args.batch_size}: elapsed {elapsed:.0f}s", flush=True)
                break
            rows = plan[begin:begin + args.batch_size]
            tic = time.monotonic()
            result = lg.run_batch(model, pool, rows, tasks, suite, base_seed=args.base_seed, device=device,
                                  chunks=args.smoke_chunks or lg.STEPS // lg.CHUNK)
            if begin == 0 and device == "cuda":
                gpu_receipt(out, pool.pids)
            with (out / "episodes.jsonl").open("a") as stream:
                for rec in result:
                    stream.write(json.dumps(rec) + "\n")
            records += result
            free, total = torch.cuda.mem_get_info() if device == "cuda" else (-2**20, -2**20)
            batches.append(dict(index=begin // args.batch_size, episodes=len(rows),
                                seconds=time.monotonic() - tic, finished=now(),
                                gpu_free_mib=free // 2**20, gpu_total_mib=total // 2**20,
                                torch_max_allocated_mib=(torch.cuda.max_memory_allocated() // 2**20
                                                         if device == "cuda" else -1)))
            done = sum(r["instructed_success"] for r in result if r["instructed_success"] is not None)
            print(f"batch {batches[-1]['index']} {len(rows)} episodes {batches[-1]['seconds']:.0f}s "
                  f"instructed successes {done} gpu free {batches[-1]['gpu_free_mib']} MiB", flush=True)
    finally:
        pool.close()
    fingerprint_after = lg.model_fingerprint(model)
    summary = lg.summarize(records)
    summary["run"] = dict(checkpoint_sha256=digest, fingerprint_before=fingerprint,
                          fingerprint_after=fingerprint_after, frozen=fingerprint == fingerprint_after,
                          planned=len(plan), completed=len(records), batches=batches,
                          elapsed_seconds=time.monotonic() - start, finished=now())
    lg.write_json(out / "summary.json", summary)
    print(json.dumps(dict(completed=len(records), planned=len(plan), criteria=summary["criteria"],
                          checks=summary["checks"]), indent=1), flush=True)
    if fingerprint != fingerprint_after:
        raise SystemExit("Model weights changed during evaluation")


def summarize(args):
    records = [json.loads(line) for line in Path(args.episodes).read_text().splitlines() if line.strip()]
    summary = lg.summarize(records)
    lg.write_json(args.output, summary)
    print(json.dumps(summary["criteria"], indent=1))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("sim-check", "model-check", "run", "summarize"))
    parser.add_argument("--output-dir")
    parser.add_argument("--model-path")
    parser.add_argument("--norm-stats")
    parser.add_argument("--expected-model-sha256")
    parser.add_argument("--source-commit")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--base-seed", type=int, default=2026)
    parser.add_argument("--batch-size", type=int, default=34)
    parser.add_argument("--explicit-trials", type=int, default=12)
    parser.add_argument("--paired-trials", type=int, default=2)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--synthetic-obs", action="store_true", help="model-check without a renderer")
    parser.add_argument("--smoke-chunks", type=int, default=0,
                        help="CPU pipeline smoke only: stop episodes after this many chunks (incomplete)")
    parser.add_argument("--max-elapsed-seconds", type=float, default=2700)
    parser.add_argument("--episodes")
    parser.add_argument("--output")
    args = parser.parse_args()
    if args.mode in ("run",) or (args.mode == "model-check" and args.device == "cuda"):
        from fluffy_bassoon.gpu_execution import validate_execution
        validate_execution("slurm", os.environ)
    {"sim-check": sim_check, "model-check": model_check, "run": run, "summarize": summarize}[args.mode](args)


if __name__ == "__main__":
    main()
