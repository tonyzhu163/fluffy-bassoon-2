"""Success of the frozen pi0.5 motor under fixed initial-noise arms (Stage 0 screen and cache).

outputs/noise-mode-icrl-20261029-r1/PLAN.md: an arm is one fixed initial-noise tensor
(ms.arm_noise) used for every chunk of an 80-step attempt; arm -1 is native sampling
with fresh N(0, I) noise from the seeded batch generator at every chunk. This script
records outcomes for each (physics variant, scene, arm, repeat). Each variant builds
its own env inside ms.object_physics, because RLinf builds object materials once at
construction. Within a variant, rows are shuffled
with a seed so arms are not confounded with batch slots; repeats land in different
slots, which tests determinism. Each batch saves its issued action chunks for the
mode-effect check. Read-only: no training. Rows are appended per batch; a failure or
the time budget keeps finished rows.
"""
import argparse
from collections import defaultdict
from contextlib import ExitStack
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np


def object_list(text):
    objects = [int(x) for x in text.split(",")]
    if not objects or len(set(objects)) != len(objects) or any(not 0 <= o < 25 for o in objects):
        raise argparse.ArgumentTypeError("Require distinct object indices in 0-24")
    return objects


def variant_list(text):
    """Comma-separated friction:density scales, for example 1:1,0.3:1,1:4."""
    try:
        variants = [tuple(float(x) for x in item.split(":")) for item in text.split(",")]
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error
    if (not variants or any(len(v) != 2 or not all(np.isfinite(v)) or min(v) <= 0 for v in variants)
            or len(set(variants)) != len(variants)):
        raise argparse.ArgumentTypeError("Require distinct positive friction:density pairs")
    return variants


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rlinf-root", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--norm-stats", type=Path, required=True)
    parser.add_argument("--expected-model-sha256", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--objects", type=object_list, required=True)
    parser.add_argument("--scenes-per-object", type=int, default=4)
    parser.add_argument("--variants", type=variant_list, default=variant_list("1:1"))
    parser.add_argument("--arms", type=int, default=16)
    parser.add_argument("--arm-seed", type=int, default=3000320000, help="arm k uses seed arm_seed + k")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--num-envs", type=int, default=48)
    parser.add_argument("--seed", type=int, default=3000330000)
    parser.add_argument("--time-budget-seconds", type=float, default=7200)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root/"src"))
    from fluffy_bassoon.rl import maniskill_pi05 as ms
    if sys.platform != "linux":
        raise RuntimeError("Run model/simulator collection on allocated Linux only")
    mode = ms.execution_mode()
    arms = [-1] + list(range(args.arms))
    per_variant = len(args.objects)*args.scenes_per_object*len(arms)*args.repeats
    if (int(os.environ.get("SLURM_STEP_NUM_TASKS", "1")) != 1 or not 1 <= args.scenes_per_object <= 1000
            or not 1 <= args.arms <= 256 or not 1 <= args.repeats <= 20 or not 1 <= args.num_envs <= 256
            or per_variant % args.num_envs or not 0 <= args.seed < 2**32
            or not 0 <= args.arm_seed < 2**62 or not args.time_budget_seconds > 0):
        raise ValueError(f"Require one srun process, valid counts, {args.num_envs} envs dividing "
                         f"{per_variant} rows per variant, valid seeds and budget")
    source = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    if source != args.source_commit or subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True).strip():
        raise ValueError("Require exact clean reviewed source")
    if ms.sha256_file(args.model_path/"model.safetensors") != args.expected_model_sha256:
        raise ValueError("pi0.5 checkpoint SHA256 differs")
    if args.norm_stats.name != "norm_stats.json" or not args.norm_stats.is_file():
        raise FileNotFoundError("Pass the checkpoint's own norm_stats.json; RLinf would silently skip normalization")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir/"trajectories").mkdir()
    started = time.monotonic()
    ms.prepare_gpu(root, args.output_dir)
    sys.path.insert(0, str(args.rlinf_root))
    import torch
    from rlinf.envs.sim.maniskill.maniskill_env import ManiskillEnv
    import rlinf.envs.sim.maniskill.tasks  # noqa: F401 -- registers PutOnPlateInScene25Main-v3
    from rlinf.envs.sim.maniskill.tasks.put_on_in_scene_multi import PutOnPlateInScene25
    from rlinf.models import get_model
    if torch.cuda.device_count() != 1:
        raise ValueError("Require one allocated physical GPU")
    torch.manual_seed(args.seed)
    arm_seeds = [args.arm_seed + k for k in range(args.arms)]
    noise = ms.arm_noise(arm_seeds)
    noise_sha256 = [ms.tensor_sha256(n) for n in noise]
    env = model = frozen_before = physics = None
    rows, stopped = [], None
    try:
        for variant_index, (friction, density) in enumerate(args.variants):
            if time.monotonic() - started > args.time_budget_seconds:
                stopped = "time_budget"
                break
            # The patch stays active for the env's lifetime, so any scene rebuild keeps the variant.
            physics = ExitStack()
            physics.enter_context(ms.object_physics(PutOnPlateInScene25, friction, density))
            env = ManiskillEnv(ms.env_config(args.rlinf_root, args.num_envs, args.seed), args.num_envs,
                               seed_offset=0, total_num_processes=1, worker_info=None, record_metrics=True)
            if model is None:
                ms.record_renderer_binding(env, args.output_dir)
                model = get_model(ms.model_config(args.model_path, args.norm_stats)).cuda().eval()
                model.requires_grad_(False)
                frozen_before = ms.model_fingerprint(model)
            base = env.env.unwrapped
            object_names = list(base.carrot_names)
            rng = np.random.default_rng([args.seed, variant_index])
            # Same scenes for every variant: the scene rng ignores the variant.
            scenes = ms.sample_scenes(base.basic_obj_infos, args.objects, args.scenes_per_object,
                                      np.random.default_rng(args.seed))
            plan = [dict(scene, scene_index=index, variant=variant_index, friction_scale=friction,
                         density_scale=density, arm=arm, arm_seed=None if arm < 0 else arm_seeds[arm],
                         noise_sha256=None if arm < 0 else noise_sha256[arm], repeat=repeat)
                    for index, scene in enumerate(scenes) for arm in arms for repeat in range(args.repeats)]
            plan = [plan[i] for i in rng.permutation(len(plan))]
            if variant_index == 0:
                contract = dict(source_commit=source, rlinf_root=str(args.rlinf_root),
                    rlinf_checkpoint_loader_sha256=ms.sha256_file(args.rlinf_root/"rlinf/models/embodiment/openpi/checkpoint.py"),
                    rlinf_task_sha256=ms.sha256_file(args.rlinf_root/"rlinf/envs/sim/maniskill/tasks/put_on_in_scene_multi.py"),
                    model_path=str(args.model_path), model_sha256=args.expected_model_sha256,
                    norm_stats=str(args.norm_stats), norm_stats_sha256=ms.sha256_file(args.norm_stats),
                    task="PutOnPlateInScene25Main-v3", objects=[object_names[o] for o in args.objects],
                    object_indices=args.objects, scenes_per_object=args.scenes_per_object,
                    overlays="training range 0-15", variants=[dict(friction_scale=f, density_scale=d)
                                                              for f, d in args.variants],
                    arms=args.arms, arm_seeds=arm_seeds, arm_noise_sha256=noise_sha256,
                    noise_shape=list(ms.NOISE_SHAPE), native_arm=-1, repeats=args.repeats,
                    num_envs=args.num_envs, seed=args.seed, rows=per_variant*len(args.variants),
                    time_budget_seconds=args.time_budget_seconds, max_episode_steps=ms.STEPS,
                    action_chunk=ms.CHUNK, denoise_steps=4, control_mode=env.cfg.init_params.control_mode,
                    perturbation="none; arm k passes arm_noise[k] as the initial noise of every chunk, "
                                 "arm -1 draws fresh N(0, I) noise from the batch generator every chunk",
                    physics="source-object friction and density scaled at env construction "
                            "(ms.object_physics); plates and arena unchanged",
                    sampler="RLinf eval-mode Euler ODE; the initial noise is its only stochastic input",
                    outcome_fields=ms.OUTCOME_FIELDS,
                    runtime={name: version(name) for name in ("torch", "mani-skill", "sapien", "rlinf-openpi", "jax")},
                    job_id=ms.job_id(), execution=mode, frozen_model_sha256=frozen_before,
                    role="frozen pi0.5 noise-arm headroom screen/outcome cache; no training")
                (args.output_dir/"contract.json").write_text(json.dumps(contract, indent=2)+"\n")
                (args.output_dir/"scenes.json").write_text(json.dumps(scenes, indent=2)+"\n")
            generator = torch.Generator(device="cuda")
            for start in range(0, len(plan), args.num_envs):
                if time.monotonic() - started > args.time_budget_seconds:
                    stopped = "time_budget"
                    break
                batch = plan[start:start+args.num_envs]
                batch_seed = args.seed + 1_000_000*variant_index + start
                batch_noise = torch.stack([noise[max(row["arm"], 0)] for row in batch])
                issued = []
                outcomes = ms.run_batch(model, env, batch, seed=batch_seed, generator=generator,
                                        initial_noise=batch_noise,
                                        fresh_noise=torch.tensor([row["arm"] < 0 for row in batch]),
                                        issued=issued)
                trajectories = torch.stack(issued, 1).numpy()
                np.savez_compressed(args.output_dir/"trajectories"/f"variant{variant_index}-batch{start:06d}.npz",
                                    issued=trajectories, row=np.arange(len(rows), len(rows)+len(batch)))
                with (args.output_dir/"outcomes.jsonl").open("a") as stream:
                    for i, row in enumerate(batch):
                        record = dict(row, row=len(rows), object_name=object_names[row["object"]],
                                      batch_seed=batch_seed, slot=i,
                                      issued_actions_sha256=hashlib.sha256(trajectories[i].tobytes()).hexdigest(),
                                      **outcomes[i])
                        rows.append(record)
                        stream.write(json.dumps(record)+"\n")
                print(json.dumps(dict(variant=variant_index, completed=len(rows),
                                      total=per_variant*len(args.variants),
                                      elapsed=time.monotonic()-started)), flush=True)
            env.env.close()
            env = None
            physics.close()
            physics = None
            if stopped:
                break
        frozen_after = ms.model_fingerprint(model) if model is not None else None
        if frozen_after != frozen_before:
            raise RuntimeError("Frozen motor weights changed during evaluation")
        by_variant_arm = defaultdict(list)
        for row in rows:
            by_variant_arm[row["variant"], row["arm"]].append(row)
        summary = dict(completed=stopped is None and len(rows) == per_variant*len(args.variants),
                       stopped=stopped, rows=len(rows), source_commit=source,
                       frozen_model_before=frozen_before, frozen_model_after=frozen_after,
                       native_actions=len(rows)*ms.STEPS, queries=len(rows)*(ms.STEPS//ms.CHUNK),
                       contract_sha256=hashlib.sha256((args.output_dir/"contract.json").read_bytes()).hexdigest(),
                       by_variant_arm=[dict(variant=v, arm=a, episodes=len(r), success_once=ms.rates(r),
                                            grasped_once=ms.rates(r, "grasped_once"))
                                       for (v, a), r in sorted(by_variant_arm.items())],
                       elapsed_seconds=time.monotonic()-started,
                       peak_cuda_memory_gib=torch.cuda.max_memory_allocated()/2**30, training_launched=False)
        (args.output_dir/"summary.json").write_text(json.dumps(summary, indent=2)+"\n")
        print(json.dumps(dict(completed=summary["completed"], stopped=stopped, rows=len(rows))), flush=True)
    except BaseException as error:
        (args.output_dir/"failure.json").write_text(json.dumps(dict(error_type=type(error).__name__,
            error=str(error), source_commit=source, rows=len(rows), training_launched=False), indent=2)+"\n")
        raise
    finally:
        if env is not None:
            env.env.close()
        if physics is not None:
            physics.close()


if __name__ == "__main__":
    main()
