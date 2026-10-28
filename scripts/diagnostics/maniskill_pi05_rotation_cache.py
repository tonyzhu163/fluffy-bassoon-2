"""Success of the frozen pi0.5 motor under a net translation rotation (headroom screen and cache).

docs/maniskill-pi05-icrl.md: a hidden per-trial angle theta and the learner's
correction phi both rotate RLinf's prepared world_vector[:2] about the vertical axis,
so outcomes depend only on the net angle theta + phi. This script records success for
each (scene, net angle, repeat) with the frozen motor. A small run is the headroom
screen; a large one is the outcome cache used for CPU meta-training. Each batch mixes
angles and scenes in a seeded shuffle, so no angle is confounded with batch position.
Read-only: no training. Rows are appended per batch; a failure keeps finished rows.
"""
import argparse
from collections import defaultdict
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np


def angle_list(text):
    angles = [float(x) for x in text.split(",")]
    if not angles or len(set(angles)) != len(angles) or any(not -180 <= a <= 180 for a in angles):
        raise argparse.ArgumentTypeError("Require distinct comma-separated angles in [-180,180] degrees")
    return angles


def object_list(text):
    objects = [int(x) for x in text.split(",")]
    if not objects or len(set(objects)) != len(objects) or any(not 0 <= o < 25 for o in objects):
        raise argparse.ArgumentTypeError("Require distinct object indices in 0-24")
    return objects


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rlinf-root", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--norm-stats", type=Path, required=True)
    parser.add_argument("--expected-model-sha256", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--angles", type=angle_list, default=angle_list("0,15,-15,30,-30,45,-45,60,-60,90,-90"))
    parser.add_argument("--objects", type=object_list, default=object_list(",".join(map(str, range(16)))))
    parser.add_argument("--scenes-per-object", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--num-envs", type=int, default=50)
    parser.add_argument("--seed", type=int, default=3000310000)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root/"src"))
    from fluffy_bassoon.rl import maniskill_pi05 as ms
    if sys.platform != "linux":
        raise RuntimeError("Run model/simulator collection on allocated Linux only")
    mode = ms.execution_mode()
    total = len(args.objects)*args.scenes_per_object*len(args.angles)*args.repeats
    if (int(os.environ.get("SLURM_STEP_NUM_TASKS", "1")) != 1 or not 1 <= args.scenes_per_object <= 1000
            or not 1 <= args.repeats <= 20 or not 1 <= args.num_envs <= 256 or total % args.num_envs
            or not 0 <= args.seed < 2**32):
        raise ValueError(f"Require one srun process, valid counts, {args.num_envs} envs dividing {total} rows, valid seed")
    source = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    if source != args.source_commit or subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True).strip():
        raise ValueError("Require exact clean reviewed source")
    if ms.sha256_file(args.model_path/"model.safetensors") != args.expected_model_sha256:
        raise ValueError("pi0.5 checkpoint SHA256 differs")
    if args.norm_stats.name != "norm_stats.json" or not args.norm_stats.is_file():
        raise FileNotFoundError("Pass the checkpoint's own norm_stats.json; RLinf would silently skip normalization")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    ms.prepare_gpu(root, args.output_dir)
    sys.path.insert(0, str(args.rlinf_root))
    import torch
    from rlinf.envs.sim.maniskill.maniskill_env import ManiskillEnv
    import rlinf.envs.sim.maniskill.tasks  # noqa: F401 -- registers PutOnPlateInScene25Main-v3
    from rlinf.models import get_model
    if torch.cuda.device_count() != 1:
        raise ValueError("Require one allocated physical GPU")
    torch.manual_seed(args.seed)
    env = None
    try:
        env = ManiskillEnv(ms.env_config(args.rlinf_root, args.num_envs, args.seed), args.num_envs,
                           seed_offset=0, total_num_processes=1, worker_info=None, record_metrics=True)
        ms.record_renderer_binding(env, args.output_dir)
        model = get_model(ms.model_config(args.model_path, args.norm_stats)).cuda().eval()
        model.requires_grad_(False)
        frozen_before = ms.model_fingerprint(model)
        base = env.env.unwrapped
        rng = np.random.default_rng(args.seed)
        scenes = ms.sample_scenes(base.basic_obj_infos, args.objects, args.scenes_per_object, rng)
        plan = [dict(scene, scene_index=index, net_angle_degrees=angle, repeat=repeat)
                for index, scene in enumerate(scenes) for angle in args.angles for repeat in range(args.repeats)]
        plan = [plan[i] for i in rng.permutation(len(plan))]
        object_names = list(base.carrot_names)
        contract = dict(source_commit=source, rlinf_root=str(args.rlinf_root),
            rlinf_checkpoint_loader_sha256=ms.sha256_file(args.rlinf_root/"rlinf/models/embodiment/openpi/checkpoint.py"),
            model_path=str(args.model_path),
            model_sha256=args.expected_model_sha256, norm_stats=str(args.norm_stats),
            norm_stats_sha256=ms.sha256_file(args.norm_stats), task="PutOnPlateInScene25Main-v3",
            objects=[object_names[o] for o in args.objects], object_indices=args.objects,
            scenes_per_object=args.scenes_per_object, overlays="training range 0-15",
            net_angles_degrees=args.angles, repeats=args.repeats, num_envs=args.num_envs, seed=args.seed,
            rows=len(plan), max_episode_steps=ms.STEPS, action_chunk=ms.CHUNK, denoise_steps=4,
            control_mode=env.cfg.init_params.control_mode,
            perturbation="RLinf prepared world_vector[:2] rotated about vertical by net angle before chunk_step; "
                         "z, rotation, gripper unchanged",
            sampler="RLinf eval-mode Euler ODE, generator seeded per batch",
            outcome_fields=ms.OUTCOME_FIELDS,
            runtime={name: version(name) for name in ("torch", "mani-skill", "sapien", "rlinf-openpi", "jax")},
            job_id=ms.job_id(), execution=mode, frozen_model_sha256=frozen_before,
            role="frozen pi0.5 rotation headroom/outcome cache; no training")
        (args.output_dir/"contract.json").write_text(json.dumps(contract, indent=2)+"\n")
        (args.output_dir/"scenes.json").write_text(json.dumps(scenes, indent=2)+"\n")
        generator = torch.Generator(device="cuda")
        rows = []
        for start in range(0, len(plan), args.num_envs):
            batch = plan[start:start+args.num_envs]
            outcomes = ms.run_batch(model, env, batch, seed=args.seed+start, generator=generator,
                                    angles_degrees=[row["net_angle_degrees"] for row in batch],
                                    audit_path=args.output_dir/'issued-actions.jsonl')
            with (args.output_dir/"outcomes.jsonl").open("a") as stream:
                for row, outcome in zip(batch, outcomes):
                    record = dict(row, object_name=object_names[row["object"]], batch_seed=args.seed+start, **outcome)
                    rows.append(record)
                    stream.write(json.dumps(record)+"\n")
            print(json.dumps(dict(completed=len(rows), total=len(plan), elapsed=time.monotonic()-started)), flush=True)
        by_angle, by_object_angle = defaultdict(list), defaultdict(list)
        for row in rows:
            by_angle[row["net_angle_degrees"]].append(row)
            by_object_angle[(row["object"], row["net_angle_degrees"])].append(row)
        frozen_after = ms.model_fingerprint(model)
        if frozen_after != frozen_before:
            raise RuntimeError("Frozen motor weights changed during evaluation")
        summary = dict(completed=True, source_commit=source,
            frozen_model_before=frozen_before, frozen_model_after=frozen_after,
            native_actions=len(rows)*ms.STEPS, settling_steps=0, queries=len(rows)*(ms.STEPS//ms.CHUNK),
            contract_sha256=hashlib.sha256((args.output_dir/"contract.json").read_bytes()).hexdigest(),
            by_angle=[dict(net_angle_degrees=a, episodes=len(by_angle[a]), success_once=ms.rates(by_angle[a]),
                           success_at_end=ms.rates(by_angle[a], "success_at_end"),
                           grasped_once=ms.rates(by_angle[a], "grasped_once"),
                           object_plate_distance_end_m=ms.rates(by_angle[a], "object_plate_distance_end_m"),
                           object_plate_distance_min_m=ms.rates(by_angle[a], "object_plate_distance_min_m"))
                      for a in sorted(by_angle)],
            by_object=[dict(object=o, name=object_names[o], success_once_by_angle={
                           str(a): ms.rates(by_object_angle[(o, a)]) for a in sorted(args.angles)})
                       for o in args.objects],
            elapsed_seconds=time.monotonic()-started,
            peak_cuda_memory_gib=torch.cuda.max_memory_allocated()/2**30, training_launched=False)
        (args.output_dir/"summary.json").write_text(json.dumps(summary, indent=2)+"\n")
        print(json.dumps(summary["by_angle"]), flush=True)
    except BaseException as error:
        (args.output_dir/"failure.json").write_text(json.dumps(dict(error_type=type(error).__name__,
            error=str(error), source_commit=source, training_launched=False), indent=2)+"\n")
        raise
    finally:
        if env is not None:
            env.env.close()


if __name__ == "__main__":
    main()
