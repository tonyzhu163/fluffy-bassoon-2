"""Per-object competence gate for RLinf's pi0.5 on ManiSkill PutOnPlateInScene25Main-v3.

Qualifies a frozen pi0.5 motor for hidden-perturbation in-context RL: success must
depend on the trial's hidden variable, not on motor failures. Every object (16
training, 9 held-out) gets the same number of episodes; overlays, positions and
orientations are drawn from the training ranges with a fixed seed, so the object is
the only systematic variable. Works for the SFT and RL checkpoints: --model-path is
the directory holding model.safetensors and --norm-stats the checkpoint's own
norm_stats.json. Uses RLinf's environment wrapper, model factory, eval-mode sampler
and action preparation unchanged (source pinned by --rlinf-root). Read-only.
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rlinf-root", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--norm-stats", type=Path, required=True)
    parser.add_argument("--expected-model-sha256", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--episodes-per-object", type=int, default=20)
    parser.add_argument("--num-envs", type=int, default=50)
    parser.add_argument("--seed", type=int, default=3000300000)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root/"src"))
    from fluffy_bassoon.rl import maniskill_pi05 as ms
    if sys.platform != "linux":
        raise RuntimeError("Run model/simulator qualification on allocated Linux only")
    mode = ms.execution_mode()
    total = ms.OBJECTS*args.episodes_per_object
    if (int(os.environ.get("SLURM_STEP_NUM_TASKS", "1")) != 1 or not 1 <= args.episodes_per_object <= 200
            or not 1 <= args.num_envs <= 256 or total % args.num_envs or not 0 <= args.seed < 2**32):
        raise ValueError("Require one srun process, 1-200 episodes per object, batches dividing 25*episodes, valid seed")
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
        model = get_model(ms.model_config(args.model_path, args.norm_stats)).cuda().eval()
        model.requires_grad_(False)
        frozen_before = ms.model_fingerprint(model)
        env = ManiskillEnv(ms.env_config(args.rlinf_root, args.num_envs, args.seed), args.num_envs,
                           seed_offset=0, total_num_processes=1, worker_info=None, record_metrics=True)
        ms.record_renderer_binding(env, args.output_dir)
        base = env.env.unwrapped
        plan = ms.sample_scenes(base.basic_obj_infos, range(ms.OBJECTS), args.episodes_per_object,
                                np.random.default_rng(args.seed))
        object_names = list(base.carrot_names)
        contract = dict(source_commit=source, rlinf_root=str(args.rlinf_root),
            rlinf_checkpoint_loader_sha256=ms.sha256_file(args.rlinf_root/"rlinf/models/embodiment/openpi/checkpoint.py"),
            model_path=str(args.model_path),
            model_sha256=args.expected_model_sha256, norm_stats=str(args.norm_stats),
            norm_stats_sha256=ms.sha256_file(args.norm_stats),
            task="PutOnPlateInScene25Main-v3", obj_set="all",
            objects=object_names, train_objects=object_names[:ms.TRAIN_OBJECTS], overlays="training range 0-15",
            episodes_per_object=args.episodes_per_object, num_envs=args.num_envs, seed=args.seed,
            max_episode_steps=ms.STEPS, action_chunk=ms.CHUNK, denoise_steps=4,
            sampler="RLinf eval-mode Euler ODE, seeded initial noise",
            gelu="RLinf e1a82eb exact GELU; checkpoint trained with tanh-approximate GELU (RLinf issue 1725)",
            outcome_fields=ms.OUTCOME_FIELDS,
            runtime={name: version(name) for name in ("torch", "mani-skill", "sapien", "rlinf-openpi", "jax")},
            job_id=ms.job_id(), execution=mode, frozen_model_sha256=frozen_before,
            role="frozen pi0.5 motor competence gate; no training")
        (args.output_dir/"contract.json").write_text(json.dumps(contract, indent=2)+"\n")
        generator = torch.Generator(device="cuda")
        rows = []
        for start in range(0, len(plan), args.num_envs):
            batch = plan[start:start+args.num_envs]
            outcomes = ms.run_batch(model, env, batch, seed=args.seed+start, generator=generator,
                                    audit_path=args.output_dir/'issued-actions.jsonl')
            with (args.output_dir/"episodes.jsonl").open("a") as stream:
                for row, outcome in zip(batch, outcomes):
                    record = dict(row, object_name=object_names[row["object"]], batch_seed=args.seed+start, **outcome)
                    rows.append(record)
                    stream.write(json.dumps(record)+"\n")
            print(json.dumps(dict(completed=len(rows), total=len(plan), success_once=ms.rates(rows))), flush=True)
        per_object = defaultdict(list)
        for row in rows:
            per_object[row["object"]].append(row)
        objects = [dict(object=i, name=object_names[i], split="train" if i < ms.TRAIN_OBJECTS else "held_out",
                        episodes=len(per_object[i]), success_once=ms.rates(per_object[i]),
                        success_at_end=ms.rates(per_object[i], "success_at_end"))
                   for i in range(len(object_names))]
        frozen_after = ms.model_fingerprint(model)
        if frozen_after != frozen_before:
            raise RuntimeError("Frozen motor weights changed during evaluation")
        summary = dict(completed=True, source_commit=source,
            frozen_model_before=frozen_before, frozen_model_after=frozen_after,
            native_actions=len(rows)*ms.STEPS, settling_steps=0, queries=len(rows)*(ms.STEPS//ms.CHUNK),
            contract_sha256=hashlib.sha256((args.output_dir/"contract.json").read_bytes()).hexdigest(),
            overall=dict(success_once=ms.rates(rows), success_at_end=ms.rates(rows, "success_at_end")),
            train_objects=dict(success_once=float(np.mean([o["success_once"] for o in objects[:ms.TRAIN_OBJECTS]]))),
            held_out_objects=dict(success_once=float(np.mean([o["success_once"] for o in objects[ms.TRAIN_OBJECTS:]]))),
            objects=objects, elapsed_seconds=time.monotonic()-started,
            peak_cuda_memory_gib=torch.cuda.max_memory_allocated()/2**30, training_launched=False)
        (args.output_dir/"summary.json").write_text(json.dumps(summary, indent=2)+"\n")
        print(json.dumps(summary), flush=True)
    except BaseException as error:
        (args.output_dir/"failure.json").write_text(json.dumps(dict(error_type=type(error).__name__,
            error=str(error), source_commit=source, training_launched=False), indent=2)+"\n")
        raise
    finally:
        if env is not None:
            env.env.close()


if __name__ == "__main__":
    main()
