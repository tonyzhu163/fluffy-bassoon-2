"""Instruction-swap check for frozen RLinf pi0.5 on ManiSkill (no training).

Modes (one simulator per process; run them as separate steps):
  single     RLinf PutOnPlateInScene25Main-v3, one object + plate, native instruction.
  multi      Two objects + plate (RLinf MultiCarrot scene, subclass in
             fluffy_bassoon.rl.maniskill_instruction_swap). Each scene block runs three
             times with the same episode ids, reset seed and sampler noise; only the
             instruction differs: names slot-0 object (RLinf's native prompt), names the
             slot-1 object, or the vague "put object on plate".
  summarize  Combine single/multi episode files into metrics.json (CPU only).
See outputs/pi05-instruction-swap-thu2-20261029-r1/PLAN.md.
"""
import argparse
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np


def parse_pairs(text):
    pairs = [tuple(int(x) for x in item.split(":")) for item in text.split(",") if item]
    if not pairs or any(len(p) != 2 or p[0] == p[1] or not all(0 <= x < 16 for x in p) for p in pairs):
        raise ValueError("Pairs are a:b with distinct training objects 0-15")
    return pairs


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("single", "multi", "summarize"), required=True)
    parser.add_argument("--rlinf-root", type=Path)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--norm-stats", type=Path)
    parser.add_argument("--expected-model-sha256")
    parser.add_argument("--checkpoint-label")
    parser.add_argument("--source-commit")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--objects", default="0,9,12,13")
    parser.add_argument("--pairs", default="0:9,12:13,0:13,9:12")
    parser.add_argument("--per-object", type=int, default=16)
    parser.add_argument("--per-pair", type=int, default=24)
    parser.add_argument("--num-envs", type=int, default=32)
    parser.add_argument("--seed", type=int, default=2910290000)
    parser.add_argument("--max-elapsed-seconds", type=float, default=1500,
                        help="no new block starts if elapsed + 1.25 x last block time would exceed this")
    parser.add_argument("--single-dir", type=Path)
    parser.add_argument("--multi-dir", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root/"src"))
    from fluffy_bassoon.rl import maniskill_pi05 as ms
    from fluffy_bassoon.rl import maniskill_instruction_swap as swap

    if args.mode == "summarize":
        single = read_jsonl(args.single_dir/"episodes.jsonl") if args.single_dir else []
        multi = read_jsonl(args.multi_dir/"episodes.jsonl")
        names = json.loads((args.multi_dir/"contract.json").read_text())["object_names"]
        metrics = swap.summarize(single, multi, names)
        metrics.update(single_episodes=len(single), multi_episodes=len(multi))
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir/"metrics.json").write_text(json.dumps(metrics, indent=2)+"\n")
        print(json.dumps({k: v for k, v in metrics.items() if k != "pair_records"}, indent=2))
        return

    if sys.platform != "linux":
        raise RuntimeError("Run model/simulator work on allocated Linux only")
    mode = ms.execution_mode()
    objects = [int(x) for x in args.objects.split(",")]
    pairs = parse_pairs(args.pairs)
    total = len(objects)*args.per_object if args.mode == "single" else len(pairs)*args.per_pair
    if (int(os.environ.get("SLURM_STEP_NUM_TASKS", "1")) != 1 or not 1 <= args.num_envs <= 64
            or total % args.num_envs or not all(0 <= o < ms.TRAIN_OBJECTS for o in objects)
            or not 0 <= args.seed < 2**32 - 1000 or not args.checkpoint_label):
        raise ValueError("Require one process, batches dividing the plan, training objects, valid seed, a label")
    source = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    if source != args.source_commit or subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True).strip():
        raise ValueError("Require exact clean reviewed source")
    if ms.sha256_file(args.model_path/"model.safetensors") != args.expected_model_sha256:
        raise ValueError("pi0.5 checkpoint SHA256 differs")
    if args.norm_stats.name != "norm_stats.json" or not args.norm_stats.is_file():
        raise FileNotFoundError("Pass the checkpoint's own norm_stats.json")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    ms.prepare_gpu(root, args.output_dir)
    sys.path.insert(0, str(args.rlinf_root))
    import torch
    from rlinf.envs.sim.maniskill.maniskill_env import ManiskillEnv
    import rlinf
    import rlinf.envs.sim.maniskill.tasks  # noqa: F401 -- registers RLinf scenes incl. variants
    from rlinf.models import get_model
    if not Path(rlinf.__file__).resolve().is_relative_to(args.rlinf_root.resolve()):
        raise RuntimeError(f"rlinf imported from {rlinf.__file__}, not --rlinf-root")
    if torch.cuda.device_count() != 1:
        raise ValueError("Require one allocated physical GPU")
    torch.manual_seed(args.seed)
    env = None
    try:
        if args.mode == "multi":
            swap.register_env()
            cfg = swap.env_config(args.rlinf_root, args.num_envs, args.seed)
        else:
            cfg = ms.env_config(args.rlinf_root, args.num_envs, args.seed)
        env = ManiskillEnv(cfg, args.num_envs, seed_offset=0, total_num_processes=1, worker_info=None,
                           record_metrics=True)
        ms.record_renderer_binding(env, args.output_dir)
        model = get_model(ms.model_config(args.model_path, args.norm_stats)).cuda().eval()
        model.requires_grad_(False)
        frozen_before = ms.model_fingerprint(model)
        base = env.env.unwrapped
        names = [base.model_db_carrot[n]["name"] for n in base.carrot_names]
        plate_name = base.model_db_plate[base.plate_names[0]]["name"]
        rng = np.random.default_rng(args.seed)
        if args.mode == "single":
            plan = ms.sample_scenes(base.basic_obj_infos, objects, args.per_object, rng)
            blocks = [plan[i:i+args.num_envs] for i in range(0, len(plan), args.num_envs)]
        else:
            plan = swap.plan_multi_scenes(base.basic_obj_infos, pairs, args.per_pair, rng)
            for scene in plan:
                decoded = swap.decode_multi_episode_id(base.basic_obj_infos, scene["episode_id"])
                if (decoded["slot0"], decoded["slot1"], decoded["overlay"], decoded["position"],
                        decoded["orientation"]) != (scene["slot0"], scene["slot1"], scene["overlay"],
                                                    scene["position"], scene["orientation"]):
                    raise AssertionError("Episode id does not round-trip")
            blocks = swap.interleave_blocks(plan, args.num_envs)
        contract = dict(source_commit=source, mode=args.mode, checkpoint_label=args.checkpoint_label,
            rlinf_root=str(args.rlinf_root),
            rlinf_checkpoint_loader_sha256=ms.sha256_file(args.rlinf_root/"rlinf/models/embodiment/openpi/checkpoint.py"),
            rlinf_multi_carrot_sha256=ms.sha256_file(args.rlinf_root/"rlinf/envs/sim/maniskill/tasks/variants/put_on_plate_25_multi_carrot.py"),
            model_path=str(args.model_path), model_sha256=args.expected_model_sha256,
            norm_stats=str(args.norm_stats), norm_stats_sha256=ms.sha256_file(args.norm_stats),
            task=cfg.init_params.id, obj_set=cfg.init_params.obj_set, basic_obj_infos=list(base.basic_obj_infos),
            object_names=names, plate_name=plate_name, objects=objects, pairs=pairs,
            per_object=args.per_object, per_pair=args.per_pair, num_envs=args.num_envs, seed=args.seed,
            conditions=list(swap.CONDITIONS) if args.mode == "multi" else ["native"],
            vague_instruction=swap.VAGUE_INSTRUCTION, instruction_template="put {object} on {plate}",
            max_elapsed_seconds=args.max_elapsed_seconds, planned_blocks=len(blocks),
            planned_episodes=len(plan)*(len(swap.CONDITIONS) if args.mode == "multi" else 1),
            max_episode_steps=ms.STEPS, action_chunk=ms.CHUNK, denoise_steps=4,
            sampler="RLinf eval-mode, generator re-seeded per batch; same seed for all conditions of a block",
            slot_fields=swap.SLOT_FIELDS, outcome_fields=ms.OUTCOME_FIELDS,
            runtime={name: version(name) for name in ("torch", "mani-skill", "sapien", "rlinf-openpi", "jax")},
            job_id=ms.job_id(), execution=mode, frozen_model_sha256=frozen_before,
            role="frozen pi0.5 instruction-following check; no training")
        (args.output_dir/"contract.json").write_text(json.dumps(contract, indent=2)+"\n")
        (args.output_dir/"plan.json").write_text(json.dumps(blocks, indent=2)+"\n")
        generator = torch.Generator(device="cuda")
        completed, mismatches, stopped, last_block = 0, 0, None, 0.0
        for index, block in enumerate(blocks):
            if time.monotonic() - started + 1.25*last_block > args.max_elapsed_seconds:
                stopped = f"elapsed cap before block {index}"
                break
            block_started = time.monotonic()
            seed = args.seed + index
            if args.mode == "single":
                outcomes = ms.run_batch(model, env, block, seed=seed, generator=generator)
                records = [dict(scene, object_name=names[scene["object"]], checkpoint=args.checkpoint_label,
                                block=index, block_seed=seed, **outcome) for scene, outcome in zip(block, outcomes)]
                swap.write_jsonl(args.output_dir/"episodes.jsonl", records)
                completed += len(records)
            else:
                for condition in swap.CONDITIONS:
                    prompts = swap.condition_instructions(block, names, condition, plate_name)
                    outcomes, mismatch = swap.run_multi_batch(model, env, block, prompts, seed=seed,
                        generator=generator, native_instruction=condition == "name_slot0")
                    mismatches += mismatch
                    records = [dict(scene, scene_key=f"{index}:{i}", condition=condition, instruction=prompt,
                                    slot0_name=names[scene["slot0"]], slot1_name=names[scene["slot1"]],
                                    checkpoint=args.checkpoint_label, block=index, block_seed=seed, **outcome)
                               for i, (scene, prompt, outcome) in enumerate(zip(block, prompts, outcomes))]
                    swap.write_jsonl(args.output_dir/"episodes.jsonl", records)
                    completed += len(records)
            last_block = time.monotonic() - block_started
            print(json.dumps(dict(block=index, completed=completed, block_seconds=last_block,
                                  elapsed=time.monotonic()-started)), flush=True)
        frozen_after = ms.model_fingerprint(model)
        if frozen_after != frozen_before:
            raise RuntimeError("Frozen motor weights changed during evaluation")
        summary = dict(completed=stopped is None, stopped=stopped, episodes=completed, source_commit=source,
            frozen_model_before=frozen_before, frozen_model_after=frozen_after,
            slot0_success_mismatch_steps=mismatches,
            contract_sha256=hashlib.sha256((args.output_dir/"contract.json").read_bytes()).hexdigest(),
            elapsed_seconds=time.monotonic()-started,
            peak_cuda_memory_gib=torch.cuda.max_memory_allocated()/2**30, training_launched=False)
        if args.mode == "multi":
            summary["metrics"] = {k: v for k, v in swap.summarize([], swap_rows(args.output_dir), names).items()
                                  if k != "pair_records"}
        (args.output_dir/"summary.json").write_text(json.dumps(summary, indent=2)+"\n")
        print(json.dumps(summary), flush=True)
    except BaseException as error:
        (args.output_dir/"failure.json").write_text(json.dumps(dict(error_type=type(error).__name__,
            error=str(error), source_commit=source, training_launched=False), indent=2)+"\n")
        raise
    finally:
        if env is not None:
            env.env.close()


def swap_rows(output_dir):
    path = output_dir/"episodes.jsonl"
    return read_jsonl(path) if path.exists() else []


if __name__ == "__main__":
    main()
