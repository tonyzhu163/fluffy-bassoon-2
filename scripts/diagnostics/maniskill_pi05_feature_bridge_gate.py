"""One live native episode with exact zero-bridge CTM/XL action parity at every chunk."""

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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rlinf-root", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--norm-stats", type=Path, required=True)
    parser.add_argument("--expected-model-sha256", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--object", type=int, default=0)
    parser.add_argument("--seed", type=int, default=3000320000)
    parser.add_argument("--width", type=int, default=64)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root / "src"))
    from fluffy_bassoon.rl import maniskill_pi05 as ms
    if sys.platform != "linux":
        raise RuntimeError("Run native motor qualification on allocated Linux only")
    mode = ms.execution_mode()
    if (int(os.environ.get("SLURM_STEP_NUM_TASKS", "1")) != 1 or not 0 <= args.object < ms.OBJECTS
            or not 0 <= args.seed < 2**32 or args.width < 4 or args.width % 4):
        raise ValueError("Require one process, valid object/seed and width divisible by four")
    source = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    if source != args.source_commit or subprocess.check_output(["git", "status", "--porcelain"], cwd=root,
                                                               text=True).strip():
        raise ValueError("Require the exact clean reviewed source")
    if ms.sha256_file(args.model_path / "model.safetensors") != args.expected_model_sha256:
        raise ValueError("Frozen motor checkpoint SHA256 differs")
    if args.norm_stats.name != "norm_stats.json" or not args.norm_stats.is_file():
        raise FileNotFoundError("Pass the checkpoint's own norm_stats.json")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    started, completed_chunks, env = time.monotonic(), 0, None
    try:
        ms.prepare_gpu(root, args.output_dir)
        sys.path.insert(0, str(args.rlinf_root))
        import torch
        from rlinf.envs.action_utils import prepare_actions
        from rlinf.envs.sim.maniskill.maniskill_env import ManiskillEnv
        import rlinf.envs.sim.maniskill.tasks  # noqa: F401 -- environment registration
        from rlinf.models import get_model
        from fluffy_bassoon.rl.pi05_feature_bridge import (
            Pi05FeatureBridge, clone_generator, predict_action_batch_with_bridge, public_chunk_feedback,
        )
        if torch.cuda.device_count() != 1:
            raise ValueError("Require exactly one allocated visible GPU")
        torch.manual_seed(args.seed)
        env = ManiskillEnv(ms.env_config(args.rlinf_root, 1, args.seed), 1, seed_offset=0,
                           total_num_processes=1, worker_info=None, record_metrics=True)
        ms.record_renderer_binding(env, args.output_dir)
        model = get_model(ms.model_config(args.model_path, args.norm_stats)).cuda().eval()
        model.requires_grad_(False)
        if (model.num_steps, model.action_horizon, model.action_dim) != (4, 8, 32):
            raise ValueError("Native motor dimensions differ from the qualified four-step pi0.5 contract")
        before = ms.model_fingerprint(model)
        bridges = {kind: Pi05FeatureBridge(core=kind, width=args.width).cuda().eval()
                   for kind in ("persistent_ctm", "transformer_xl")}
        bridge_before = {kind: ms.model_fingerprint(module) for kind, module in bridges.items()}
        memories = {kind: None for kind in bridges}
        base = env.env.unwrapped
        ms.check_object_infos(base.basic_obj_infos)
        scene = ms.sample_scenes(base.basic_obj_infos, [args.object], 1,
                                 np.random.default_rng(args.seed))[0]
        contract = dict(source_commit=source, job_id=ms.job_id(), execution=mode, seed=args.seed,
                        model_path=str(args.model_path), model_sha256=args.expected_model_sha256,
                        norm_stats=str(args.norm_stats), norm_stats_sha256=ms.sha256_file(args.norm_stats),
                        frozen_model_before=before, scene=scene, task="PutOnPlateInScene25Main-v3",
                        episode_steps=ms.STEPS, action_chunk=ms.CHUNK, denoise_steps=4,
                        bridges=list(bridges), width=args.width, residual_initialization="exact zero",
                        comparison="same live observation; cloned CUDA generator; exact torch.equal",
                        feedback_fields=["previous executed action (7)", "previous attempt reward",
                                         "attempt boundary", "attempt index / K", "chunk index / chunks"],
                        feedback_reward="zero: this qualification has only the first attempt",
                        source_hashes={name: ms.sha256_file(args.rlinf_root / name) for name in (
                            "rlinf/models/embodiment/openpi/pi0.py",
                            "rlinf/models/embodiment/openpi/env_io.py",
                            "rlinf/models/embodiment/openpi/modules/gemma.py",
                            "rlinf/models/embodiment/openpi/checkpoint.py")},
                        runtime={name: version(name) for name in ("torch", "mani-skill", "sapien", "rlinf-openpi")},
                        training_launched=False, scope="zero-bridge parity and one native live episode only")
        (args.output_dir / "contract.json").write_text(json.dumps(contract, indent=2) + "\n")
        obs, _ = env.reset(seed=args.seed, options={"episode_id": torch.tensor([scene["episode_id"]],
                                                                              device=env.device)})
        if base.select_carrot_ids.cpu().tolist() != [args.object]:
            raise AssertionError("Simulator object differs from the planned scene")
        outcomes = ms.EpisodeOutcomes(1)
        generator = torch.Generator(device="cuda").manual_seed(args.seed)
        previous_action = torch.zeros(1, 7, device="cuda", dtype=torch.float32)
        instructions = list(obs["task_descriptions"])
        with torch.no_grad(), (args.output_dir / "issued-actions.jsonl").open("x", buffering=1) as stream:
            for chunk_index in range(ms.STEPS // ms.CHUNK):
                feedback = public_chunk_feedback(previous_action, chunk_index=chunk_index)
                state_before = generator.get_state().clone()
                native_actions, native_result = model.predict_action_batch(obs, mode="eval", rng=generator)
                state_after = generator.get_state().clone()
                native_chunk = prepare_actions(raw_chunk_actions=native_actions, env_type="maniskill",
                                               model_type="openpi", num_action_chunks=ms.CHUNK,
                                               action_dim=7, policy="widowx_bridge")
                comparisons = {}
                for kind, bridge in bridges.items():
                    replay_generator = clone_generator(generator).set_state(state_before)
                    audit = {}
                    actions, memories[kind], model_actions = predict_action_batch_with_bridge(
                        model, bridge, obs, feedback, memories[kind], rng=replay_generator,
                        return_model_actions=True, audit=audit)
                    prepared = prepare_actions(raw_chunk_actions=actions, env_type="maniskill",
                                               model_type="openpi", num_action_chunks=ms.CHUNK,
                                               action_dim=7, policy="widowx_bridge")
                    equalities = dict(model_actions=torch.equal(model_actions, native_result["model_actions"]),
                                      decoded_actions=torch.equal(actions, native_actions),
                                      prepared_actions=torch.equal(prepared, native_chunk),
                                      noise_stream=torch.equal(replay_generator.get_state(), state_after))
                    if not all(equalities.values()):
                        raise AssertionError(f"Zero {kind} bridge parity failed at chunk {chunk_index}: {equalities}")
                    if len(audit["suffix_shapes"]) != 4 or audit["recurrence_updates"] != 1:
                        raise AssertionError("Bridge recurrence or native denoising count changed")
                    comparisons[kind] = dict(equalities=equalities, **audit)
                record = dict(chunk_index=chunk_index, step_start=chunk_index * ms.CHUNK,
                              feedback=feedback.cpu().tolist(), native_model_action_shape=list(native_result["model_actions"].shape),
                              decoded_action_shape=list(native_actions.shape), prepared_action_shape=list(native_chunk.shape),
                              executed_actions=native_chunk.cpu().tolist(), comparisons=comparisons)
                # Execute the proven common action once; all methods share this live trajectory.
                obs_list, _, _, _, infos_list = env.chunk_step(native_chunk)
                for info in infos_list:
                    outcomes.update(info)
                previous_action = native_chunk[:, -1].detach().to(device="cuda", dtype=torch.float32)
                obs = obs_list[-1]
                completed_chunks += 1
                stream.write(json.dumps(record, allow_nan=False) + "\n")
                print(json.dumps(dict(completed_chunks=completed_chunks, total_chunks=16,
                                      exact_parity=True, cores=list(bridges))), flush=True)
        after = ms.model_fingerprint(model)
        bridge_after = {kind: ms.model_fingerprint(module) for kind, module in bridges.items()}
        if before != after or bridge_before != bridge_after:
            raise AssertionError("Motor or bridge weights changed during qualification")
        if ms.sha256_file(args.model_path / "model.safetensors") != args.expected_model_sha256:
            raise AssertionError("Frozen checkpoint file changed during qualification")
        if completed_chunks != 16:
            raise AssertionError("Incomplete native episode")
        summary = dict(completed=True, source_commit=source, job_id=ms.job_id(),
                       exact_parity=True, chunks=completed_chunks, native_actions=ms.STEPS,
                       methods=["native", *bridges], recurrence_updates_per_bridge=completed_chunks,
                       frozen_model_before=before, frozen_model_after=after,
                       bridge_before=bridge_before, bridge_after=bridge_after,
                       instruction=instructions[0], outcome=outcomes.rows()[0],
                       contract_sha256=hashlib.sha256((args.output_dir / "contract.json").read_bytes()).hexdigest(),
                       elapsed_seconds=time.monotonic() - started,
                       peak_cuda_memory_gib=torch.cuda.max_memory_allocated() / 2**30,
                       training_launched=False, adaptation_tested=False)
        (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps(summary), flush=True)
    except BaseException as error:
        (args.output_dir / "failure.json").write_text(json.dumps(dict(error_type=type(error).__name__,
            error=str(error), completed_chunks=completed_chunks, source_commit=source,
            training_launched=False), indent=2) + "\n")
        raise
    finally:
        if env is not None:
            env.env.close()


if __name__ == "__main__":
    main()
