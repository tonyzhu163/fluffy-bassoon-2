"""Success of the released RLinf pi0.5 MetaWorld SFT under fixed noise arms and discrete goals.

outputs/libero-noise-icrl-20261029-r1/PLAN.md, Meta-World companion, no training. For
each task and goal visibility (marker visible or hidden), records success for every
(goal, scene, arm, repeat); arm -1 is native sampling seeded per episode. Each
(task, visibility) directory holds outcomes.jsonl, first-chunk trajectories and
gates.json: the noise-mode gates G1-G4 with goals as conditions, determinism, mode
effect and goal coverage (the mode-spread check, meaningful for the hidden goal).
Run inside an allocated GPU step (FLUFFY_GPU_EXECUTION=slurm or baige).
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/"src"))
from fluffy_bassoon.meta_rl import noise_screen as ns  # noqa: E402
from fluffy_bassoon.rl import metaworld_pi05 as mw  # noqa: E402


def episode_seed(base, task_index, visible, scene, repeat):
    return int(np.random.default_rng([base, task_index, int(visible), scene, repeat]).integers(2**62))


def load_policy(rlinf_root, model_path):
    sys.path.insert(0, str(rlinf_root))
    from toolkits.standalone_eval_scripts.openpi import create_trained_policy
    from rlinf.models.embodiment.openpi.dataconfig import _CONFIGS_DICT
    return create_trained_policy(_CONFIGS_DICT["pi05_metaworld"], str(model_path),
                                 sample_kwargs={"num_steps": mw.FLOW_STEPS})


def analyse(directory, goals):
    rows = ns.load_rows(directory/"outcomes.jsonl")
    result = ns.gates(rows, condition="variant", train=goals, evaluate=goals)
    result["determinism"] = ns.determinism(rows)
    result["mode_effect"] = ns.mode_effect(rows, ns.load_trajectories(directory/"trajectories"))
    result["goal_coverage"] = mw.goal_coverage(rows)
    (directory/"gates.json").write_text(json.dumps(result, indent=2, default=str)+"\n")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rlinf-root", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True, help="RLinf-Pi05-MetaWorld-SFT snapshot")
    parser.add_argument("--tasks", default="reach-v3,push-v3")
    parser.add_argument("--visibility", default="visible,hidden")
    parser.add_argument("--scenes", type=int, default=8)
    parser.add_argument("--scene-seed", type=int, default=20261029)
    parser.add_argument("--arms", type=int, default=16)
    parser.add_argument("--arm-seed", type=int, default=4000320000, help="arm k uses seed arm_seed + k")
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--episode-seed", type=int, default=7)
    parser.add_argument("--time-budget-seconds", type=float, default=3600)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    tasks, visibility = args.tasks.split(","), args.visibility.split(",")
    for name in tasks:
        mw.task_spec(name)
    if (not set(visibility) <= {"visible", "hidden"} or not 2 <= args.scenes <= 500
            or not 2 <= args.arms <= 256 or not 1 <= args.repeats <= 20 or not args.time_budget_seconds > 0):
        parser.error("Require visibility in {visible,hidden}, 2-500 scenes, 2-256 arms, 1-20 repeats")
    args.output.mkdir(parents=True, exist_ok=False)

    from fluffy_bassoon.rl import maniskill_pi05 as ms
    import torch
    ms.prepare_gpu(ROOT, args.output)
    policy = load_policy(args.rlinf_root, args.model_path)
    fingerprint = ms.model_fingerprint(policy._model)
    arm_seeds = [args.arm_seed + k for k in range(args.arms)]
    noise = mw.arm_noise(arm_seeds)
    contract = dict(
        model_path=str(args.model_path), model_sha256=ms.sha256_file(args.model_path/"model.safetensors"),
        model_fingerprint=fingerprint, rlinf_root=str(args.rlinf_root), tasks=tasks, visibility=visibility,
        goals={name: mw.TASKS[name]["goals"] for name in tasks}, scenes=args.scenes, scene_seed=args.scene_seed,
        arm_seeds=arm_seeds, arm_noise_sha256=[hashlib.sha256(n.tobytes()).hexdigest() for n in noise],
        noise_shape=list(mw.NOISE_SHAPE), native_arm=mw.NATIVE, repeats=args.repeats,
        episode_seed=args.episode_seed, camera_id=mw.CAMERA_ID, camera_pos=list(mw.CAMERA_POS),
        resolution=mw.RESOLUTION, settle=mw.SETTLE, chunk=mw.CHUNK, steps=mw.STEPS, flow_steps=mw.FLOW_STEPS,
        perturbation="arm k passes arm_noise[k] as the initial noise of every chunk; arm -1 samples fresh "
                     "noise every chunk from torch's generator seeded per episode")
    (args.output/"contract.json").write_text(json.dumps(contract, indent=2)+"\n")

    start, summary, timed_out = time.monotonic(), {}, False
    arms = [mw.NATIVE] + list(range(args.arms))
    for task_index, name in enumerate(tasks):
        _, goals = mw.task_spec(name)
        positions = mw.scenes(name, args.scenes, args.scene_seed)
        for visible in (v == "visible" for v in visibility):
            env = mw.make_env(name, goal_visible=visible)
            directory = args.output/f"{name}-{'visible' if visible else 'hidden'}"
            (directory/"trajectories").mkdir(parents=True)
            pixel_difference = mw.goal_visibility(env, name, positions[0])
            if not visible and pixel_difference:
                raise RuntimeError(f"{name}: hidden goal still changes {pixel_difference} pixels")
            rows, first_chunks = [], []
            with open(directory/"outcomes.jsonl", "w") as stream:
                for g, goal in enumerate(goals):
                    for scene, position in enumerate(positions):
                        for arm in arms:
                            for repeat in range(args.repeats):
                                if time.monotonic() - start > args.time_budget_seconds:
                                    timed_out = True
                                    break
                                torch.manual_seed(episode_seed(args.episode_seed, task_index, visible, scene, repeat))
                                fixed = None if arm == mw.NATIVE else noise[arm]
                                success, digest, issued = mw.rollout(
                                    env, lambda batch: policy.infer(batch, noise=fixed)["actions"],
                                    name, goal, position)
                                row = dict(row=len(rows), task=name, goal_visible=visible, variant=g,
                                           scene_index=scene, arm=arm, repeat=repeat, success_once=success,
                                           issued_actions_sha256=digest, slot=0, chunks=len(issued))
                                rows.append(row)
                                first_chunks.append(issued[:1])
                                stream.write(json.dumps(row)+"\n")
            env.close()
            np.savez_compressed(directory/"trajectories/first-chunks.npz", row=np.arange(len(rows)),
                                issued=np.stack(first_chunks))
            if timed_out:
                break
            result = analyse(directory, list(range(len(goals))))
            summary[directory.name] = dict(
                pixels_changed_by_goal=pixel_difference, all_passed=result["all_passed"],
                gates={k: v["mean"] for k, v in result["summary"].items()},
                native=float(np.mean([r["success_once"] for r in rows if r["arm"] == mw.NATIVE])),
                goal_coverage_passed=result["goal_coverage"]["passed"],
                deterministic=result["determinism"]["deterministic"])
        if timed_out:
            break
    if ms.model_fingerprint(policy._model) != fingerprint:
        raise RuntimeError("Motor weights changed during the screen")
    (args.output/"summary.json").write_text(json.dumps(dict(summary=summary, timed_out=timed_out), indent=2)+"\n")
    print(json.dumps(dict(summary=summary, timed_out=timed_out)))


if __name__ == "__main__":
    main()
