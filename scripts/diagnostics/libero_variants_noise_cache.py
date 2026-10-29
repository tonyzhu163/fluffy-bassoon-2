"""Headroom of the released RLinf pi0.5 LIBERO motor on LIBERO-Plus/PRO under fixed noise arms.

outputs/libero-noise-icrl-20261029-r1/PLAN.md, Stage L0, no training. For each package
(--kinds plus,pro), selects --per-category LIBERO-10 variants of every perturbation type
(seeded, spread over base tasks; 'clean' is the unperturbed base task) and records success
for every (variant, scene, arm, repeat); arm -1 is native sampling. Scenes are the
variant's first --scenes initial states, with one fixture seed per scene shared by all arms.
Each <kind>/ directory holds outcomes.jsonl, first-chunk trajectories and gates.json:
per perturbation type, gates G1-G4 with variants as conditions, determinism and mode
effect. Run inside an allocated GPU step (FLUFFY_GPU_EXECUTION=slurm or baige) with
MUJOCO_GL=egl and MUJOCO_EGL_DEVICE_ID set to the allocated device.
"""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/"src"))
from fluffy_bassoon.meta_rl import noise_screen as ns  # noqa: E402
from fluffy_bassoon.rl import libero_variants as lv  # noqa: E402


def episode_seed(base, kind, variant, scene, repeat):
    return int(np.random.default_rng([base, ("plus", "pro").index(kind), variant, scene, repeat]).integers(2**62))


def plan_rows(variants, *, scenes, arms, repeats, base_seed, init_states):
    rows = []
    for variant in variants:
        states = init_states(variant)
        if len(states) < scenes:
            raise ValueError(f"{variant['name']} has only {len(states)} initial states")
        for scene in range(scenes):
            for arm in [lv.NATIVE] + list(range(arms)):
                for repeat in range(repeats):
                    rows.append(dict(kind=variant["kind"], category=variant["category"], variant=variant["variant"],
                                     base_task=variant["base_task"], name=variant["name"],
                                     language=variant["language"], bddl=variant["bddl"], scene_index=scene,
                                     arm=arm, repeat=repeat, init_state=np.asarray(states[scene]),
                                     episode_seed=episode_seed(base_seed, variant["kind"], variant["variant"],
                                                               scene, repeat)))
    return rows


def analyse(directory):
    """Per perturbation type: gates over its variants, plus native and best-arm rates."""
    rows = ns.load_rows(directory/"outcomes.jsonl")
    trajectories = ns.load_trajectories(directory/"trajectories")
    by_category = defaultdict(list)
    for row in rows:
        by_category[row["category"]].append(row)
    result = dict(determinism=ns.determinism(rows), mode_effect=ns.mode_effect(rows, trajectories), categories={})
    for category, subset in sorted(by_category.items()):
        variants = sorted({r["variant"] for r in subset})
        entry = dict(variants=variants, episodes=len(subset),
                     native=float(np.mean([r["success_once"] for r in subset if r["arm"] == lv.NATIVE])))
        arms = sorted({r["arm"] for r in subset if r["arm"] != lv.NATIVE})
        entry["best_arm_pooled"] = float(max(np.mean([r["success_once"] for r in subset if r["arm"] == a])
                                             for a in arms))
        if len(variants) >= 2:
            gates = ns.gates(subset, condition="variant", train=variants, evaluate=variants)
            entry.update(gates={k: v["mean"] for k, v in gates["summary"].items()},
                         passed={k: v["passed"] for k, v in gates["summary"].items()},
                         all_passed=gates["all_passed"], per_variant=gates["per_condition"])
        else:
            entry["gates"] = "fewer than two variants"
        result["categories"][category] = entry
    (directory/"gates.json").write_text(json.dumps(result, indent=2, default=str)+"\n")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--kinds", default="plus,pro")
    parser.add_argument("--categories", help="comma-separated perturbation types to keep (default all)")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--norm-stats", type=Path, required=True)
    parser.add_argument("--expected-model-sha256", required=True)
    parser.add_argument("--per-category", type=int, default=2)
    parser.add_argument("--scenes", type=int, default=4)
    parser.add_argument("--arms", type=int, default=8)
    parser.add_argument("--arm-seed", type=int, default=5000320000, help="arm k uses seed arm_seed + k")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--variant-seed", type=int, default=0)
    parser.add_argument("--episode-seed", type=int, default=7)
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--steps", type=int, default=lv.STEPS)
    parser.add_argument("--time-budget-seconds", type=float, default=6*3600)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    kinds = args.kinds.split(",")
    keep = None if args.categories is None else set(args.categories.split(","))
    if (not set(kinds) <= {"plus", "pro"} or not 2 <= args.scenes <= 50 or not 2 <= args.arms <= 64
            or not 1 <= args.repeats <= 10 or not 1 <= args.per_category <= 50 or not 1 <= args.num_envs <= 64
            or not 1 <= args.steps <= 1000 or not args.time_budget_seconds > 0):
        parser.error("Require kinds in {plus,pro}, 2-50 scenes, 2-64 arms, 1-10 repeats, 1-64 envs")
    args.output.mkdir(parents=True, exist_ok=False)

    import torch
    from fluffy_bassoon.rl import maniskill_pi05 as ms
    from rlinf.models import get_model
    ms.prepare_gpu(ROOT, args.output)
    digest = ms.sha256_file(args.model_path/"model.safetensors")
    if digest != args.expected_model_sha256:
        raise RuntimeError(f"Checkpoint SHA256 {digest} differs from the expected value")
    torch.manual_seed(0)
    model = get_model(lv.model_config(args.model_path, args.norm_stats)).to("cuda").eval().requires_grad_(False)
    fingerprint = ms.model_fingerprint(model)

    def predict(states, prompts, noise):
        with torch.no_grad():
            actions, _ = model.predict_action_batch(lv.model_obs(states, prompts), mode="eval",
                                                    noise=torch.from_numpy(noise).to("cuda"))
        return actions.detach().float().cpu().numpy()

    arm_seeds = [args.arm_seed + k for k in range(args.arms)]
    arms = lv.arm_noise(arm_seeds)
    start, summary, timed_out = time.monotonic(), {}, False
    for kind in kinds:
        variants = [v for v in lv.enumerate_variants(kind) if keep is None or v["category"] in keep]
        chosen = lv.select(variants, args.per_category, args.variant_seed)
        rows = plan_rows(chosen, scenes=args.scenes, arms=args.arms, repeats=args.repeats,
                         base_seed=args.episode_seed, init_states=lv.init_states)
        directory = args.output/kind
        (directory/"trajectories").mkdir(parents=True)
        contract = dict(kind=kind, model_path=str(args.model_path), model_sha256=digest, model_fingerprint=fingerprint,
                        norm_stats=str(args.norm_stats), suite=lv.SUITE, categories=sorted({v["category"] for v in chosen}),
                        available_per_category={c: sum(v["category"] == c for v in variants)
                                                for c in sorted({v["category"] for v in variants})},
                        variants=[{k: v for k, v in c.items() if k != "init_states"} for c in chosen],
                        scenes=args.scenes, arms=args.arms, arm_seeds=arm_seeds,
                        arm_noise_sha256=[hashlib.sha256(a.tobytes()).hexdigest() for a in arms],
                        repeats=args.repeats, episodes=len(rows), steps=args.steps, settle=lv.SETTLE,
                        chunk=lv.CHUNK, horizon=lv.HORIZON, native_arm=lv.NATIVE,
                        perturbation="arm k passes arm_noise[k] as the initial noise of every chunk; arm -1 draws "
                                     "fresh noise per chunk seeded by (episode seed, chunk)")
        (directory/"contract.json").write_text(json.dumps(contract, indent=2)+"\n")
        pool = lv.EnvPool(args.num_envs, kind)
        written, first_chunks = 0, []
        try:
            with open(directory/"outcomes.jsonl", "w") as stream:
                for begin in range(0, len(rows), args.num_envs):
                    if time.monotonic() - start > args.time_budget_seconds:
                        timed_out = True
                        break
                    for record, chunk in lv.run_batch(predict, pool, rows[begin:begin+args.num_envs], arms=arms,
                                                      steps=args.steps):
                        record["row"] = written
                        stream.write(json.dumps(record)+"\n")
                        first_chunks.append(chunk if len(chunk) else np.zeros((1, lv.CHUNK, 7), np.float32))
                        written += 1
                    stream.flush()
        finally:
            pool.close()
        np.savez_compressed(directory/"trajectories/first-chunks.npz", row=np.arange(written),
                            issued=np.stack(first_chunks) if first_chunks else np.zeros((0, 1, lv.CHUNK, 7)))
        if timed_out:
            summary[kind] = dict(episodes=written, planned=len(rows), timed_out=True)
            break
        result = analyse(directory)
        summary[kind] = {c: dict(native=e["native"], best_arm_pooled=e["best_arm_pooled"],
                                 gates=e["gates"], all_passed=e.get("all_passed"))
                         for c, e in result["categories"].items()}
    if ms.model_fingerprint(model) != fingerprint:
        raise RuntimeError("Motor weights changed during the screen")
    (args.output/"summary.json").write_text(json.dumps(dict(summary=summary, timed_out=timed_out), indent=2)+"\n")
    print(json.dumps(dict(summary=summary, timed_out=timed_out), default=str))


if __name__ == "__main__":
    main()
