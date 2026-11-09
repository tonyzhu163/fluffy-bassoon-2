#!/usr/bin/env python
"""One allocated GPU's complete native episodes for the existing PPO driver."""
import argparse
import json
import os
from pathlib import Path
import platform
import subprocess

import imageio.v2 as imageio
import torch
from hydra import compose, initialize_config_dir

from fluffy_bassoon.baige_gpu import verify_selected_cuda
from fluffy_bassoon.meta_rl.residual_ppo import ResidualActorCritic, ResidualPPOConfig
from fluffy_bassoon.robocasa.residual import (
    FrozenRoboCasaMotor, RoboCasaResidualEnvironment, collect_pending, collect_pending_lanes,
    native_environment_factory,
)
from fluffy_bassoon.robocasa.residual_parallel import rollout_envs_per_gpu
from fluffy_bassoon.rlinf.native_ppo import ActionJournal, snapshot_rng
from fluffy_bassoon.meta_rl.warmstart import atomic_save
from fluffy_bassoon.rl.action_residual import ResidualBound


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True)
    args = parser.parse_args()
    if platform.system() != "Linux" or torch.cuda.device_count() != 1:
        raise RuntimeError("Exactly one selected allocated GPU per native rollout worker")
    verify_selected_cuda(torch)
    request = torch.load(args.request, map_location="cpu", weights_only=False)
    profile, code = request["profile"], Path(__file__).resolve().parents[2]
    configured_lanes = rollout_envs_per_gpu(profile)
    head = lambda path: subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
    if head(code) != profile["source_commit"] or head(os.environ["RLINF_PATH"]) != profile["rlinf_execution_commit"]:
        raise ValueError("Immutable reviewed source for native workers")
    with initialize_config_dir(config_dir=str(code / "benchmarks/robocasa/configs"), version_base=None):
        cfg = compose(config_name="robocasa365_pi05_eval")
    initializer = profile["initializer"]
    cfg.rollout.model.model_path = initializer["model_directory"]
    cfg.rollout.model.openpi_data.norm_stats_path = initializer["norm_path"]
    from rlinf.models.embodiment.openpi import get_model
    model = get_model(cfg.rollout.model, torch_dtype=torch.bfloat16).to("cuda").eval().requires_grad_(False)
    stats = json.loads(Path(initializer["norm_path"]).read_text())
    motor = FrozenRoboCasaMotor(model, stats.get("norm_stats", stats), seed=profile["motor_seed"])
    policy = ResidualActorCritic(ResidualPPOConfig(**request["config"])).to("cuda").eval()
    policy.load_state_dict(request["policy"], strict=True)
    bound = ResidualBound(**request["bound"])
    cap = None if profile["maximum_all_in_actions"] is None else 1_000_000
    lanes = min(configured_lanes, len(request["cases"]))
    # One journal per lane; the rank directory sums every lane's escrow.
    journals = [ActionJournal(profile["accounting_root"], rank=request["rank"], world=request["world"],
                              attempt=request["attempt"] + ("" if lane == 0 else f"-lane{lane}"),
                              maximum_actions=cap) for lane in range(lanes)]
    environments = [RoboCasaResidualEnvironment(native_environment_factory(cfg.env.eval), request["cases"],
        horizon=request["horizon"], output=Path(request["output"]) / "videos",
        monitoring_root=request["monitoring_root"], journal=journal, writer_factory=imageio.get_writer,
        episode_id_prefix=request["episode_id_prefix"], rank=request["rank"]) for journal in journals]
    try:
        if lanes == 1:
            records, episodes = collect_pending(policy, bound, motor, environments[0], episodes=len(request["cases"]),
                generator=None, deterministic=request["deterministic"], verify_zero=request["verify_zero"],
                planned_motor_seeds=request["seeds"], planned_action_noise=request["noise"])
            state = environments[0].state_dict()
        else:
            records, episodes = collect_pending_lanes(policy, bound, motor, environments,
                planned_motor_seeds=request["seeds"], planned_action_noise=request["noise"],
                deterministic=request["deterministic"], verify_zero=request["verify_zero"])
            state = dict(environments[0].state_dict(), cursor=len(request["cases"]))
        if configured_lanes != 1:
            state.update(rollout_lanes_per_gpu=configured_lanes,
                         lane_states=[env.state_dict() for env in environments])
        atomic_save(Path(request["output"]) / "result.pt", dict(cases=request["cases"], records=records,
            episodes=episodes, environment=state, global_rng=snapshot_rng(),
            motor_rng=motor.state_dict(), physical_actions=journals[0].charged(),
            rollout_lanes_per_gpu=configured_lanes, actual_lane_count=lanes))
    finally:
        for env in environments:
            env.close()


if __name__ == "__main__":
    main()
