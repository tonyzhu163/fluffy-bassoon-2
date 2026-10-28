"""Official AMAGO1 off-policy learner with the common Meta-World V3 trial contract.

The upstream Agent, loss reduction, AdamW optimizer and target updates are used
directly. This module replaces collection, replay indexing and checkpointing;
the original experiment driver cannot resume environment/replay/RNG state.
"""

import argparse
from contextlib import contextmanager
from fractions import Fraction
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import signal
import sys
import time

import gymnasium as gym
import numpy as np
import torch

from fluffy_bassoon.evaluation_stages import write_json
from .checkpoint import rng_state, restore_rng
from .metaworld_env import benchmark_pools, make_lanes, task_digest
from .metaworld_workers import batch_reset, batch_step as step_lanes, make_worker_lanes
from .screen import allocation, clean_source


UPSTREAM_COMMIT = "514c1c2a471418aaf7829ba60ad0582b7764e643"
PROTOCOL = "metaworld_official_amago1_v3_adapter_v1"
CACHED_ATTENTION = ("flash", "torch-sdpa")


class TorchSDPAAttention(torch.nn.Module):
    """Parameter-free replacement for upstream FlashAttention2's primitive.

    Keep its BF16 QKV/cache convention, scale, causality and dropout. Projections,
    positional embeddings, head scaling, residual blocks and Agent are upstream.
    PyTorch selects its available native SDPA kernel; no CUDA extension is built.
    """

    def __init__(self, causal=True, attention_dropout=0.):
        super().__init__()
        self.causal = causal
        self.dropout = attention_dropout

    def forward(self, qkv, key_cache=None, val_cache=None, cache_seqlens=None):
        q, k, v = qkv.to(torch.bfloat16).unbind(2)
        cached = (key_cache is not None, val_cache is not None, cache_seqlens is not None)
        if any(cached) and not all(cached):
            raise ValueError("SDPA cache requires keys, values and per-lane lengths")
        dropout = self.dropout if self.training else 0.
        if not any(cached):
            out = torch.nn.functional.scaled_dot_product_attention(
                q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
                dropout_p=dropout, is_causal=self.causal)
        else:
            if self.training or q.shape[1] != 1:
                raise ValueError("Native SDPA cache supports eval-mode one-step queries")
            batch = q.shape[0]
            if (key_cache.shape != val_cache.shape or key_cache.shape[0] != batch
                    or key_cache.shape[2:] != k.shape[2:] or cache_seqlens.shape != (batch,)
                    or key_cache.dtype != torch.bfloat16 or val_cache.dtype != torch.bfloat16):
                raise ValueError("SDPA cache differs from the upstream BF16 cache contract")
            minimum, maximum = torch.aminmax(cache_seqlens)
            # One sync per attention layer selects an actual-length key slice.
            # Full-capacity keys would make short-trial inference needlessly slow.
            low, high = torch.stack((minimum, maximum)).tolist()
            if low < 0 or high >= key_cache.shape[1]:
                raise ValueError("SDPA cache length lies outside its allocated capacity")
            lanes = torch.arange(batch, device=q.device)
            positions = cache_seqlens.long()
            key_cache[lanes, positions] = k[:, 0]
            val_cache[lanes, positions] = v[:, 0]
            stop = high + 1
            keys, values = key_cache[:, :stop], val_cache[:, :stop]
            allowed = torch.arange(stop, device=q.device)[None] <= positions[:, None]
            if low != high:
                # Upstream initializes unused cache slots to NaN. A mask alone
                # does not prevent NaN in QK or zero*NaN in the value reduction.
                keys = torch.where(allowed[:, :, None, None], keys, 0.)
                values = torch.where(allowed[:, :, None, None], values, 0.)
            out = torch.nn.functional.scaled_dot_product_attention(
                q.transpose(1, 2), keys.transpose(1, 2), values.transpose(1, 2),
                attn_mask=allowed[:, None, None, :], dropout_p=0., is_causal=False)
            # A one-token query has an offset within the cache. is_causal=True
            # would incorrectly retain only key zero rather than all past keys.
        return out.transpose(1, 2)


def install_torch_sdpa(policy):
    """Swap only parameter-free primitives; keep the upstream state schema."""
    from amago.nets.transformer import FlashAttention
    changed = 0
    for module in list(policy.modules()):
        primitive = getattr(module, "attention", None)
        if isinstance(primitive, FlashAttention):
            replacement = TorchSDPAAttention(primitive.causal, primitive.dropout)
            replacement.train(primitive.training)
            module.attention = replacement
            changed += 1
    if changed != policy.traj_encoder.tformer.n_layers:
        raise ValueError("Could not replace exactly the official attention primitives")
    return changed


def vendor_root():
    return Path(__file__).resolve().parents[3] / "third_party" / "amago"


def load_upstream():
    """Reject edited or substituted upstream algorithm files before importing."""
    root = vendor_root()
    source = json.loads((root / "SOURCE.json").read_text())
    if source["commit"] != UPSTREAM_COMMIT or source["modified_upstream_files"]:
        raise ValueError("Require the unmodified pinned AMAGO1 source")
    for relative, expected in source["files"].items():
        if hashlib.sha256((root / relative).read_bytes()).hexdigest() != expected:
            raise ValueError(f"AMAGO1 upstream source changed: {relative}")
    sys.path.insert(0, str(root / "upstream"))
    import amago
    if Path(amago.__file__).resolve() != (root / "upstream/amago/__init__.py").resolve():
        raise ValueError("Another AMAGO distribution was imported before the pinned source")
    return amago


def synchronize(device):
    if str(device).startswith("cuda"):
        torch.cuda.synchronize(device)


@contextmanager
def preserve_random_streams():
    saved = rng_state()
    try:
        yield
    finally:
        restore_rng(saved)


@contextmanager
def extend_evaluation_positions(policy):
    """Preserve the trained table/schema and clamp only extrapolated lookups.

    Raw absolute timestep indices stay unchanged in trajectories and Transformer
    calls. This evaluation-only prehook maps an otherwise out-of-range learned
    position lookup to its last allocated training-table row. No parameters are
    added or reset; extrapolation is not a claim that this row encodes later time.
    """
    embedding = policy.traj_encoder.tformer.position_embedding
    if not isinstance(embedding, torch.nn.Embedding):
        raise ValueError("AMAGO evaluation extension expects its original learned position table")
    def clamp_lookup(module, inputs):
        return (inputs[0].clamp(max=module.num_embeddings - 1),)
    handle = embedding.register_forward_pre_hook(clamp_lookup)
    try:
        yield
    finally:
        handle.remove()


def make_experiment(amago, *, horizon, max_seq_len=2000, memory_size=256,
                    memory_layers=3, attention="flash", batch_size=24,
                    grad_updates=1000, warmup_epochs=10, warmup_grad_updates=None,
                    device="cuda"):
    """Use original example defaults, with all method-specific losses intact."""
    import gin
    gin.clear_config()
    config = {}
    amago.cli_utils.turn_off_goal_conditioning(config)
    amago.cli_utils.switch_traj_encoder(config, arch="transformer",
                                      memory_size=memory_size, layers=memory_layers)
    config["amago.nets.traj_encoders.TformerTrajEncoder.attention"] = (
        "flash" if attention == "torch-sdpa" else attention)
    for key, value in config.items():
        gin.bind_parameter(key, value)
    # AMAGO's timestep encoder reads the conventional observation key. The
    # observation is 39D state + two normalized public clocks. Its standard
    # rl2 vector adds boundary, previous scaled reward, trial clock and action.
    policy = amago.Agent(
        obs_space=gym.spaces.Dict({"observation": gym.spaces.Box(
            -np.inf, np.inf, (41,), dtype=np.float32)}),
        goal_space=gym.spaces.Box(0., 0., (1, 1), dtype=np.float32),
        rl2_space=gym.spaces.Box(-np.inf, np.inf, (7,), dtype=np.float32),
        action_space=gym.spaces.Box(-1., 1., (4,), dtype=np.float32),
        max_seq_len=max_seq_len, horizon=horizon)
    if attention == "torch-sdpa":
        install_torch_sdpa(policy)
    policy.to(device)
    # We invoke upstream compute_loss/train_step/init_optimizer directly while
    # replacing its obsolete environment driver and incomplete checkpoints.
    experiment = amago.Experiment(make_train_env=None, make_val_env=None,
        parallel_actors=1, max_seq_len=max_seq_len, traj_save_len=horizon + 1,
        run_name="official_amago1", gpu=0 if str(device).startswith("cuda") else -1,
        batch_size=batch_size, train_grad_updates_per_epoch=grad_updates,
        warmup_epochs=warmup_epochs, half_precision=False, fast_inference=attention in CACHED_ATTENTION)
    experiment.DEVICE = torch.device(device)
    experiment.policy = policy
    experiment.init_optimizer()
    if warmup_grad_updates is not None:
        if type(warmup_grad_updates) is not int or warmup_grad_updates < 1:
            raise ValueError("Require positive explicit optimizer-step warmup")
        # Keep upstream LinearLR and its initialization; decouple only the
        # explicit optimizer-step duration from a changed collection cadence.
        experiment.warmup_scheduler.total_iters = warmup_grad_updates
    return experiment


def round_grad_updates(iteration, environment_steps, *, fixed=1000, ratio=None):
    """Round cumulative rational UTD targets, preserving the ratio on resume."""
    if ratio is None:
        return fixed
    ratio = Fraction(ratio)
    if ratio <= 0 or environment_steps * ratio < 1:
        raise ValueError("UTD must allocate at least one optimizer update per round")
    before = iteration * environment_steps * ratio
    after = (iteration + 1) * environment_steps * ratio
    return after.numerator // after.denominator - before.numerator // before.denominator


def sampled_transition_count(batch):
    """Count replay transition presentations once, independent of gamma/critic heads."""
    from amago.loading import MAGIC_PAD_VAL
    valid_states = ~(batch.rl2s == MAGIC_PAD_VAL).all(-1)
    return int((valid_states[:, :-1] & valid_states[:, 1:]).sum())


def make_timestep(observation, *, time_idx, length, attempts, horizon, terminal=False,
                  reward_blind=False):
    """Preserve terminal feedback when physical episodes reset within a trial."""
    from amago.hindsight import GoalSeq, Timestep
    clock = np.asarray(observation["clock"], np.float32) / np.array(
        [max(attempts - 1, 1), horizon], np.float32)
    return Timestep(
        obs={"observation": np.concatenate((observation["state"], clock)).astype(np.float32)},
        prev_action=np.asarray(observation["previous_action"], np.float32).copy(),
        achieved_goal=[np.ones(1, np.float32)],
        goal_seq=GoalSeq(seq=[np.zeros(1, np.float32)], active_idx=0),
        time=time_idx / length,
        reset=bool(time_idx == 0 or observation["attempt_end"][0]),
        real_reward=0. if reward_blind else float(observation["previous_reward"][0]),
        raw_time_idx=time_idx, terminal=terminal)


class ExplorationClock(gym.Env):
    """Expose the official exploration wrapper's public trial clock."""

    def __init__(self, env):
        # A worker lane proxy intentionally is not a Gym Env. This small Env
        # view lets the unchanged official Gym ActionWrapper consume either.
        self.env = env
        self.action_space, self.observation_space = env.action_space, env.observation_space
        self.horizon = env.attempts * env.horizon
        self.step_count = 0

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def step(self, action):
        return self.env.step(action)

    def close(self):
        self.env.close()


def make_explorers(envs, *, seed=0):
    from amago.envs.env_utils import ExplorationWrapper
    # The official Meta-World example changes only this exploration anneal.
    explorers = [ExplorationWrapper(ExplorationClock(env), steps_anneal=2_000_000)
                 for env in envs]
    numpy_rng = np.random.RandomState(seed)
    for explorer in explorers:
        explorer._fluffy_numpy_rng = numpy_rng
    return explorers


@contextmanager
def exploration_random_stream(explorers):
    """Run unchanged upstream np.random calls on a checkpointed parent stream."""
    stream = explorers[0]._fluffy_numpy_rng
    if any(explorer._fluffy_numpy_rng is not stream for explorer in explorers):
        raise ValueError("Ordered exploration lanes must share one parent NumPy stream")
    parent = np.random.get_state()
    np.random.set_state(stream.get_state())
    try:
        yield
    finally:
        stream.set_state(np.random.get_state())
        np.random.set_state(parent)


@torch.no_grad()
def collect_trials(policy, envs, seeds, *, max_seq_len, attention="flash",
                   explorers=None, deterministic=False, condition="normal",
                   clock_attempts=None, batch_step=None):
    """Synchronous batched inference; every lane completes one whole trial."""
    from amago.hindsight import Trajectory
    if condition not in ("normal", "memory_reset", "reward_blind"):
        raise ValueError("Unknown history intervention")
    if not envs:
        raise ValueError("Require at least one trial lane")
    device = next(policy.parameters()).device
    length = envs[0].attempts * envs[0].horizon
    if len(seeds) != len(envs) or any(e.attempts * e.horizon != length for e in envs):
        raise ValueError("Collect equal length seeded trials")
    clock_attempts = envs[0].attempts if clock_attempts is None else clock_attempts
    if type(clock_attempts) is not int or clock_attempts < 1:
        raise ValueError("Require a positive training clock attempt count")
    clock_length = clock_attempts * envs[0].horizon
    policy.eval()
    trajectories = []
    resets = batch_reset(envs, seeds)
    for index, (env, reset) in enumerate(zip(envs, resets)):
        observation, _ = reset
        if explorers is not None:
            explorers[index].global_multiplier = random.random()
        trajectories.append(Trajectory(max_goals=1, timesteps=[make_timestep(
            observation, time_idx=0, length=clock_length, attempts=clock_attempts,
            horizon=env.horizon, reward_blind=condition == "reward_blind")]))
    hidden = policy.traj_encoder.init_hidden_state(len(envs), device) if attention in CACHED_ATTENTION else None
    last_reset = 0
    for step in range(length):
        # Cache mode consumes exactly one new timestep. Vanilla mode reconstructs
        # the last max_seq_len observations, matching upstream slow inference.
        sequences = [t.make_sequence(last_only=hidden is not None) for t in trajectories]
        start = max(last_reset, step + 1 - max_seq_len)
        obs = {"observation": torch.as_tensor(np.stack([
            s[0]["observation"] if hidden is not None else s[0]["observation"][start:]
            for s in sequences]), device=device)}
        goals = torch.as_tensor(np.stack([s[1] if hidden is not None else s[1][start:]
                                         for s in sequences]), device=device)
        rl2s = torch.as_tensor(np.stack([s[2] if hidden is not None else s[2][start:]
                                        for s in sequences]), device=device)
        times = torch.arange(step if hidden is not None else start, step + 1,
                             device=device).expand(len(envs), -1)
        seq_lengths = torch.full((len(envs), 1, 1), obs["observation"].shape[1],
                                 dtype=torch.long, device=device)
        actions, hidden = policy.get_actions(obs=obs, goals=goals, rl2s=rl2s,
            seq_lengths=seq_lengths, time_idxs=times, hidden_state=hidden,
            sample=not deterministic)
        executed_actions = actions.copy()
        if explorers is not None:
            with exploration_random_stream(explorers):
                for index, action in enumerate(actions):
                    explorers[index].env.step_count = step
                    executed_actions[index] = explorers[index].action(action)
        # Optional simulation-only seam. Inference, exploration order, previous
        # executed-action feedback and trial/replay semantics stay in this driver.
        results = (batch_step(executed_actions) if batch_step is not None else
                   step_lanes(envs, executed_actions))
        if len(results) != len(envs):
            raise ValueError("Batch step must return one five-tuple per ordered lane")
        for index, (env, result) in enumerate(zip(envs, results)):
            observation, reward, terminated, truncated, _ = result
            if bool(terminated or truncated) != (step == length - 1):
                raise RuntimeError("AMAGO trial terminated outside its declared horizon")
            if not np.isclose(reward, observation["previous_reward"][0]):
                raise RuntimeError("Previous reward does not match replay target")
            trajectories[index].add_timestep(make_timestep(observation,
                time_idx=step + 1, length=clock_length, attempts=clock_attempts,
                horizon=env.horizon, terminal=terminated or truncated,
                reward_blind=condition == "reward_blind"))
        if condition == "memory_reset" and (step + 1) % envs[0].horizon == 0:
            last_reset = step + 1
            if hidden is not None:
                hidden = policy.traj_encoder.init_hidden_state(len(envs), device)
    return [trajectory.freeze() for trajectory in trajectories]


def summarize(envs):
    outcomes = np.asarray([env.outcomes for env in envs], dtype=bool)
    families = sorted({env.family for env in envs})
    per_family = {family: outcomes[[e.family == family for e in envs]].mean(0).tolist()
                  for family in families}
    curve = np.asarray(list(per_family.values())).mean(0)
    return dict(success=float(curve.mean()), success_by_episode=curve.tolist(),
        per_family=per_family, outcomes=outcomes.tolist(),
        native_returns=[e.native_returns for e in envs],
        family_by_lane=[e.family for e in envs], trials=len(envs),
        environment_steps=sum(e.attempts * e.horizon for e in envs))


class ReplayStore:
    """Immutable content addressed trials; checkpoints own the sampling index.

    Files excluded by FIFO retention remain on disk because an older checkpoint
    may reference them. Garbage collection is a separate offline operation.
    """

    def __init__(self, directory, capacity):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.capacity = capacity
        self.entries = []

    def append(self, trajectory):
        from amago.hindsight import FrozenTraj
        if not isinstance(trajectory, FrozenTraj):
            raise TypeError("Replay requires an official FrozenTraj")
        temporary = self.directory / "incoming.partial.npz"
        with temporary.open("xb") as stream:
            np.savez(stream, **trajectory.to_dict())
            stream.flush()
            os.fsync(stream.fileno())
        digest = hashlib.sha256(temporary.read_bytes()).hexdigest()
        path = self.directory / f"{digest}.npz"
        temporary.replace(path)
        self.entries.append(dict(path=str(path.resolve()), sha256=digest))
        self.entries = self.entries[-self.capacity:]

    def restore(self, entries):
        if len(entries) > self.capacity:
            raise ValueError("Replay checkpoint exceeds the configured capacity")
        for entry in entries:
            if hashlib.sha256(Path(entry["path"]).read_bytes()).hexdigest() != entry["sha256"]:
                raise ValueError("Replay bytes changed since the checkpoint")
        self.entries = list(entries)

    def batch(self, size, max_seq_len):
        from amago.loading import RLData, RLData_pad_collate, load_traj_from_disk
        # Preserve upstream uniform-trajectory sampling and random_slice rules.
        return RLData_pad_collate([RLData(load_traj_from_disk(
            random.choice(self.entries)["path"])).random_slice(max_seq_len) for _ in range(size)])


def save_snapshot(path, *, experiment, replay, sampler, explorers, iteration, counters, contract):
    path = Path(path)
    temporary = path.with_suffix(".partial")
    payload = dict(version=1, boundary="between_complete_trials_and_updates",
        contract=contract, iteration=iteration, counters=dict(counters),
        policy=experiment.policy.state_dict(), optimizer=experiment.optimizer.state_dict(),
        scheduler=experiment.warmup_scheduler.state_dict(), scaler=experiment.grad_scaler.state_dict(),
        replay=replay.entries, sampler=sampler.bit_generator.state, rng=rng_state(),
        exploration_steps=[e.global_step for e in explorers],
        exploration_numpy_rng=explorers[0]._fluffy_numpy_rng.get_state())
    with temporary.open("xb") as stream:
        torch.save(payload, stream)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def restore_snapshot(path, *, experiment, replay, sampler, explorers, contract):
    saved = torch.load(path, map_location="cpu", weights_only=False)
    if (saved.get("version") != 1 or saved.get("contract") != contract
            or saved.get("boundary") != "between_complete_trials_and_updates"
            or len(saved["exploration_steps"]) != len(explorers)):
        raise ValueError("AMAGO checkpoint source or scientific contract differs")
    replay.restore(saved["replay"])
    experiment.policy.load_state_dict(saved["policy"], strict=True)
    experiment.optimizer.load_state_dict(saved["optimizer"])
    experiment.warmup_scheduler.load_state_dict(saved["scheduler"])
    experiment.grad_scaler.load_state_dict(saved["scaler"])
    sampler.bit_generator.state = saved["sampler"]
    for explorer, steps in zip(explorers, saved["exploration_steps"]):
        explorer.global_step = steps
        explorer._fluffy_numpy_rng.set_state(saved["exploration_numpy_rng"])
    restore_rng(saved["rng"])
    return saved["iteration"], dict(saved["counters"])


def parser():
    p = argparse.ArgumentParser(description=__doc__, fromfile_prefix_chars="@")
    p.add_argument("--source-commit", required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--seed", type=int, default=1000)
    p.add_argument("--benchmark-seed", type=int, default=0)
    p.add_argument("--num-envs", type=int, default=10)
    p.add_argument("--attempts", type=int, default=5)
    p.add_argument("--horizon", type=int, default=500)
    p.add_argument("--updates", type=int, default=1000)
    p.add_argument("--grad-updates", type=int, default=1000)
    schedule = p.add_mutually_exclusive_group()
    schedule.add_argument("--optimizer-updates-per-transition", type=Fraction,
        help="Optional exact rational UTD, e.g.1/12; overrides fixed --grad-updates per round")
    schedule.add_argument("--replay-epochs", type=int,
        help="Optional sampled replay transition passes per new collection; changes original AMAGO schedule")
    p.add_argument("--batch-size", type=int, default=24)
    p.add_argument("--max-seq-len", type=int, default=2000)
    p.add_argument("--memory-size", type=int, default=256)
    p.add_argument("--memory-layers", type=int, default=3)
    p.add_argument("--replay-capacity", type=int, default=20_000)
    p.add_argument("--warmup-epochs", type=int, default=10)
    p.add_argument("--warmup-grad-updates", type=int,
        help="Explicit optimizer-step warmup; overrides warmup-epochs times fixed grad-updates")
    p.add_argument("--eval-every", type=int, default=20)
    p.add_argument("--eval-attempts", type=int, default=5)
    p.add_argument("--eval-lanes", type=int, default=10)
    p.add_argument("--attention", choices=("flash", "torch-sdpa", "vanilla"), default="flash")
    p.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--env-workers", type=int, default=0,
                   help="Zero: serial simulator reference; positive: CPU worker groups")
    p.add_argument("--allocated-cpus", type=int, default=16)
    p.add_argument("--max-elapsed-seconds", type=float, default=3600.)
    p.add_argument("--resume", type=Path)
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    evidence = allocation()
    source = clean_source(args.source_commit)
    positive = (args.num_envs, args.attempts, args.horizon, args.updates, args.grad_updates,
                args.batch_size, args.max_seq_len, args.memory_size, args.memory_layers,
                args.replay_capacity, args.warmup_epochs, args.eval_every, args.eval_lanes,
                args.threads, args.allocated_cpus)
    if (any(v < 1 for v in positive) or args.horizon > 500 or args.num_envs % 10
            or args.eval_lanes % 10 or args.memory_size % 64 or args.memory_size > 1024
            or args.env_workers < 0
            or args.warmup_grad_updates is not None and args.warmup_grad_updates < 1
            or args.eval_attempts < 1 or args.replay_epochs is not None and args.replay_epochs < 1
            or args.optimizer_updates_per_transition is not None and (
                args.optimizer_updates_per_transition <= 0 or
                args.num_envs * args.attempts * args.horizon * args.optimizer_updates_per_transition < 1)
            or not 0 <= args.seed < 1_000_000_000 or not 0 <= args.benchmark_seed < 1_000_000
            or not np.isfinite(args.max_elapsed_seconds) or args.max_elapsed_seconds <= 0
            or args.attention == "flash" and args.device != "cuda"):
        raise ValueError("Invalid AMAGO ML10 execution contract")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    amago = load_upstream()
    if args.attention == "flash":
        import flash_attn  # Fail early, before generating expensive simulator tasks.
        if not hasattr(flash_attn, "flash_attn_with_kvcache"):
            raise RuntimeError("Require FlashAttention2 with KV cache support")
    pools = benchmark_pools("ml10", args.benchmark_seed)
    factory = make_worker_lanes if args.env_workers else make_lanes
    worker_options = (dict(workers=args.env_workers, allocated_cpus=args.allocated_cpus)
                      if args.env_workers else {})
    envs = factory(pools["train"], args.num_envs, benchmark="ml10",
                   attempts=args.attempts, horizon=args.horizon, **worker_options)
    new_steps_per_round = args.num_envs * args.attempts * args.horizon
    replay_steps_per_update = args.batch_size * min(args.max_seq_len, args.attempts * args.horizon)
    update_ratio = (Fraction(args.replay_epochs, replay_steps_per_update)
                    if args.replay_epochs is not None else args.optimizer_updates_per_transition)
    if update_ratio is not None and new_steps_per_round * update_ratio < 1:
        raise ValueError("Configured replay exposure must yield at least one update per complete round")
    warmup_grad_updates = args.warmup_grad_updates
    if args.replay_epochs is not None and warmup_grad_updates is None:
        warmup_target = args.warmup_epochs * new_steps_per_round * update_ratio
        warmup_grad_updates = max(1, warmup_target.numerator // warmup_target.denominator)
    experiment = make_experiment(amago, horizon=args.attempts * args.horizon,
        max_seq_len=args.max_seq_len, memory_size=args.memory_size,
        memory_layers=args.memory_layers, attention=args.attention,
        batch_size=args.batch_size, grad_updates=args.grad_updates,
        warmup_epochs=args.warmup_epochs, warmup_grad_updates=warmup_grad_updates,
        device=args.device)
    explorers = make_explorers(envs, seed=args.seed)
    sampler = np.random.default_rng(args.seed)
    replay = ReplayStore(args.output_dir / "replay", args.replay_capacity)
    versions = {name: importlib.metadata.version(name) for name in
                ("torch", "numpy", "gymnasium", "mujoco", "metaworld", "gin-config", "einops", "wandb")}
    if args.attention == "flash":
        versions["flash-attn"] = importlib.metadata.version("flash-attn")
    contract = dict(protocol=(PROTOCOL if args.replay_epochs is None else
        "metaworld_official_amago1_v3_matched_sampled_replay_v1"),
        source_commit=source, upstream_commit=UPSTREAM_COMMIT,
        benchmark="ml10", seed=args.seed, benchmark_seed=args.benchmark_seed,
        pool_hashes={k: task_digest(v[1]) for k, v in pools.items()},
        core="official_amago1", attempts=args.attempts, horizon=args.horizon,
        num_envs=args.num_envs, goal_visible=True, resample_instance=True,
        observation="39D state + normalized public clocks; official boundary/reward/time/action rl2 vector",
        reward="native dense * 0.1 feedback/replay; upstream Agent reward_multiplier=10 for TD targets",
        normalization="official FFTstepEncoder InputNorm updated during gradient steps; PopArt",
        architecture=dict(memory_size=args.memory_size, memory_layers=args.memory_layers,
                          max_seq_len=args.max_seq_len, attention=args.attention, heads=8,
                          sdpa_dispatch=({name: getattr(torch.backends.cuda, name)()
                              for name in ("flash_sdp_enabled", "mem_efficient_sdp_enabled",
                                           "math_sdp_enabled", "cudnn_sdp_enabled")
                              if hasattr(torch.backends.cuda, name)}
                              if args.attention == "torch-sdpa" else None)),
        optimization=dict(learner="unmodified upstream Agent + Experiment.train_step",
            grad_updates=args.grad_updates, batch_size=args.batch_size,
            optimizer_updates_per_transition=(str(update_ratio)
                if update_ratio is not None else
                str(Fraction(args.grad_updates, args.num_envs * args.attempts * args.horizon))),
            gradient_update_schedule=("cumulative exact rational target floored at each complete round"
                if update_ratio is not None else "fixed per complete round"),
            replay_epochs=args.replay_epochs,
            sampled_replay_target_per_round=(new_steps_per_round * args.replay_epochs
                                            if args.replay_epochs is not None else None),
            new_environment_steps_per_round=new_steps_per_round,
            replay_transition_presentations_per_optimizer_step=replay_steps_per_update,
            optimizer="upstream AdamW lr=1e-4 weight_decay=1e-3", critic_loss_weight=10.,
            warmup_epochs=args.warmup_epochs,
            warmup_grad_updates=experiment.warmup_scheduler.total_iters,
            gradient_clip=1., reward_multiplier=10.,
            multigamma=True, num_critics=4, popart=True, gamma=.999,
            online_coeff=1., offline_coeff=.1, tau=.003, half_precision=False),
        replay=dict(capacity_trials=args.replay_capacity, sampling="uniform trial then uniform valid window",
                    checkpoint="immutable hashed NPZ files plus active FIFO index"),
        exploration=("official ExplorationWrapper, steps_anneal=2000000 per lane; "
                     "parent-owned checkpointed NumPy Gaussian stream initialized from run seed"),
        evaluation=dict(attempts=args.eval_attempts, lanes=args.eval_lanes,
            seed_start=1_000_000_000, action_selection="deterministic", eval_every=args.eval_every,
            clock_encoding_attempts=args.attempts,
            trial_time_divisor=args.attempts * args.horizon,
            positional_extension="evaluation-only lookup clamp above training horizon; raw absolute indices retained",
            success="any success within episode; average episodes and families equally"),
        attention_adapter=("parameter-free native PyTorch SDPA with upstream BF16 QKV/cache; "
                           "original projection/position/head scaling/blocks/Agent unchanged"
                           if args.attention == "torch-sdpa" else "none"),
        device=args.device, dtype="float32; flash/torch-sdpa attention internal bfloat16", versions=versions)
    counters = dict(environment_steps=0, optimizer_steps=0, evaluation_steps=0,
                    sampled_replay_transitions=0, target_sampled_replay_transitions=0,
                    collection_seconds=0., optimization_seconds=0., training_seconds=0.,
                    evaluation_seconds=0., cumulative_elapsed_seconds=0.)
    iteration = 0
    if args.resume:
        iteration, counters = restore_snapshot(args.resume, experiment=experiment,
            replay=replay, sampler=sampler, explorers=explorers, contract=contract)
    if iteration >= args.updates:
        raise ValueError("Target updates must exceed the restored iteration")
    parent_elapsed = counters["cumulative_elapsed_seconds"]
    parameters = sum(p.numel() for p in experiment.policy.parameters())
    trainable = sum(p.numel() for p in experiment.policy.trainable_params if p.requires_grad)
    write_json(args.output_dir / "config.json", dict(contract=contract, allocation=evidence,
        parameters=parameters, trainable_parameters=trainable,
        execution=dict(env_workers=args.env_workers, allocated_cpus=args.allocated_cpus,
                       worker_backend=envs[0]._worker_batch.manifest if args.env_workers else "serial"),
        target_updates=args.updates, max_elapsed_seconds=args.max_elapsed_seconds,
        resume=str(args.resume) if args.resume else None,
        departures=["V3 goal-visible resampled-instance common benchmark adapter",
                    "complete-trial collection instead of 1000-step partial epochs",
                    "FIFO replay index; archived bytes retained for checkpoint recovery",
                    "source/replay/scheduler/sampler/RNG complete checkpoints",
                    "original 2000-step context is shorter than K5H500 (2500 steps)"]
                   + (["matched sampled replay passes replace original AMAGO optimizer/data schedule"]
                      if args.replay_epochs is not None else []),
        accounting="one allocated GPU; all active wall time charged, including CPU simulation/evaluation"))
    import gin
    (args.output_dir / "operative.gin").write_text(
        f"# Runtime primitive backend: {args.attention}; see config.json attention_adapter.\n"
        + gin.operative_config_str())
    stop = False
    def request_stop(signum, frame):
        nonlocal stop
        stop = True
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, request_stop)

    def snapshot():
        synchronize(args.device)
        counters["cumulative_elapsed_seconds"] = parent_elapsed + time.monotonic() - started
        return save_snapshot(args.output_dir / "latest.pt", experiment=experiment,
            replay=replay, sampler=sampler, explorers=explorers, iteration=iteration,
            counters=counters, contract=contract)

    def evaluation(conditions=("normal",)):
        before = time.monotonic()
        result = {}
        with preserve_random_streams(), extend_evaluation_positions(experiment.policy):
            for split in ("train_eval", "test"):
                lanes = factory(pools[split], args.eval_lanes, benchmark="ml10",
                                attempts=args.eval_attempts, horizon=args.horizon, **worker_options)
                try:
                    result[split] = {}
                    for condition in conditions:
                        collect_trials(experiment.policy, lanes,
                            range(1_000_000_000, 1_000_000_000 + len(lanes)),
                            max_seq_len=args.max_seq_len, attention=args.attention,
                            deterministic=True, condition=condition, clock_attempts=args.attempts)
                        result[split][condition] = summarize(lanes)
                finally:
                    for lane in lanes:
                        lane.close()
        synchronize(args.device)
        counters["evaluation_seconds"] += time.monotonic() - before
        counters["evaluation_steps"] += sum(v["environment_steps"] for split in result.values()
                                             for v in split.values())
        return result

    def record(log, **values):
        synchronize(args.device)
        elapsed = parent_elapsed + time.monotonic() - started
        row = dict(iteration=iteration, environment_steps=counters["environment_steps"],
            optimizer_steps=counters["optimizer_steps"], elapsed_seconds=elapsed,
            sampled_replay_transitions=counters["sampled_replay_transitions"],
            target_sampled_replay_transitions=counters["target_sampled_replay_transitions"],
            sampled_replay_passes=(counters["sampled_replay_transitions"] / counters["environment_steps"]
                                  if counters["environment_steps"] else None),
            gpu_hours=elapsed / 3600 if args.device == "cuda" else None,
            training_gpu_hours=counters["training_seconds"] / 3600 if args.device == "cuda" else None,
            **values)
        log.write(json.dumps(row, allow_nan=False) + "\n")
        compact = {key: value for key, value in row.items() if key not in ("evaluation", "rollout")}
        if "evaluation" in row:
            compact["success"] = {k: {c: r["success"] for c, r in v.items()}
                                  for k, v in row["evaluation"].items()}
        print(json.dumps(compact, allow_nan=False), flush=True)

    last_evaluation = None
    try:
        snapshot()
        with (args.output_dir / "metrics.jsonl").open("x", buffering=1) as log:
            last_evaluation = evaluation()
            record(log, evaluation=last_evaluation)
            while iteration < args.updates and not stop and time.monotonic() - started < args.max_elapsed_seconds:
                before = time.monotonic()
                trajectories = collect_trials(experiment.policy, envs,
                    sampler.integers(0, 1_000_000_000, size=len(envs)),
                    max_seq_len=args.max_seq_len, attention=args.attention, explorers=explorers)
                for trajectory in trajectories:
                    replay.append(trajectory)
                synchronize(args.device)
                collection = time.monotonic() - before
                optimization_start = time.monotonic()
                experiment.policy.train()
                grad_updates = round_grad_updates(iteration, len(envs) * args.attempts * args.horizon,
                    fixed=args.grad_updates, ratio=update_ratio)
                sampled_this_round = 0
                for _ in range(grad_updates):
                    batch = replay.batch(args.batch_size, args.max_seq_len)
                    sampled = sampled_transition_count(batch)
                    if sampled != replay_steps_per_update:
                        raise ValueError("Replay mask differs from the configured transition exposure")
                    # These are the exact upstream loss reductions, backward,
                    # clipping, optimizer and target-network update operations.
                    losses = experiment.train_step(batch, log_step=False)
                    if not all(bool(torch.isfinite(losses[key]).all()) for key in
                               ("actor_loss", "critic_loss", "Global Grad Norm")):
                        raise FloatingPointError("Nonfinite official AMAGO loss or gradient")
                    experiment.warmup_scheduler.step(counters["optimizer_steps"])
                    counters["optimizer_steps"] += 1
                    sampled_this_round += sampled
                if not all(bool(torch.isfinite(value).all()) for value in
                           experiment.policy.state_dict().values()):
                    raise FloatingPointError("Nonfinite official AMAGO parameters or buffers")
                synchronize(args.device)
                optimization = time.monotonic() - optimization_start
                counters["collection_seconds"] += collection
                counters["optimization_seconds"] += optimization
                counters["training_seconds"] += time.monotonic() - before
                counters["environment_steps"] += len(envs) * args.attempts * args.horizon
                counters["sampled_replay_transitions"] += sampled_this_round
                if args.replay_epochs is not None:
                    counters["target_sampled_replay_transitions"] += new_steps_per_round * args.replay_epochs
                iteration += 1
                snapshot()
                values = dict(training={k: float(losses[k].detach().cpu())
                                        for k in ("actor_loss", "critic_loss", "Global Grad Norm")},
                    rollout=summarize(envs), rollout_success=summarize(envs)["success"],
                    timing=dict(rollout_seconds=collection, update_seconds=optimization),
                    replay_trials=len(replay.entries), optimizer_steps_this_round=grad_updates)
                if iteration % args.eval_every == 0 or iteration == args.updates:
                    last_evaluation = evaluation()
                    values["evaluation"] = last_evaluation
                record(log, **values)
            last_evaluation = evaluation(("normal", "memory_reset", "reward_blind"))
            record(log, final=True, evaluation=last_evaluation)
        digest = snapshot()
        write_json(args.output_dir / "summary.json", dict(completed=iteration == args.updates,
            iteration=iteration, source_commit=source, counters=counters,
            checkpoint_sha256=digest, last_evaluation=last_evaluation,
            cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated() if args.device == "cuda" else 0))
    except BaseException as exc:
        write_json(args.output_dir / "failure.json", dict(error=type(exc).__name__, message=str(exc),
            last_complete_iteration=iteration, checkpoint=str(args.output_dir / "latest.pt"),
            attempted_elapsed_seconds=parent_elapsed + time.monotonic() - started,
            note="Failed work is charged allocation time but is absent from the complete checkpoint"))
        raise
    finally:
        for env in envs:
            env.close()


if __name__ == "__main__":
    main()
