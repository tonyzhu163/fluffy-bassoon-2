"""Two-stage CPU learner for the simulated cross-arm pilot. See LEARNER.md."""

import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import NamedTuple

import numpy as np
import torch
from torch import nn


class _Episode(NamedTuple):
    episode_id: str
    commands: np.ndarray  # Issued controls, or frozen proposals for a joint query.
    target: np.ndarray | None  # Measured motion, or calibrated action labels.


def _rows(values, field, *, bounded=False, allow_empty=False):
    raw = np.asarray(values)
    if raw.ndim != 2 or raw.shape[1] != 7 or (not allow_empty and len(raw) == 0):
        raise ValueError(f"{field} must have shape [T,7], got {raw.shape}")
    if raw.dtype.kind not in "fiu" or not np.isfinite(raw).all():
        raise ValueError(f"{field} must contain finite real values")
    if bounded and ((raw < -1).any() or (raw > 1).any()):
        raise ValueError(f"{field} outside native [-1,1] bounds")
    result = np.ascontiguousarray(raw, dtype=np.float32)
    if not np.isfinite(result).all():
        raise ValueError(f"{field} exceeds float32 range")
    return result


def _proposal_rows(proposal):
    proposal = np.asarray(proposal)
    single = proposal.ndim == 1
    return _rows(proposal[None] if single else proposal, "proposal", bounded=True), single


def _episodes(records, role, *, proposals=False):
    by_body = {}
    seen = set()
    for record in records:
        body, uid = record["body"], record["episode_id"]
        if not all(isinstance(s, str) and s.strip() for s in (body, uid)):
            raise ValueError(f"{role} requires nonempty string body and episode_id")
        if (body, uid) in seen:
            raise ValueError(f"{role} repeats {(body, uid)}")
        seen.add((body, uid))
        key = "proposals" if proposals else "commands"
        commands = _rows(record[key], f"{role}/{uid}/{key}", bounded=True)
        motion = None if proposals else _rows(record["motion"], f"{role}/{uid}/motion")
        if motion is not None and commands.shape != motion.shape:
            raise ValueError(f"{role}/{uid} must pair every issued command with one motion")
        by_body.setdefault(body, []).append(_Episode(uid, commands, motion))
    return by_body


def _check_splits(pools, bodies):
    prior = {}
    for role, pool in pools.items():
        if set(pool) != set(bodies):
            raise ValueError(f"{role} has {sorted(pool)}, expected exactly {sorted(bodies)}")
        ids = {ep.episode_id for episodes in pool.values() for ep in episodes}
        for uid in ids:
            if uid in prior:
                raise ValueError(f"episode lineage {uid!r} shared by {prior[uid]} and {role}")
            prior[uid] = role


class _Response(NamedTuple):
    gain: np.ndarray  # Row-vector map from six OSC controls to metres/radians.
    drift: np.ndarray
    report: dict


def _fit_response(episodes):
    commands = np.concatenate([ep.commands[:, :6] for ep in episodes]).astype(np.float64)
    motion = np.concatenate([ep.target[:, :6] for ep in episodes]).astype(np.float64)
    design = np.column_stack((commands, np.ones(len(commands))))
    coef, _, rank, sv = np.linalg.lstsq(design, motion, rcond=None)
    if rank != 7:
        raise ValueError("Calibration needs independent excitation of all six Cartesian axes plus offset")
    gain, drift = coef[:6], coef[6]
    if np.linalg.matrix_rank(gain) != 6:
        raise ValueError("Observed Cartesian calibration response has a missing direction")
    return _Response(gain, drift, {
        "episode_ids": [ep.episode_id for ep in episodes],
        "rows": len(commands),
        "design_condition": float(sv[0] / sv[-1]),
        "cartesian_gain": gain.tolist(),
        "drift_m_rad": drift.tolist(),
        "fit_rmse_m_rad": np.sqrt(np.mean((design @ coef - motion) ** 2, axis=0)).tolist(),
    })


class _Teacher:
    def __init__(self, reference, target):
        # Normalize the six motion errors by the reference arm's per-axis response.
        # Among commands with similar motion prefer a small change to the proposal.
        scale = np.linalg.norm(reference.gain, axis=0)
        target_gain = target.gain / scale
        damping = 0.01
        inverse = np.linalg.solve(target_gain @ target_gain.T + damping * np.eye(6), target_gain).T
        self._delta_gain = ((reference.gain - target.gain) / scale) @ inverse
        self._delta_bias = ((reference.drift - target.drift) / scale) @ inverse
        self.diagnostics = {
            "reference": reference.report,
            "target": target.report,
            "reference_motion_scale_m_rad": scale.tolist(),
            "inverse_damping": damping,
        }

    def act(self, proposal):
        p, single = _proposal_rows(proposal)
        result = p.copy()
        arm = p[:, :6]
        result[:, :6] = np.clip(arm + arm @ self._delta_gain + self._delta_bias, -1, 1)
        return result[0] if single else result


def fit_teacher(reference_calibration_episodes, target_calibration_episodes):
    """Fit a privileged affine response per arm using separate calibration trajectories."""
    responses = []
    for name, records in (("reference", reference_calibration_episodes),
                          ("target", target_calibration_episodes)):
        pool = _episodes(records, name)
        if len(pool) != 1:
            raise ValueError(f"{name} calibration must contain exactly one body")
        responses.append(_fit_response(next(iter(pool.values()))))
    return _Teacher(*responses)


def _support(commands, motion, size):
    """Evenly cover the supplied completed practice, using a mask for short sets."""
    if commands.shape != motion.shape:
        raise ValueError("Support commands and motion must have matching [K,7] shapes")
    count = min(len(commands), size)
    packed = np.zeros((size, 15), dtype=np.float32)
    if count:
        indices = np.linspace(0, len(commands) - 1, count, dtype=int)
        packed[:count, :7] = commands[indices]
        packed[:count, 7:14] = motion[indices]
        packed[:count, 14] = 1
    return packed


class _Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("motion_scale", torch.ones(7, device="cpu"))
        self.reader = nn.Sequential(nn.Linear(63, 32, device="cpu"), nn.Tanh(),
                                    nn.Linear(32, 16, device="cpu"), nn.Tanh())
        # Each head generates a small context-dependent affine map of a new input.
        self.motion_head = nn.Linear(16, 8 * 7, device="cpu")
        self.action_head = nn.Linear(16, 8 * 6, device="cpu")
        nn.init.zeros_(self.action_head.weight)
        nn.init.zeros_(self.action_head.bias)

    def encode(self, support):
        commands = support[..., :7]
        motions = support[..., 7:14] / self.motion_scale
        valid = support[..., 14:]
        # Signed command/motion products expose direction and gain to a set reader.
        products = (commands.unsqueeze(-1) * motions.unsqueeze(-2)).flatten(-2)
        tokens = self.reader(torch.cat((commands, motions, products), dim=-1))
        return (tokens * valid).sum(1) / valid.sum(1).clamp_min(1)

    @staticmethod
    def _affine(head, context, commands):
        matrix = head(context).reshape(len(commands), 8, -1)
        return (commands.unsqueeze(-1) * matrix[:, :7]).sum(1) + matrix[:, 7]

    def predict_motion(self, context, commands):
        return self._affine(self.motion_head, context, commands)

    def adapt(self, context, proposals):
        residual = self._affine(self.action_head, context, proposals)
        return torch.cat((proposals[:, :6] + residual, proposals[:, 6:]), dim=1)


class _Pair(NamedTuple):
    support_id: str
    support: np.ndarray
    query: _Episode


def _pair_episodes(support, queries, size, rng, *, independent_commands):
    paired = {}
    for body, query_episodes in queries.items():
        practices = [(ep, _support(ep.commands, ep.target, size)) for ep in support[body]]
        pairs = []
        for query in query_episodes:
            eligible = [(ep, data) for ep, data in practices if ep.episode_id != query.episode_id
                        and not (independent_commands and np.array_equal(ep.commands, query.commands))]
            if not eligible:
                raise ValueError(f"{body}/{query.episode_id} needs an independent support episode"
                                 " with a different command trajectory")
            ep, data = eligible[rng.integers(len(eligible))]
            pairs.append(_Pair(ep.episode_id, data, query))
        paired[body] = pairs
    return paired


def _batch(paired, bodies, n, rng):
    """Sample bodies, then complete episode pairings, then a query control."""
    supports, commands, targets = [], [], []
    for body_index in rng.integers(len(bodies), size=n):
        candidates = paired[bodies[body_index]]
        support_pair = candidates[rng.integers(len(candidates))]
        query = support_pair.query
        t = rng.integers(len(query.commands))
        supports.append(support_pair.support)
        commands.append(query.commands[t])
        targets.append(query.target[t])
    return tuple(torch.from_numpy(np.stack(items)) for items in (supports, commands, targets))


def _fit_stage(network, paired, bodies, steps, batch_size, rng, *, stage):
    motion_stage = stage == "pretrain"
    head = network.motion_head if motion_stage else network.action_head
    params = list(network.reader.parameters()) + list(head.parameters())
    optimizer = torch.optim.Adam(params, lr=0.003)
    reader_before = [p.detach().clone() for p in network.reader.parameters()]
    head_before = [p.detach().clone() for p in head.parameters()]

    def loss_on(batch, *, cleared=False, training=False):
        history, x, y = batch
        z = network.encode(history)
        if cleared:
            z = torch.zeros_like(z)
        if motion_stage:
            pred = network.predict_motion(z, x)
            return (pred - y / network.motion_scale).square().mean()
        if training:
            # Train the same reader with occasional empty practice for the cleared control.
            keep = torch.from_numpy((rng.random(len(x)) >= 0.125).astype(np.float32))
            z = z * keep[:, None]
        pred = network.adapt(z, x)
        # Regress the pre-saturation arm command to the feasible teacher action.
        # This retains a recovery gradient even if a predicted command overshoots.
        if not training:
            pred = pred.clamp(-1, 1)
        return (pred[:, :6] - y[:, :6]).square().mean()

    audit = _batch(paired, bodies, batch_size, rng)
    with torch.no_grad():
        initial_loss = float(loss_on(audit))
        if not motion_stage and not torch.equal(network.adapt(network.encode(audit[0]), audit[1]), audit[1]):
            raise RuntimeError("Residual adapter failed the initial identity check")
    losses = []
    for step in range(1, steps + 1):
        optimizer.zero_grad(set_to_none=True)
        loss = loss_on(_batch(paired, bodies, batch_size, rng), training=True)
        loss.backward()
        nn.utils.clip_grad_norm_(params, 5.0, error_if_nonfinite=True)
        optimizer.step()
        if step in (1, steps) or step % max(1, steps // 5) == 0:
            losses.append({"step": step, "minibatch_loss": float(loss.detach())})
    with torch.no_grad():
        final_loss = float(loss_on(audit))
        cleared_loss = float(loss_on(audit, cleared=True))
        reader_change = sum((p - old).square().sum() for p, old in
                            zip(network.reader.parameters(), reader_before)).sqrt().item()
        head_change = sum((p - old).square().sum() for p, old in
                          zip(head.parameters(), head_before)).sqrt().item()
        if not np.isfinite([initial_loss, final_loss, cleared_loss, reader_change, head_change]).all():
            raise FloatingPointError(f"Nonfinite {stage} fit")
    return {
        "steps": steps, "curve": losses, "reader_change_l2": reader_change, "head_change_l2": head_change,
        "sampled_train_mse": {"initial": initial_loss, "retained": final_loss, "cleared": cleared_loss},
        "episode_pairs": {body: [{"support": p.support_id, "query": p.query.episode_id}
                                 for p in pairs] for body, pairs in paired.items()},
    }


def _positive_int(value, field):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"{field} must be a positive integer")
    return int(value)


def _new_network(seed):
    # Isolate initialization from the runner's RNG and from any allocated GPU.
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        return _Network().float()


class _Frozen:
    def __init__(self, network, support_size):
        self._network = network.requires_grad_(False).eval()
        self._support_size = support_size

    @torch.inference_mode()
    def act(self, proposal, support_commands=None, support_motion=None):
        p, single = _proposal_rows(proposal)
        if (support_commands is None) != (support_motion is None):
            raise ValueError("Supply both support_commands and support_motion, or omit both")
        if support_commands is None:
            commands = motion = np.empty((0, 7), dtype=np.float32)
        else:
            commands = _rows(support_commands, "support_commands", bounded=True, allow_empty=True)
            motion = _rows(support_motion, "support_motion", allow_empty=True)
        history = _support(commands, motion, self._support_size)
        context = self._network.encode(torch.from_numpy(history[None])).expand(len(p), -1)
        actions = self._network.adapt(context, torch.from_numpy(p)).clamp(-1, 1).numpy()
        if not np.isfinite(actions).all():
            raise FloatingPointError("Nonfinite action")
        return actions[0] if single else actions

    def save(self, path):
        path = Path(path)
        payload = {"format": "crossarm-v1", "support_size": self._support_size,
                   "weights": self._network.state_dict()}
        with NamedTemporaryFile(dir=path.parent, delete=False) as f:
            try:
                torch.save(payload, f)
                f.close()
                os.replace(f.name, path)
            finally:
                Path(f.name).unlink(missing_ok=True)


def load(path):
    """Load a CPU inference checkpoint with the training-only motion scale fixed."""
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload["format"] != "crossarm-v1":
        raise ValueError("Unknown cross-arm checkpoint format")
    size = _positive_int(payload["support_size"], "support_size")
    network = _new_network(0)
    network.load_state_dict(payload["weights"], strict=True)
    if not all(t.isfinite().all().item() for t in network.state_dict().values()):
        raise ValueError("Checkpoint contains nonfinite values")
    if (network.motion_scale <= 0).any():
        raise ValueError("Invalid motion scale in checkpoint")
    return _Frozen(network, size)


def fit(calibration, pretrain, joint_support, joint_queries, *,
        train_bodies=("Panda", "Sawyer"), reference_body="Panda", seed=0,
        pretrain_steps=300, joint_steps=400, batch_size=32, support_size=16):
    """Fit both stages on source episodes and return (frozen model, JSON-able evidence)."""
    bodies = tuple(train_bodies)
    if len(set(bodies)) != len(bodies) or len(bodies) < 2 or reference_body not in bodies:
        raise ValueError("Specify at least two distinct training bodies including the reference")
    pretrain_steps = _positive_int(pretrain_steps, "pretrain_steps")
    joint_steps = _positive_int(joint_steps, "joint_steps")
    if joint_steps < 2:
        raise ValueError("Joint fit needs a second update for gradients through the zero-init head")
    batch_size = _positive_int(batch_size, "batch_size")
    size = _positive_int(support_size, "support_size")
    rng = np.random.default_rng(seed)
    pools = {"calibration": _episodes(calibration, "calibration"),
             "pretrain": _episodes(pretrain, "pretrain"),
             "joint_support": _episodes(joint_support, "joint_support"),
             "joint_queries": _episodes(joint_queries, "joint_queries", proposals=True)}
    _check_splits(pools, bodies)
    responses = {body: _fit_response(pools["calibration"][body]) for body in bodies}
    teachers = {body: _Teacher(responses[reference_body], response) for body, response in responses.items()}
    labels = {body: [ep._replace(target=teachers[body].act(ep.commands)) for ep in episodes]
              for body, episodes in pools["joint_queries"].items()}
    practices = pools["pretrain"]
    motion_pairs = _pair_episodes(practices, practices, size, rng, independent_commands=True)
    action_pairs = _pair_episodes(pools["joint_support"], labels, size, rng, independent_commands=False)

    network = _new_network(int(seed))
    motions = np.concatenate([ep.target for body in bodies for ep in practices[body]]).astype(np.float64)
    scales = np.maximum(np.sqrt(np.mean(motions ** 2, axis=0)), np.finfo(np.float32).eps)
    network.motion_scale.copy_(torch.from_numpy(scales))
    report = {
        "seed": int(seed), "train_bodies": list(bodies), "reference_body": reference_body,
        "batch_size": batch_size, "support_size": size, "motion_rms_scale": scales.tolist(),
        "teachers": {body: teacher.diagnostics for body, teacher in teachers.items()},
        "pretrain": _fit_stage(network, motion_pairs, bodies, pretrain_steps, batch_size, rng, stage="pretrain"),
        "joint": _fit_stage(network, action_pairs, bodies, joint_steps, batch_size, rng, stage="joint"),
    }
    return _Frozen(network, size), report
