"""Decoupled causal history belief encoder mapping public rollouts to steering latents."""

from dataclasses import dataclass


@dataclass(frozen=True)
class HistoryEncoderConfig:
    """Configuration for packing control tokens and running the causal belief encoder."""

    controls_per_token: int = 5
    control_channels: int = 15
    max_support_controls: int = 80
    max_prefix_controls: int = 40
    query_dim: int = 2056
    width: int = 128
    layers: int = 2
    heads: int = 4
    mlp_ratio: int = 4
    num_conditions: int = 4
    latent_dim: int = 16

    def __post_init__(self):
        int_fields = (
            ("controls_per_token", self.controls_per_token),
            ("control_channels", self.control_channels),
            ("max_support_controls", self.max_support_controls),
            ("max_prefix_controls", self.max_prefix_controls),
            ("query_dim", self.query_dim),
            ("width", self.width),
            ("layers", self.layers),
            ("heads", self.heads),
            ("mlp_ratio", self.mlp_ratio),
            ("num_conditions", self.num_conditions),
            ("latent_dim", self.latent_dim),
        )
        for name, value in int_fields:
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer, got {value!r}")
        if self.max_support_controls % self.controls_per_token != 0:
            raise ValueError("max_support_controls must be divisible by controls_per_token")
        if self.max_prefix_controls % self.controls_per_token != 0:
            raise ValueError("max_prefix_controls must be divisible by controls_per_token")
        if self.width % self.heads != 0:
            raise ValueError("width must be divisible by heads")

    @property
    def support_token_count(self):
        return self.max_support_controls // self.controls_per_token

    @property
    def prefix_token_count(self):
        return self.max_prefix_controls // self.controls_per_token

    @property
    def token_input_dim(self):
        return self.controls_per_token * (self.control_channels + 1)


def pack_control_tokens(
    issued,
    positions,
    initial_position,
    gripper,
    rewards,
    waits,
    *,
    max_controls,
    controls_per_token=5,
    episode_boundary_at_start=True,
):
    """Pack public control-rate rollout streams into fixed-stride tokens.

    Each control step packs 16 channels: 7 issued actions, 3 end-effector
    displacements relative to ``initial_position``, 2 gripper positions,
    1 reward, 1 wait indicator, 1 episode-boundary flag, and 1 validity bit.
    """
    import numpy as np

    if type(max_controls) is not int or max_controls < 1:
        raise ValueError(f"max_controls must be a positive integer, got {max_controls!r}")
    if type(controls_per_token) is not int or controls_per_token < 1:
        raise ValueError(f"controls_per_token must be a positive integer, got {controls_per_token!r}")
    if max_controls % controls_per_token != 0:
        raise ValueError("max_controls must be divisible by controls_per_token")

    issued_arr = np.asarray(issued, dtype=np.float32)
    pos_arr = np.asarray(positions, dtype=np.float32)
    init_pos = np.asarray(initial_position, dtype=np.float32).reshape(3)
    grip_arr = np.asarray(gripper, dtype=np.float32)
    rew_arr = np.asarray(rewards, dtype=np.float32)
    wait_arr = np.asarray(waits, dtype=np.float32)

    lengths = {len(issued_arr), len(pos_arr), len(grip_arr), len(rew_arr), len(wait_arr)}
    if len(lengths) != 1:
        raise ValueError(f"Mismatched control-stream lengths: {sorted(lengths)}")
    raw_len = len(issued_arr)
    n = min(raw_len, max_controls)
    n_tokens = max_controls // controls_per_token
    grid = np.zeros((max_controls, 16), dtype=np.float32)
    if n > 0:
        if issued_arr.shape[1:] != (7,) or pos_arr.shape[1:] != (3,) or grip_arr.shape[1:] != (2,):
            raise ValueError("Expected shapes [T,7] issued, [T,3] positions, and [T,2] gripper")
        pos_seq = np.concatenate((init_pos[None], pos_arr[:n]), axis=0)
        delta_pos = np.diff(pos_seq, axis=0)
        grid[:n, 0:7] = issued_arr[:n]
        grid[:n, 7:10] = delta_pos
        grid[:n, 10:12] = grip_arr[:n]
        grid[:n, 12] = rew_arr[:n]
        grid[:n, 13] = wait_arr[:n]
        if episode_boundary_at_start:
            grid[0, 14] = 1.0
        grid[:n, 15] = 1.0
    tokens = grid.reshape(n_tokens, controls_per_token * 16)
    token_valid = (grid[:, 15].reshape(n_tokens, controls_per_token) > 0.5).any(axis=1)
    return tokens, token_valid


def _layer_norm(x, scale=None, bias=None, eps=1e-6):
    import jax
    import jax.numpy as jnp

    mean = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.mean(jnp.square(x - mean), axis=-1, keepdims=True)
    normed = (x - mean) * jax.lax.rsqrt(var + eps)
    if scale is not None:
        normed = normed * scale + bias
    return normed


def init_history_encoder(key, cfg):
    """Initialize causal transformer history belief encoder parameters."""
    import jax
    import jax.numpy as jnp

    init = jax.nn.initializers.lecun_normal()
    norm_init = jax.nn.initializers.normal(stddev=0.02)
    keys = jax.random.split(key, 8 + cfg.layers)
    k_ctrl, k_sseg, k_pseg, k_spos, k_ppos, k_q, k_qpos, k_logits = keys[:8]
    width = cfg.width
    ffn_width = cfg.mlp_ratio * width
    blocks = []
    for layer_key in keys[8:]:
        k_qkv, k_out, k_f1, k_f2 = jax.random.split(layer_key, 4)
        blocks.append(
            dict(
                ln1_scale=jnp.ones((width,), dtype=jnp.float32),
                ln1_bias=jnp.zeros((width,), dtype=jnp.float32),
                qkv_w=init(k_qkv, (width, 3 * width), dtype=jnp.float32),
                qkv_b=jnp.zeros((3 * width,), dtype=jnp.float32),
                attn_out_w=init(k_out, (width, width), dtype=jnp.float32),
                attn_out_b=jnp.zeros((width,), dtype=jnp.float32),
                ln2_scale=jnp.ones((width,), dtype=jnp.float32),
                ln2_bias=jnp.zeros((width,), dtype=jnp.float32),
                ffn_w1=init(k_f1, (width, ffn_width), dtype=jnp.float32),
                ffn_b1=jnp.zeros((ffn_width,), dtype=jnp.float32),
                ffn_w2=init(k_f2, (ffn_width, width), dtype=jnp.float32),
                ffn_b2=jnp.zeros((width,), dtype=jnp.float32),
            )
        )
    return dict(
        control_proj_w=init(k_ctrl, (cfg.token_input_dim, width), dtype=jnp.float32),
        control_proj_b=jnp.zeros((width,), dtype=jnp.float32),
        support_seg=norm_init(k_sseg, (1, 1, width), dtype=jnp.float32),
        prefix_seg=norm_init(k_pseg, (1, 1, width), dtype=jnp.float32),
        support_pos=norm_init(k_spos, (1, cfg.support_token_count, width), dtype=jnp.float32),
        prefix_pos=norm_init(k_ppos, (1, cfg.prefix_token_count, width), dtype=jnp.float32),
        query_proj_w=init(k_q, (cfg.query_dim, width), dtype=jnp.float32),
        query_proj_b=jnp.zeros((width,), dtype=jnp.float32),
        query_pos=norm_init(k_qpos, (1, 1, width), dtype=jnp.float32),
        blocks=blocks,
        final_ln_scale=jnp.ones((width,), dtype=jnp.float32),
        final_ln_bias=jnp.zeros((width,), dtype=jnp.float32),
        logits_w=init(k_logits, (width, cfg.num_conditions), dtype=jnp.float32),
        logits_b=jnp.zeros((cfg.num_conditions,), dtype=jnp.float32),
        latent_w=jnp.zeros((width, cfg.latent_dim), dtype=jnp.float32),
        latent_b=jnp.zeros((cfg.latent_dim,), dtype=jnp.float32),
    )


def apply_history_encoder(
    params,
    cfg,
    support_tokens,
    support_valid,
    prefix_tokens,
    prefix_valid,
    query_features,
    *,
    retain_support=True,
    retain_prefix=True,
):
    """Run causal self-attention over masked support/prefix tokens and query observation."""
    import jax
    import jax.numpy as jnp

    batch = query_features.shape[0]
    s_mask = jnp.asarray(support_valid, dtype=bool) if retain_support else jnp.zeros_like(support_valid, dtype=bool)
    p_mask = jnp.asarray(prefix_valid, dtype=bool) if retain_prefix else jnp.zeros_like(prefix_valid, dtype=bool)

    s_tok = jnp.where(s_mask[..., None], support_tokens, 0.0)
    p_tok = jnp.where(p_mask[..., None], prefix_tokens, 0.0)
    s_emb = s_tok @ params["control_proj_w"] + params["control_proj_b"] + params["support_seg"] + params["support_pos"]
    p_emb = p_tok @ params["control_proj_w"] + params["control_proj_b"] + params["prefix_seg"] + params["prefix_pos"]
    s_emb = jnp.where(s_mask[..., None], s_emb, 0.0)
    p_emb = jnp.where(p_mask[..., None], p_emb, 0.0)

    if cfg.query_dim > 8:
        q_in = jnp.concatenate((query_features[:, :8], _layer_norm(query_features[:, 8:])), axis=-1)
    else:
        q_in = query_features
    q_emb = (q_in @ params["query_proj_w"] + params["query_proj_b"])[:, None, :] + params["query_pos"]

    x = jnp.concatenate((s_emb, p_emb, q_emb), axis=1)
    valid = jnp.concatenate((s_mask, p_mask, jnp.ones((batch, 1), dtype=bool)), axis=1)
    x = jnp.where(valid[..., None], x, 0.0)

    seq_len = x.shape[1]
    head_dim = cfg.width // cfg.heads
    causal = jnp.tril(jnp.ones((seq_len, seq_len), dtype=bool))[None, None, :, :]
    eye = jnp.eye(seq_len, dtype=bool)[None, None, :, :]
    # Keep self-loop on masked rows so softmax never divides by zero during backward.
    attn_mask = (causal & valid[:, None, None, :]) | ((~valid)[:, None, :, None] & eye)

    for block in params["blocks"]:
        normed = _layer_norm(x, block["ln1_scale"], block["ln1_bias"])
        qkv = normed @ block["qkv_w"] + block["qkv_b"]
        q, k, v = jnp.split(qkv, 3, axis=-1)
        q = jnp.transpose(q.reshape(batch, seq_len, cfg.heads, head_dim), (0, 2, 1, 3))
        k = jnp.transpose(k.reshape(batch, seq_len, cfg.heads, head_dim), (0, 2, 1, 3))
        v = jnp.transpose(v.reshape(batch, seq_len, cfg.heads, head_dim), (0, 2, 1, 3))
        scores = jnp.einsum("bhqd,bhkd->bhqk", q, k) * (head_dim ** -0.5)
        scores = jnp.where(attn_mask, scores, -1e9)
        weights = jax.nn.softmax(scores, axis=-1)
        attended = jnp.einsum("bhqk,bhkd->bhqd", weights, v)
        attended = jnp.transpose(attended, (0, 2, 1, 3)).reshape(batch, seq_len, cfg.width)
        x = x + (attended @ block["attn_out_w"] + block["attn_out_b"])
        x = jnp.where(valid[..., None], x, 0.0)

        ffn_in = _layer_norm(x, block["ln2_scale"], block["ln2_bias"])
        ffn_hidden = jax.nn.silu(ffn_in @ block["ffn_w1"] + block["ffn_b1"])
        x = x + (ffn_hidden @ block["ffn_w2"] + block["ffn_b2"])
        x = jnp.where(valid[..., None], x, 0.0)

    h_q = _layer_norm(x[:, -1, :], params["final_ln_scale"], params["final_ln_bias"])
    logits = h_q @ params["logits_w"] + params["logits_b"]
    latent_delta = h_q @ params["latent_w"] + params["latent_b"]
    return logits, latent_delta


def decode_steering_latent(logits, latent_delta, condition_table, *, mode="prototype"):
    """Map predicted belief logits and delta into a steering latent vector [B, D_z]."""
    import jax
    import jax.numpy as jnp

    if mode == "prototype":
        return condition_table[jnp.argmax(logits, axis=-1)]
    if mode == "soft":
        return jax.nn.softmax(logits, axis=-1) @ condition_table + latent_delta
    raise ValueError(f"Unsupported decoding mode {mode!r}; expected 'prototype' or 'soft'")


def _belief_view_loss(params, cfg, batch, condition_table, *, retain_support, retain_prefix, latent_weight):
    import jax
    import jax.numpy as jnp

    logits, delta = apply_history_encoder(
        params,
        cfg,
        batch["support_tokens"],
        batch["support_valid"],
        batch["prefix_tokens"],
        batch["prefix_valid"],
        batch["query_features"],
        retain_support=retain_support,
        retain_prefix=retain_prefix,
    )
    cond_ids = jnp.asarray(batch["condition_id"], dtype=jnp.int32)
    one_hot = jax.nn.one_hot(cond_ids, cfg.num_conditions, dtype=jnp.float32)
    ce = -jnp.sum(one_hot * jax.nn.log_softmax(logits, axis=-1), axis=-1).mean()
    frozen_table = jax.lax.stop_gradient(condition_table)
    target_z = (
        jnp.asarray(batch["target_latent"], dtype=jnp.float32)
        if "target_latent" in batch
        else frozen_table[cond_ids]
    )
    soft_z = decode_steering_latent(logits, delta, frozen_table, mode="soft")
    latent_mse = jnp.mean(jnp.square(soft_z - target_z))
    acc = jnp.mean(jnp.argmax(logits, axis=-1) == cond_ids)
    return ce + latent_weight * latent_mse, ce, latent_mse, acc


def history_belief_loss(
    params,
    cfg,
    batch,
    condition_table,
    *,
    cleared_weight=0.5,
    prefix_weight=0.5,
    latent_weight=0.25,
):
    """Combined belief classification + latent regression loss across retained, cleared, and prefix views."""
    ret_loss, ret_ce, ret_mse, ret_acc = _belief_view_loss(
        params, cfg, batch, condition_table, retain_support=True, retain_prefix=True, latent_weight=latent_weight
    )
    clr_loss, clr_ce, _, clr_acc = _belief_view_loss(
        params, cfg, batch, condition_table, retain_support=False, retain_prefix=False, latent_weight=latent_weight
    )
    pfx_loss, pfx_ce, _, pfx_acc = _belief_view_loss(
        params, cfg, batch, condition_table, retain_support=False, retain_prefix=True, latent_weight=latent_weight
    )
    denom = 1.0 + cleared_weight + prefix_weight
    total = (ret_loss + cleared_weight * clr_loss + prefix_weight * pfx_loss) / denom
    metrics = dict(
        loss=total,
        retained_ce=ret_ce,
        retained_mse=ret_mse,
        retained_acc=ret_acc,
        cleared_ce=clr_ce,
        cleared_acc=clr_acc,
        prefix_ce=pfx_ce,
        prefix_acc=pfx_acc,
    )
    return total, metrics


def counterfactual_margin_loss(logits_a, logits_b, targets_a, targets_b, *, pair_mask=None):
    """Supervise how log-probabilities change across paired evidence contexts."""
    import jax
    import jax.numpy as jnp

    diff_logits = jax.nn.log_softmax(logits_a, axis=-1) - jax.nn.log_softmax(logits_b, axis=-1)
    diff_targets = jnp.asarray(targets_a, dtype=jnp.float32) - jnp.asarray(targets_b, dtype=jnp.float32)
    per_pair = jnp.mean(jnp.square(diff_logits - diff_targets), axis=-1)
    if pair_mask is None:
        return jnp.mean(per_pair)
    mask = jnp.asarray(pair_mask, dtype=jnp.float32)
    return jnp.sum(per_pair * mask) / jnp.maximum(jnp.sum(mask), 1.0)
