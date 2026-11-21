"""V2 Equivariant Visuomotor SysID readers, Perceiver-ICBA baseline, SO(3) Procrustes & streaming.

Key upgrades over `readers.py` (V1):
1. `factored_equiv_v2`: Action-blind visual motion extractor with:
   - Dense DIS optical flow + luminance-gradient channels (`dis_lumaflow`, 9 channels) OR
     pure-JAX `luminance_flow` (7 channels) / `rgb` (8 channels).
   - Residual multi-scale spatial-attention CNN (preserves fine end-effector motion cues).
   - **Exact SO(2)-invariant vertical Z-parallax cancellation**: subtracts a learned
     `local_dz * u_z(visual_feat)` vector before Procrustes alignment so oblique cameras
     do not leak vertical lifting/lowering into planar yaw estimation.
2. `perceiver_icba`: Faithful Yang, Finn & Sadigh–style monolithic Perceiver-IO sequence
   encoder (learnable latent array cross-attending over `[vis_feat, issued, delta_eef, delta_jaw]`
   tokens + latent self-attention blocks) for non-strawman monolithic scaling-law comparisons.
3. Differentiable 3D `procrustes_so3`: Full SO(3) SVD Procrustes contraction for 3D
   command-frame rotations (yaw, pitch, roll) + anisotropic per-axis gains `(g_x, g_y, g_z)`.
4. `StreamingProcrustesEstimator`: O(1)-memory online sufficient-statistic accumulator
   allowing zero-probe, within-episode adaptation on Attempt 1.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np

from .readers import (
    _apply_mlp,
    _apply_scale_and_grip,
    _coord_grid,
    _init_mlp,
    _layer_norm,
    _normalize_cs,
    _procrustes_yaw,
    _unit_rgb,
    _vis_inputs,
)


V2_MODEL_NAMES = (
    "factored_equiv_v2",
    "factored_no_equiv_v2",
    "perceiver_icba",
    "factored_equiv_so3",
    "equiv_flow_homography",
)

V2_VISUAL_MODES = ("luminance_flow", "rgb", "dis_lumaflow")


@dataclass(frozen=True)
class ReaderV2Config:
    """Configuration for V2 equivariant readers and Perceiver-ICBA monolithic baseline."""

    model_type: str = "factored_equiv_v2"
    visual_mode: str = "dis_lumaflow"
    max_chunks: int = 32
    cnn_width: int = 96
    mlp_width: int = 192
    use_dz_parallax: bool = True
    # Perceiver-ICBA hyperparameters (monolithic baseline)
    perceiver_latents: int = 8
    perceiver_width: int = 192
    perceiver_layers: int = 3
    perceiver_heads: int = 4

    def __post_init__(self):
        if self.model_type not in V2_MODEL_NAMES:
            raise ValueError(f"Unknown V2 model_type {self.model_type!r}; expected {V2_MODEL_NAMES}")
        if self.visual_mode not in V2_VISUAL_MODES:
            raise ValueError(f"Unknown V2 visual_mode {self.visual_mode!r}; expected {V2_VISUAL_MODES}")
        if self.perceiver_width % self.perceiver_heads != 0:
            raise ValueError("perceiver_width must be divisible by perceiver_heads")


def _conv2d_nhwc(x, weight, bias, stride=1):
    import jax

    y = jax.lax.conv_general_dilated(
        x,
        weight,
        window_strides=(stride, stride),
        padding="SAME",
        dimension_numbers=("NHWC", "HWIO", "NHWC"),
    )
    return y + bias[None, None, None, :]


def _init_res_cnn(key, in_channels: int, cnn_width: int):
    """4-stage residual spatial-attention CNN preserving sharp end-effector flow."""
    import jax
    import jax.numpy as jnp

    keys = jax.random.split(key, 10)
    init = jax.nn.initializers.lecun_normal()
    w1 = max(24, cnn_width // 2)
    w2 = cnn_width
    return dict(
        c1_w=init(keys[0], (3, 3, in_channels, w1), dtype=jnp.float32),
        c1_b=jnp.zeros((w1,), dtype=jnp.float32),
        c2_w=init(keys[1], (3, 3, w1, w2), dtype=jnp.float32),
        c2_b=jnp.zeros((w2,), dtype=jnp.float32),
        # Residual block at stride-4 resolution (16x16 for 64x64 input)
        r1_w=init(keys[2], (3, 3, w2, w2), dtype=jnp.float32),
        r1_b=jnp.zeros((w2,), dtype=jnp.float32),
        r2_w=init(keys[3], (3, 3, w2, w2), dtype=jnp.float32),
        r2_b=jnp.zeros((w2,), dtype=jnp.float32),
        c3_w=init(keys[4], (3, 3, w2, w2), dtype=jnp.float32),
        c3_b=jnp.zeros((w2,), dtype=jnp.float32),
        # Dual attention heads (multi-head spatial pooling: 2 heads -> 2*w2 features)
        pool_w=init(keys[5], (w2, 2), dtype=jnp.float32),
        pool_b=jnp.zeros((2,), dtype=jnp.float32),
        proj_w=init(keys[6], (2 * w2, w2), dtype=jnp.float32),
        proj_b=jnp.zeros((w2,), dtype=jnp.float32),
    )


def _apply_res_cnn(p, images):
    import jax
    import jax.numpy as jnp

    h = jax.nn.silu(_conv2d_nhwc(images, p["c1_w"], p["c1_b"], stride=2))
    h = jax.nn.silu(_conv2d_nhwc(h, p["c2_w"], p["c2_b"], stride=2))
    res = jax.nn.silu(_conv2d_nhwc(h, p["r1_w"], p["r1_b"], stride=1))
    res = _conv2d_nhwc(res, p["r2_w"], p["r2_b"], stride=1)
    h = jax.nn.silu(h + res)
    h = jax.nn.silu(_conv2d_nhwc(h, p["c3_w"], p["c3_b"], stride=2))
    n, gh, gw, c = h.shape
    flat = h.reshape(n, gh * gw, c)
    logits = flat @ p["pool_w"] + p["pool_b"]  # [N, HW, 2]
    w = jax.nn.softmax(logits, axis=1)
    pooled = jnp.einsum("nsh,nsc->nhc", w, flat).reshape(n, 2 * c)
    return jax.nn.silu(pooled @ p["proj_w"] + p["proj_b"])


def _init_perceiver(key, in_dim: int, max_chunks: int, n_latents: int, width: int, layers: int, out_dim: int):
    """Monolithic Perceiver-IO encoder (Yang, Finn & Sadigh style)."""
    import jax
    import jax.numpy as jnp

    init = jax.nn.initializers.lecun_normal()
    norm_init = jax.nn.initializers.normal(stddev=0.02)
    keys = jax.random.split(key, 6 + layers)
    # Cross-attention block: latents attend to variable-length sequence tokens.
    k_q, k_kv, k_co, k_cf1, k_cf2 = jax.random.split(keys[3], 5)
    cross = dict(
        ln_q_s=jnp.ones((width,), dtype=jnp.float32),
        ln_q_b=jnp.zeros((width,), dtype=jnp.float32),
        ln_kv_s=jnp.ones((width,), dtype=jnp.float32),
        ln_kv_b=jnp.zeros((width,), dtype=jnp.float32),
        q_w=init(k_q, (width, width), dtype=jnp.float32),
        q_b=jnp.zeros((width,), dtype=jnp.float32),
        kv_w=init(k_kv, (width, 2 * width), dtype=jnp.float32),
        kv_b=jnp.zeros((2 * width,), dtype=jnp.float32),
        out_w=init(k_co, (width, width), dtype=jnp.float32),
        out_b=jnp.zeros((width,), dtype=jnp.float32),
        ln2_s=jnp.ones((width,), dtype=jnp.float32),
        ln2_b=jnp.zeros((width,), dtype=jnp.float32),
        f1_w=init(k_cf1, (width, 4 * width), dtype=jnp.float32),
        f1_b=jnp.zeros((4 * width,), dtype=jnp.float32),
        f2_w=init(k_cf2, (4 * width, width), dtype=jnp.float32),
        f2_b=jnp.zeros((width,), dtype=jnp.float32),
    )
    self_blocks = []
    for bk in keys[6:]:
        k1, k2, k3, k4 = jax.random.split(bk, 4)
        self_blocks.append(
            dict(
                ln1_s=jnp.ones((width,), dtype=jnp.float32),
                ln1_b=jnp.zeros((width,), dtype=jnp.float32),
                qkv_w=init(k1, (width, 3 * width), dtype=jnp.float32),
                qkv_b=jnp.zeros((3 * width,), dtype=jnp.float32),
                out_w=init(k2, (width, width), dtype=jnp.float32),
                out_b=jnp.zeros((width,), dtype=jnp.float32),
                ln2_s=jnp.ones((width,), dtype=jnp.float32),
                ln2_b=jnp.zeros((width,), dtype=jnp.float32),
                f1_w=init(k3, (width, 4 * width), dtype=jnp.float32),
                f1_b=jnp.zeros((4 * width,), dtype=jnp.float32),
                f2_w=init(k4, (4 * width, width), dtype=jnp.float32),
                f2_b=jnp.zeros((width,), dtype=jnp.float32),
            )
        )
    return dict(
        in_w=init(keys[0], (in_dim, width), dtype=jnp.float32),
        in_b=jnp.zeros((width,), dtype=jnp.float32),
        pos=norm_init(keys[1], (1, max_chunks, width), dtype=jnp.float32),
        latents=norm_init(keys[2], (1, n_latents, width), dtype=jnp.float32),
        cross=cross,
        self_blocks=self_blocks,
        final_s=jnp.ones((width,), dtype=jnp.float32),
        final_b=jnp.zeros((width,), dtype=jnp.float32),
        head_w=init(keys[4], (width, out_dim), dtype=jnp.float32),
        head_b=jnp.zeros((out_dim,), dtype=jnp.float32),
    )


def _apply_perceiver(params, tokens, valid, heads: int):
    import jax
    import jax.numpy as jnp

    bsz, seq_len, _ = tokens.shape
    width = params["in_b"].shape[0]
    head_dim = width // heads
    has_valid = jnp.any(valid, axis=1)

    kv_tok = tokens @ params["in_w"] + params["in_b"] + params["pos"][:, :seq_len, :]
    kv_tok = jnp.where(valid[..., None], kv_tok, 0.0)
    z = jnp.broadcast_to(params["latents"], (bsz, params["latents"].shape[1], width))

    # Cross-attention from latents to valid input tokens.
    c = params["cross"]
    q_in = _layer_norm(z, c["ln_q_s"], c["ln_q_b"])
    kv_in = _layer_norm(kv_tok, c["ln_kv_s"], c["ln_kv_b"])
    q = (q_in @ c["q_w"] + c["q_b"]).reshape(bsz, z.shape[1], heads, head_dim)
    kv = (kv_in @ c["kv_w"] + c["kv_b"]).reshape(bsz, seq_len, 2, heads, head_dim)
    k, v = kv[:, :, 0], kv[:, :, 1]
    q = jnp.transpose(q, (0, 2, 1, 3))
    k = jnp.transpose(k, (0, 2, 1, 3))
    v = jnp.transpose(v, (0, 2, 1, 3))
    scores = jnp.einsum("bhld,bhsd->bhls", q, k) * (head_dim ** -0.5)
    # Safe mask when all valid==False (cleared context)
    safe_valid = jnp.where(has_valid[:, None], valid, jnp.ones_like(valid))
    scores = jnp.where(safe_valid[:, None, None, :], scores, -1e9)
    w = jax.nn.softmax(scores, axis=-1)
    att = jnp.einsum("bhls,bhsd->bhld", w, v)
    att = jnp.transpose(att, (0, 2, 1, 3)).reshape(bsz, z.shape[1], width)
    z = z + jnp.where(has_valid[:, None, None], att @ c["out_w"] + c["out_b"], 0.0)
    ff = jax.nn.silu(_layer_norm(z, c["ln2_s"], c["ln2_b"]) @ c["f1_w"] + c["f1_b"])
    z = z + (ff @ c["f2_w"] + c["f2_b"])

    # Latent self-attention stack.
    for blk in params["self_blocks"]:
        n1 = _layer_norm(z, blk["ln1_s"], blk["ln1_b"])
        qkv = n1 @ blk["qkv_w"] + blk["qkv_b"]
        sq, sk, sv = jnp.split(qkv, 3, axis=-1)
        sq = jnp.transpose(sq.reshape(bsz, z.shape[1], heads, head_dim), (0, 2, 1, 3))
        sk = jnp.transpose(sk.reshape(bsz, z.shape[1], heads, head_dim), (0, 2, 1, 3))
        sv = jnp.transpose(sv.reshape(bsz, z.shape[1], heads, head_dim), (0, 2, 1, 3))
        sc = jnp.einsum("bhqd,bhkd->bhqk", sq, sk) * (head_dim ** -0.5)
        sw = jax.nn.softmax(sc, axis=-1)
        sa = jnp.transpose(jnp.einsum("bhqk,bhkd->bhqd", sw, sv), (0, 2, 1, 3)).reshape(bsz, z.shape[1], width)
        z = z + (sa @ blk["out_w"] + blk["out_b"])
        ff = jax.nn.silu(_layer_norm(z, blk["ln2_s"], blk["ln2_b"]) @ blk["f1_w"] + blk["f1_b"])
        z = z + (ff @ blk["f2_w"] + blk["f2_b"])

    pooled = jnp.mean(_layer_norm(z, params["final_s"], params["final_b"]), axis=1)
    return pooled @ params["head_w"] + params["head_b"]


def _vis_inputs_v2(batch, visual_mode: str):
    import jax.numpy as jnp

    mask = jnp.asarray(batch["valid"], dtype=bool)[..., None, None, None]
    before = jnp.where(mask, _unit_rgb(batch["img_before"]), 0.0)
    after = jnp.where(mask, _unit_rgb(batch["img_after"]), 0.0)
    if visual_mode in ("luminance_flow", "rgb"):
        return _vis_inputs(before, after, visual_mode=visual_mode)
    # dis_lumaflow: 9 channels = [fx, fy, ||f||, dt_luma, ||dRGB||, gx, gy, x, y]
    b, k, h, w, _ = before.shape
    coords = jnp.broadcast_to(_coord_grid(h, w)[None, None], (b, k, h, w, 2))
    diff = after - before
    luma_b = jnp.mean(before, axis=-1, keepdims=True)
    luma_a = jnp.mean(after, axis=-1, keepdims=True)
    dt = luma_a - luma_b
    motion_mag = jnp.sqrt(jnp.mean(jnp.square(diff), axis=-1, keepdims=True) + 1e-6)
    avg = 0.5 * (luma_b + luma_a)
    gx = jnp.pad(0.5 * (avg[:, :, :, 2:, :] - avg[:, :, :, :-2, :]), ((0, 0), (0, 0), (0, 0), (1, 1), (0, 0)))
    gy = jnp.pad(0.5 * (avg[:, :, 2:, :, :] - avg[:, :, :-2, :, :]), ((0, 0), (0, 0), (1, 1), (0, 0), (0, 0)))
    if "dis_flow" in batch:
        flow = jnp.where(mask, jnp.asarray(batch["dis_flow"], dtype=jnp.float32), 0.0)
    else:
        flow = jnp.zeros((b, k, h, w, 2), dtype=jnp.float32)
    flow_mag = jnp.sqrt(jnp.sum(jnp.square(flow), axis=-1, keepdims=True) + 1e-6)
    x = jnp.concatenate((flow, flow_mag, dt, motion_mag, gx, gy, coords), axis=-1)
    return x.reshape(b * k, h, w, 9), b, k


def procrustes_so3(commands_3d, vectors_3d, weights, has_valid):
    """Differentiable weighted 3D Orthogonal Procrustes onto SO(3).

    Given command vectors `a_k in R^3`, action-blind visual 3D motion vectors `v_k in R^3`,
    and non-negative weights `w_k`, solves:
        R_hat = argmax_{R in SO(3)} sum_k w_k <v_k, R a_k>
    via SVD of the 3x3 cross-covariance `H = sum_k w_k v_k a_k^T = U S V^T`:
        R_hat = U diag(1, 1, det(U V^T)) V^T.

    Exact SO(3) equivariance: replacing `a_k -> Q^T a_k` for any `Q in SO(3)` shifts
    `H -> H Q` and therefore `R_hat -> R_hat Q` identically.
    """
    import jax.numpy as jnp

    # H[b, i, j] = sum_k w[b, k] * v[b, k, i] * a[b, k, j]
    h_mat = jnp.einsum("bk,bki,bkj->bij", weights, vectors_3d, commands_3d)
    # Small Tikhonov stabilization for zero/degenerate batches so SVD gradients stay finite.
    eye = jnp.eye(3, dtype=jnp.float32)[None]
    h_norm = jnp.sqrt(jnp.sum(jnp.square(h_mat), axis=(-1, -2), keepdims=True) + 1e-12)
    h_safe = jnp.where(has_valid[:, None, None] & (h_norm > 1e-6), h_mat, eye)
    u, _, vt = jnp.linalg.svd(h_safe, full_matrices=False)
    uvt = jnp.matmul(u, vt)
    det_sign = jnp.where(jnp.linalg.det(uvt) >= 0.0, 1.0, -1.0)
    d = jnp.stack(
        (
            jnp.ones_like(det_sign),
            jnp.ones_like(det_sign),
            det_sign,
        ),
        axis=-1,
    )
    r_hat = jnp.einsum("bij,bj,bjk->bik", u, d, vt)
    return jnp.where(has_valid[:, None, None], r_hat, eye)


def _init_flow_mask_cnn(key, in_channels: int = 7, width: int = 48):
    """Direction-blind pixel segmentation CNN for EquivFlow-Procrustes.

    Sees ONLY scalar motion magnitude / luminance / spatial coordinates (NEVER flow
    direction or action commands), predicting:
      1. Spatial attention logits over the 32x32 flow grid (end-effector segmentation).
      2. Per-chunk scalar confidence weight for weighted Procrustes.
    """
    import jax
    import jax.numpy as jnp

    keys = jax.random.split(key, 6)
    init = jax.nn.initializers.lecun_normal()
    return dict(
        c1_w=init(keys[0], (3, 3, in_channels, width), dtype=jnp.float32),
        c1_b=jnp.zeros((width,), dtype=jnp.float32),
        c2_w=init(keys[1], (3, 3, width, width), dtype=jnp.float32),
        c2_b=jnp.zeros((width,), dtype=jnp.float32),
        c3_w=init(keys[2], (3, 3, width, width), dtype=jnp.float32),
        c3_b=jnp.zeros((width,), dtype=jnp.float32),
        mask_w=init(keys[3], (1, 1, width, 1), dtype=jnp.float32),
        mask_b=jnp.zeros((1,), dtype=jnp.float32),
        conf_w=init(keys[4], (width, 1), dtype=jnp.float32),
        conf_b=jnp.zeros((1,), dtype=jnp.float32),
        flow_mag_beta=jnp.asarray(1.5, dtype=jnp.float32),
    )


def _apply_flow_mask_pooling(p, batch):
    """Linearly pool 2D optical flow with a direction-blind learned spatial attention mask."""
    import jax
    import jax.numpy as jnp

    valid = jnp.asarray(batch["valid"], dtype=bool)
    mask = valid[..., None, None, None]
    before = jnp.where(mask, _unit_rgb(batch["img_before"]), 0.0)
    after = jnp.where(mask, _unit_rgb(batch["img_after"]), 0.0)
    b, k, h, w, _ = before.shape
    if "dis_flow" in batch:
        flow = jnp.where(mask, jnp.asarray(batch["dis_flow"], dtype=jnp.float32), 0.0)
    else:
        flow = jnp.zeros((b, k, h, w, 2), dtype=jnp.float32)

    # Strictly direction-blind scalar pixel features: [||f||, |dt|, ||dRGB||, ||grad||, luma_b, x, y]
    diff = after - before
    luma_b = jnp.mean(before, axis=-1, keepdims=True)
    luma_a = jnp.mean(after, axis=-1, keepdims=True)
    abs_dt = jnp.abs(luma_a - luma_b)
    d_rgb = jnp.sqrt(jnp.mean(jnp.square(diff), axis=-1, keepdims=True) + 1e-6)
    avg = 0.5 * (luma_b + luma_a)
    gx = jnp.pad(0.5 * (avg[:, :, :, 2:, :] - avg[:, :, :, :-2, :]), ((0, 0), (0, 0), (0, 0), (1, 1), (0, 0)))
    gy = jnp.pad(0.5 * (avg[:, :, 2:, :, :] - avg[:, :, :-2, :, :]), ((0, 0), (0, 0), (1, 1), (0, 0), (0, 0)))
    grad_mag = jnp.sqrt(jnp.square(gx) + jnp.square(gy) + 1e-6)
    flow_mag = jnp.sqrt(jnp.sum(jnp.square(flow), axis=-1, keepdims=True) + 1e-6)
    coords = jnp.broadcast_to(_coord_grid(h, w)[None, None], (b, k, h, w, 2))
    x = jnp.concatenate((flow_mag, abs_dt, d_rgb, grad_mag, luma_b, coords), axis=-1)
    x_flat = x.reshape(b * k, h, w, 7)

    # Stride-2 -> H/2 x W/2 (32x32 for 64x64)
    feat = jax.nn.silu(_conv2d_nhwc(x_flat, p["c1_w"], p["c1_b"], stride=2))
    feat = jax.nn.silu(_conv2d_nhwc(feat, p["c2_w"], p["c2_b"], stride=1))
    feat = jax.nn.silu(feat + _conv2d_nhwc(feat, p["c3_w"], p["c3_b"], stride=1))
    logits = _conv2d_nhwc(feat, p["mask_w"], p["mask_b"], stride=1)[..., 0]  # [BK, H/2, W/2]

    # Exact 2x2 average-pooled dense flow on the same H/2 x W/2 grid.
    flow_flat = flow.reshape(b * k, h // 2, 2, w // 2, 2, 2).mean(axis=(2, 4))  # [BK, H/2, W/2, 2]
    fmag_ds = jnp.sqrt(jnp.sum(jnp.square(flow_flat), axis=-1) + 1e-6)
    # Soft motion prior focuses initial attention on moving pixels while CNN suppresses non-EEF motion.
    beta = jax.nn.softplus(p["flow_mag_beta"])
    attn_logits = (logits + beta * jnp.log(fmag_ds + 1e-3)).reshape(b * k, -1)
    attn = jax.nn.softmax(attn_logits, axis=-1)  # [BK, HW/4]
    flow_vec = jnp.einsum("ns,nsc->nc", attn, flow_flat.reshape(b * k, -1, 2)).reshape(b, k, 2)

    # Per-chunk confidence weight (direction-blind)
    gfeat = jnp.einsum("ns,nsc->nc", attn, feat.reshape(b * k, -1, feat.shape[-1]))
    conf_logit = (gfeat @ p["conf_w"] + p["conf_b"]).reshape(b, k)
    conf_w = jnp.where(valid, jax.nn.softplus(conf_logit) + 1e-4, 0.0)
    return flow_vec, conf_w


def init_reader_v2(key, cfg: ReaderV2Config):
    """Initialize parameters for any V2 architecture."""
    import jax
    import jax.numpy as jnp

    k1, k2, k3, k4 = jax.random.split(key, 4)
    vis_ch = {"luminance_flow": 7, "rgb": 8, "dis_lumaflow": 9}[cfg.visual_mode]

    if cfg.model_type == "perceiver_icba":
        return dict(
            cnn=_init_res_cnn(k1, vis_ch, cfg.cnn_width),
            perceiver=_init_perceiver(
                k2,
                cfg.cnn_width + 12,
                cfg.max_chunks,
                cfg.perceiver_latents,
                cfg.perceiver_width,
                cfg.perceiver_layers,
                4,
            ),
        )

    scale_head = _init_mlp(k3, (5, 64, 64, 2))
    grip_head = _init_mlp(k4, (4, 48, 48, 2))

    if cfg.model_type == "factored_equiv_v2":
        # Action-blind visual vector head outputs [vx, vy, conf_logit, uz_x, uz_y].
        # v_planar = [vx, vy] - local_dz * [uz_x, uz_y] cancels oblique-camera Z-parallax
        # while remaining 100% blind to horizontal commands a_xy (exact SO(2) equivariance).
        out_dim = 5 if cfg.use_dz_parallax else 3
        return dict(
            cnn=_init_res_cnn(k1, vis_ch, cfg.cnn_width),
            vis_vec=_init_mlp(k2, (cfg.cnn_width, cfg.mlp_width, cfg.mlp_width // 2, out_dim)),
            global_uz=jnp.zeros((2,), dtype=jnp.float32),
            scale_head=scale_head,
            grip_head=grip_head,
        )

    if cfg.model_type == "factored_no_equiv_v2":
        return dict(
            cnn=_init_res_cnn(k1, vis_ch, cfg.cnn_width),
            vis_head=_init_mlp(k2, (cfg.cnn_width + 3, cfg.mlp_width, cfg.mlp_width // 2, 3)),
            scale_head=scale_head,
            grip_head=grip_head,
        )

    if cfg.model_type == "factored_equiv_so3":
        # Outputs action-blind 3D visual motion vector [vx, vy, vz, conf_logit].
        # Anisotropic per-axis gain head shares features per axis, producing 3 log-scales.
        return dict(
            cnn=_init_res_cnn(k1, vis_ch, cfg.cnn_width),
            vis_vec3=_init_mlp(k2, (cfg.cnn_width, cfg.mlp_width, cfg.mlp_width // 2, 4)),
            scale_head=scale_head,
            grip_head=grip_head,
        )

    if cfg.model_type == "equiv_flow_homography":
        # Direction-blind spatial attention mask over dense flow + linear 2x2 camera homography
        # + exact SO(2)-invariant local_dz vertical parallax cancellation -> zero cardinal quantization!
        return dict(
            mask_cnn=_init_flow_mask_cnn(k1, in_channels=7, width=max(32, cfg.cnn_width // 2)),
            homography=jnp.eye(2, dtype=jnp.float32),
            uz=jnp.zeros((2,), dtype=jnp.float32),
            scale_head=scale_head,
            grip_head=grip_head,
        )

    raise ValueError(cfg.model_type)


def extract_visual_vectors_v2(params, cfg: ReaderV2Config, batch):
    """Return action-blind planar visual motion vectors [B, K, 2] and weights [B, K]."""
    import jax
    import jax.numpy as jnp

    valid = jnp.asarray(batch["valid"], dtype=bool)
    if cfg.model_type == "equiv_flow_homography":
        flow_vec, w = _apply_flow_mask_pooling(params["mask_cnn"], batch)
        # Linear 2x2 camera homography: strictly preserves all continuous 2D flow directions!
        v_raw = jnp.einsum("ij,bkj->bki", params["homography"], flow_vec)
        if cfg.use_dz_parallax:
            local_dz = jnp.where(valid, 100.0 * jnp.asarray(batch["delta_eef"], dtype=jnp.float32)[..., 2], 0.0)
            v_planar = v_raw - local_dz[..., None] * params["uz"][None, None, :]
        else:
            v_planar = v_raw
        return v_planar, w

    images, b, k = _vis_inputs_v2(batch, cfg.visual_mode)
    vis = _apply_res_cnn(params["cnn"], images).reshape(b, k, -1)
    vec_out = _apply_mlp(params["vis_vec"], vis)
    v_raw = vec_out[..., :2]
    w = jnp.where(valid, jax.nn.softplus(vec_out[..., 2]) + 1e-4, 0.0)
    if cfg.use_dz_parallax:
        # local_dz (in cm) is strictly invariant to planar command-frame yaw.
        local_dz = jnp.where(valid, 100.0 * jnp.asarray(batch["delta_eef"], dtype=jnp.float32)[..., 2], 0.0)
        u_z = vec_out[..., 3:5] + params["global_uz"][None, None, :]
        v_planar = v_raw - local_dz[..., None] * u_z
    else:
        v_planar = v_raw
    return v_planar, w


def apply_reader_v2(params, cfg: ReaderV2Config, batch):
    """Forward pass for V2 readers.

    For SO(2) models: returns `(cos_yaw, sin_yaw, log_scale, grip_logit)`.
    For `factored_equiv_so3`: returns `(rot_mat_3x3, log_scale_xyz, grip_logit)`.
    """
    import jax
    import jax.numpy as jnp

    valid = jnp.asarray(batch["valid"], dtype=bool)
    has_valid = jnp.any(valid, axis=1)

    def sig(field):
        return jnp.where(valid[..., None], jnp.asarray(batch[field], dtype=jnp.float32), 0.0)

    issued = sig("issued_mean")
    delta_eef = 100.0 * sig("delta_eef")
    delta_jaw = 100.0 * sig("delta_jaw")

    if cfg.model_type == "perceiver_icba":
        images, b, k = _vis_inputs_v2(batch, cfg.visual_mode)
        vis = _apply_res_cnn(params["cnn"], images).reshape(b, k, -1)
        tok = jnp.concatenate((vis, issued, delta_eef, delta_jaw), axis=-1)
        raw = _apply_perceiver(params["perceiver"], tok, valid, cfg.perceiver_heads)
        cos_hat, sin_hat = _normalize_cs(raw[:, 0], raw[:, 1], has_valid)
        log_scale = jnp.where(has_valid, raw[:, 2], 0.0)
        grip_logit = jnp.where(has_valid, raw[:, 3], 10.0)
        return cos_hat, sin_hat, log_scale, grip_logit

    log_scale, grip_logit = _apply_scale_and_grip(params, issued, delta_eef, delta_jaw, valid)

    if cfg.model_type in ("factored_equiv_v2", "equiv_flow_homography"):
        v_planar, w = extract_visual_vectors_v2(params, cfg, batch)
        cos_hat, sin_hat = _procrustes_yaw(issued[..., :2], v_planar, w, has_valid)
        return cos_hat, sin_hat, log_scale, grip_logit

    if cfg.model_type == "factored_no_equiv_v2":
        images, b, k = _vis_inputs_v2(batch, cfg.visual_mode)
        vis = _apply_res_cnn(params["cnn"], images).reshape(b, k, -1)
        vis_in = jnp.concatenate((vis, issued[..., :2], delta_eef[..., 2:3]), axis=-1)
        v_out = _apply_mlp(params["vis_head"], vis_in)
        v_w = jnp.where(valid, jax.nn.softplus(v_out[..., 2]) + 1e-4, 0.0)
        c_raw = jnp.sum(v_w * v_out[..., 0], axis=1)
        s_raw = jnp.sum(v_w * v_out[..., 1], axis=1)
        cos_hat, sin_hat = _normalize_cs(c_raw, s_raw, has_valid)
        return cos_hat, sin_hat, log_scale, grip_logit

    if cfg.model_type == "factored_equiv_so3":
        images, b, k = _vis_inputs_v2(batch, cfg.visual_mode)
        vis = _apply_res_cnn(params["cnn"], images).reshape(b, k, -1)
        vec3_out = _apply_mlp(params["vis_vec3"], vis)
        v_3d = vec3_out[..., :3]
        w = jnp.where(valid, jax.nn.softplus(vec3_out[..., 3]) + 1e-4, 0.0)
        r_hat = procrustes_so3(issued[..., :3], v_3d, w, has_valid)
        # Per-axis anisotropic log-scale estimation using rotated commands vs local delta_eef.
        iss_xyz = issued[..., :3]
        deef_xyz = delta_eef[..., :3]
        eps = 1e-4
        scale_feats = jnp.stack(
            (
                iss_xyz,
                deef_xyz,
                jnp.abs(iss_xyz),
                jnp.abs(deef_xyz),
                jnp.log(jnp.abs(deef_xyz) + eps) - jnp.log(jnp.abs(iss_xyz) + eps),
            ),
            axis=-1,
        )
        s_out = _apply_mlp(params["scale_head"], scale_feats)  # [B, K, 3, 2]
        s_val, s_conf = s_out[..., 0], s_out[..., 1]
        s_w = jnp.where(valid[..., None], jax.nn.softplus(s_conf) + 1e-4, 0.0)
        s_den = jnp.maximum(jnp.sum(s_w, axis=1), 1e-6)
        log_scale_xyz = jnp.where(has_valid[:, None], jnp.sum(s_w * s_val, axis=1) / s_den, 0.0)
        return r_hat, log_scale_xyz, grip_logit

    raise ValueError(cfg.model_type)


def reader_loss_v2(params, cfg: ReaderV2Config, batch):
    """Training loss for V2 readers (SO(2) or SO(3))."""
    import jax
    import jax.numpy as jnp

    if cfg.model_type == "factored_equiv_so3":
        r_hat, log_scale_xyz, grip_logit = apply_reader_v2(params, cfg, batch)
        r_true = jnp.asarray(batch["rot_mat"], dtype=jnp.float32)  # [B, 3, 3]
        # Chordal SO(3) geodesic proxy: ||R_hat - R_true||_F^2 / 4 = 3/2 - 1/2 tr(R_hat^T R_true)
        rot_loss = jnp.mean(jnp.sum(jnp.square(r_hat - r_true), axis=(-1, -2))) * 0.25
        log_scale_t = jnp.log(jnp.asarray(batch["xyz_scale"], dtype=jnp.float32))
        scale_loss = jnp.mean(jnp.square(log_scale_xyz - log_scale_t))
        grip_t = jnp.asarray(batch["gripper_sign"] > 0, dtype=jnp.float32)
        grip_loss = jnp.mean(jax.nn.softplus(grip_logit) - grip_t * grip_logit)
        total = rot_loss + scale_loss + grip_loss
        return total, dict(total=total, rot_loss=rot_loss, scale_loss=scale_loss, grip_loss=grip_loss)

    cos_h, sin_h, log_scale_h, grip_logit_h = apply_reader_v2(params, cfg, batch)
    yaw_rad = jnp.deg2rad(jnp.asarray(batch["yaw_deg"], dtype=jnp.float32))
    cos_t = jnp.cos(yaw_rad)
    sin_t = jnp.sin(yaw_rad)
    log_scale_t = jnp.log(jnp.asarray(batch["xy_scale"], dtype=jnp.float32))
    grip_t = jnp.asarray(batch["gripper_sign"] > 0, dtype=jnp.float32)

    yaw_loss = jnp.mean(1.0 - (cos_h * cos_t + sin_h * sin_t))
    scale_loss = jnp.mean(jnp.square(log_scale_h - log_scale_t))
    grip_loss = jnp.mean(jax.nn.softplus(grip_logit_h) - grip_t * grip_logit_h)
    total = yaw_loss + scale_loss + grip_loss
    return total, dict(total=total, yaw_loss=yaw_loss, scale_loss=scale_loss, grip_loss=grip_loss)


def predict_embodiment_v2(params, cfg: ReaderV2Config, batch):
    """Return decoded SO(2) predictions `(yaw_deg_hat, xy_scale_hat, grip_sign_hat)`."""
    cos_h, sin_h, log_scale_h, grip_logit_h = map(np.asarray, apply_reader_v2(params, cfg, batch))
    if not all(np.isfinite(x).all() for x in (cos_h, sin_h, log_scale_h, grip_logit_h)):
        raise FloatingPointError("Nonfinite V2 reader prediction")
    yaw_deg = np.degrees(np.arctan2(sin_h, cos_h)).astype(np.float32)
    xy_scale = np.exp(np.clip(log_scale_h, np.log(0.25), np.log(4.0))).astype(np.float32)
    grip_sign = np.where(grip_logit_h >= 0.0, 1, -1).astype(np.int32)
    return yaw_deg, xy_scale, grip_sign


class StreamingProcrustesEstimator:
    """O(1)-memory online streaming equivariant Procrustes accumulator.

    Accumulates additive sufficient statistics chunk-by-chunk during Attempt 1 (zero-probe,
    zero-reset online adaptation) or across attempts, updating `(yaw_hat, scale_hat, grip_hat)`
    in constant time per chunk.
    """

    def __init__(self, *, prior_weight: float = 1e-3, min_excitation: float = 0.02):
        self.prior_weight = float(prior_weight)
        self.min_excitation = float(min_excitation)
        self.reset()

    def reset(self):
        self.dot_sum = self.prior_weight
        self.cross_sum = 0.0
        self.weight_sum = 0.0
        self.scale_num = 0.0
        self.scale_den = 0.0
        self.grip_num = 0.0
        self.grip_den = 0.0
        self.n_chunks = 0

    def update(
        self,
        issued_xy: np.ndarray,
        v_planar: np.ndarray,
        vis_weight: float,
        *,
        log_scale_est: float | None = None,
        scale_weight: float = 1.0,
        grip_logit_est: float | None = None,
        grip_weight: float = 1.0,
    ):
        a = np.asarray(issued_xy, dtype=np.float64).reshape(2)
        v = np.asarray(v_planar, dtype=np.float64).reshape(2)
        w = max(float(vis_weight), 0.0)
        if np.linalg.norm(a) >= self.min_excitation and w > 0.0:
            self.dot_sum += w * (a[0] * v[0] + a[1] * v[1])
            self.cross_sum += w * (a[0] * v[1] - a[1] * v[0])
            self.weight_sum += w
        if log_scale_est is not None and scale_weight > 0.0:
            self.scale_num += float(scale_weight) * float(log_scale_est)
            self.scale_den += float(scale_weight)
        if grip_logit_est is not None and grip_weight > 0.0:
            self.grip_num += float(grip_weight) * float(grip_logit_est)
            self.grip_den += float(grip_weight)
        self.n_chunks += 1

    def estimate(self) -> tuple[float, float, int]:
        yaw_deg = float(math.degrees(math.atan2(self.cross_sum, self.dot_sum)))
        log_s = (self.scale_num / self.scale_den) if self.scale_den > 1e-8 else 0.0
        scale = float(np.exp(np.clip(log_s, math.log(0.25), math.log(4.0))))
        grip_logit = (self.grip_num / self.grip_den) if self.grip_den > 1e-8 else 10.0
        grip = 1 if grip_logit >= 0.0 else -1
        return yaw_deg, scale, grip
