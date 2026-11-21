"""JAX history readers and analytic proprioceptive baseline for cross-embodiment SysID."""

from dataclasses import dataclass
import math
import numpy as np


MODEL_NAMES = (
    "frame0_probe",
    "joint_transformer",
    "factored_no_equiv",
    "factored_equiv",
)


FRAME0_FIELDS = ("frame0_img", "frame0_state")
HISTORY_FIELDS = ("img_before", "img_after", "issued_mean", "delta_eef", "delta_jaw", "valid")


VISUAL_MODES = ("rgb", "luminance_flow", "hybrid")


@dataclass(frozen=True)
class ReaderConfig:
    """Hyperparameters shared across cross-embodiment neural history readers."""

    model_type: str = "factored_equiv"
    max_chunks: int = 32
    cnn_width: int = 64
    transformer_width: int = 128
    transformer_layers: int = 2
    transformer_heads: int = 4
    mlp_width: int = 128
    visual_mode: str = "rgb"

    def __post_init__(self):
        if self.model_type not in MODEL_NAMES:
            raise ValueError(f"Unknown model_type {self.model_type!r}; expected one of {MODEL_NAMES}")
        if self.visual_mode not in VISUAL_MODES:
            raise ValueError(f"Unknown visual_mode {self.visual_mode!r}; expected one of {VISUAL_MODES}")
        if self.transformer_width % self.transformer_heads != 0:
            raise ValueError("transformer_width must be divisible by transformer_heads")


def analytic_proprio_sysid(
    issued_mean: np.ndarray,
    delta_eef: np.ndarray,
    delta_jaw: np.ndarray,
    valid: np.ndarray,
    nominal_gain_m_per_cmd: float = 0.01,
):
    """Estimate (yaw_deg, xy_scale, gripper_sign) using ONLY local proprioception and commands.

    Supports both unbatched [K, ...] and batched [B, K, ...] inputs.
    Because local proprioception is invariant to hand-eye yaw, yaw_deg_hat stays near 0 deg.
    """
    iss = np.asarray(issued_mean, dtype=np.float64)
    deef = np.asarray(delta_eef, dtype=np.float64)
    djaw = np.asarray(delta_jaw, dtype=np.float64)
    val = np.asarray(valid, dtype=bool)
    unbatched = iss.ndim == 2
    if unbatched:
        iss, deef, djaw, val = iss[None], deef[None], djaw[None], val[None]

    iss = np.where(val[..., None], iss, 0.0)
    deef = np.where(val[..., None], deef, 0.0)
    djaw = np.where(val[..., None], djaw, 0.0)
    w = val.astype(np.float64)
    has_valid = val.any(axis=1)

    # 1. Planar OLS rotation between issued XY and local delta_eef XY.
    c_raw = np.sum(w * (iss[..., 0] * deef[..., 0] + iss[..., 1] * deef[..., 1]), axis=1)
    s_raw = np.sum(w * (iss[..., 0] * deef[..., 1] - iss[..., 1] * deef[..., 0]), axis=1)
    norm = np.hypot(c_raw, s_raw)
    yaw_deg = np.where(has_valid & (norm > 1e-10), np.degrees(np.arctan2(s_raw, c_raw)), 0.0)

    # 2. Translation scale ratio relative to calibrated nominal gain (m per issued command unit).
    num = np.sum(w * np.sum(iss[..., :3] * deef[..., :3], axis=-1), axis=1)
    den = float(nominal_gain_m_per_cmd) * np.sum(w * np.sum(iss[..., :3] ** 2, axis=-1), axis=1)
    scale = np.where(has_valid & (den > 1e-8) & (num > 0.0), num / np.maximum(den, 1e-12), 1.0)
    scale = np.clip(scale, 0.25, 4.0)

    # 3. Gripper polarity: nominal +1 closes jaw (delta_jaw[..., 0] < 0).
    grip_score = -np.sum(w * iss[..., 6] * djaw[..., 0], axis=1)
    grip_sign = np.where(~has_valid | (np.abs(grip_score) < 1e-7) | (grip_score >= 0.0), 1, -1).astype(np.int32)

    if unbatched:
        return float(yaw_deg[0]), float(scale[0]), int(grip_sign[0])
    return yaw_deg.astype(np.float32), scale.astype(np.float32), grip_sign


def fit_nominal_gain(issued_mean: np.ndarray, delta_eef: np.ndarray, valid: np.ndarray) -> float:
    """Fit scalar m-per-command gain on nominal (xy_scale=1.0) training trials."""
    iss = np.asarray(issued_mean, dtype=np.float64)
    deef = np.asarray(delta_eef, dtype=np.float64)
    w = np.asarray(valid, dtype=bool)
    iss = np.where(w[..., None], iss, 0.0)
    deef = np.where(w[..., None], deef, 0.0)
    num = float(np.sum(w * np.sum(iss[..., :3] * deef[..., :3], axis=-1)))
    den = float(np.sum(w * np.sum(iss[..., :3] ** 2, axis=-1)))
    if not np.isfinite(num + den) or den <= 1e-12 or num <= 0.0:
        raise ValueError("Cannot calibrate nominal gain from degenerate training trajectories")
    return num / den


def image_motion(before, after, valid):
    """Estimate image translation (pixels) by motion-weighted brightness constancy.

    Work one valid frame pair at a time. Static background gets zero weight.
    A textureless pair has no observable translation and yields a zero vector.
    """
    motion = np.zeros((*valid.shape, 2), dtype=np.float64)
    for index in zip(*np.nonzero(valid), strict=True):
        a = before[index].astype(np.float64).mean(axis=-1)
        b = after[index].astype(np.float64).mean(axis=-1)
        gy, gx = np.gradient((a + b) * 0.5)
        dt = (b - a).ravel()
        weight = np.sqrt(np.abs(dt))
        design = np.column_stack((gx.ravel(), gy.ravel())) * weight[:, None]
        flow, _, rank, _ = np.linalg.lstsq(design, -weight * dt, rcond=1e-5)
        if rank == 2:
            motion[index] = flow
    return motion


def fit_visual_projection(motion, delta_eef, valid):
    """Calibrate pixels per metre from nominal fitting traces in public coordinates."""
    projection, _, rank, _ = np.linalg.lstsq(delta_eef[valid], motion[valid], rcond=1e-5)
    singular = np.linalg.svd(projection[:2], compute_uv=False)
    if rank < 3 or singular[-1] <= 1e-5 * singular[0]:
        raise ValueError("Nominal calibration did not excite a measurable planar camera projection")
    return projection


def analytic_visual_yaw(issued, motion, local_dz, valid, projection):
    """Remove vertical image motion and fit planar rotation against issued XY."""
    planar = motion - local_dz[..., None] * projection[2]
    xy = planar @ np.linalg.pinv(projection[:2], rcond=1e-5)
    a = np.where(valid[..., None], issued[..., :2], 0.0)
    xy = np.where(valid[..., None], xy, 0.0)
    dot = (a * xy).sum(axis=(-1, -2))
    cross = (a[..., 0] * xy[..., 1] - a[..., 1] * xy[..., 0]).sum(axis=-1)
    return np.where(np.hypot(dot, cross) > 1e-10, np.degrees(np.arctan2(cross, dot)), 0.0)


def farneback_image_motion(img_before, img_after, valid) -> np.ndarray:
    """Dense OpenCV Farneback optical flow magnitude-weighted mean [B, K, 2]."""
    import cv2

    b_arr = np.asarray(img_before)
    a_arr = np.asarray(img_after)
    v_arr = np.asarray(valid, dtype=bool)
    bsz, n_k = v_arr.shape
    out = np.zeros((bsz, n_k, 2), dtype=np.float32)
    for bi in range(bsz):
        for ki in range(n_k):
            if not v_arr[bi, ki]:
                continue
            ib = b_arr[bi, ki]
            ia = a_arr[bi, ki]
            if ib.dtype != np.uint8:
                ib = np.clip(ib * 255.0, 0, 255).astype(np.uint8)
                ia = np.clip(ia * 255.0, 0, 255).astype(np.uint8)
            gb = cv2.cvtColor(ib, cv2.COLOR_RGB2GRAY) if ib.ndim == 3 else ib
            ga = cv2.cvtColor(ia, cv2.COLOR_RGB2GRAY) if ia.ndim == 3 else ia
            flow = cv2.calcOpticalFlowFarneback(gb, ga, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            mag = np.hypot(flow[..., 0], flow[..., 1])
            w = np.where(mag > 0.15, mag, 0.0)
            w_sum = float(w.sum())
            if w_sum > 1e-6:
                out[bi, ki, 0] = float((w * flow[..., 0]).sum() / w_sum)
                out[bi, ki, 1] = float((w * flow[..., 1]).sum() / w_sum)
    return out


def _coord_grid(height: int, width: int):
    import jax.numpy as jnp

    ys = jnp.linspace(-1.0, 1.0, height, dtype=jnp.float32)
    xs = jnp.linspace(-1.0, 1.0, width, dtype=jnp.float32)
    yy, xx = jnp.meshgrid(ys, xs, indexing="ij")
    return jnp.stack((xx, yy), axis=-1)


def _conv2d_nhwc(x, weight, bias, stride=2):
    import jax

    y = jax.lax.conv_general_dilated(
        x,
        weight,
        window_strides=(stride, stride),
        padding="SAME",
        dimension_numbers=("NHWC", "HWIO", "NHWC"),
    )
    return y + bias[None, None, None, :]


def _layer_norm(x, scale=None, bias=None, eps=1e-6):
    import jax
    import jax.numpy as jnp

    mean = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.mean(jnp.square(x - mean), axis=-1, keepdims=True)
    normed = (x - mean) * jax.lax.rsqrt(var + eps)
    if scale is not None:
        normed = normed * scale + bias
    return normed


def _init_cnn(key, in_channels: int, cnn_width: int):
    import jax
    import jax.numpy as jnp

    k1, k2, k3, kp = jax.random.split(key, 4)
    init = jax.nn.initializers.lecun_normal()
    w1_c = max(16, cnn_width // 2)
    return dict(
        c1_w=init(k1, (3, 3, in_channels, w1_c), dtype=jnp.float32),
        c1_b=jnp.zeros((w1_c,), dtype=jnp.float32),
        c2_w=init(k2, (3, 3, w1_c, cnn_width), dtype=jnp.float32),
        c2_b=jnp.zeros((cnn_width,), dtype=jnp.float32),
        c3_w=init(k3, (3, 3, cnn_width, cnn_width), dtype=jnp.float32),
        c3_b=jnp.zeros((cnn_width,), dtype=jnp.float32),
        pool_w=init(kp, (cnn_width, 1), dtype=jnp.float32),
        pool_b=jnp.zeros((1,), dtype=jnp.float32),
    )


def _apply_cnn(cnn_params, images):
    """Run 3-layer stride-2 CNN on [N, H, W, C] and spatial-attention-pool to [N, D]."""
    import jax
    import jax.numpy as jnp

    h = jax.nn.silu(_conv2d_nhwc(images, cnn_params["c1_w"], cnn_params["c1_b"], stride=2))
    h = jax.nn.silu(_conv2d_nhwc(h, cnn_params["c2_w"], cnn_params["c2_b"], stride=2))
    h = jax.nn.silu(_conv2d_nhwc(h, cnn_params["c3_w"], cnn_params["c3_b"], stride=2))
    n, gh, gw, c = h.shape
    flat = h.reshape(n, gh * gw, c)
    attn_logits = flat @ cnn_params["pool_w"] + cnn_params["pool_b"]
    weights = jax.nn.softmax(attn_logits, axis=1)
    return jnp.sum(weights * flat, axis=1)


def _init_mlp(key, dims):
    import jax
    import jax.numpy as jnp

    init = jax.nn.initializers.lecun_normal()
    keys = jax.random.split(key, len(dims) - 1)
    layers = []
    for lk, (d_in, d_out) in zip(keys, zip(dims[:-1], dims[1:], strict=True), strict=True):
        layers.append(dict(w=init(lk, (d_in, d_out), dtype=jnp.float32), b=jnp.zeros((d_out,), dtype=jnp.float32)))
    return layers


def _apply_mlp(layers, x):
    import jax

    h = x
    for layer in layers[:-1]:
        h = jax.nn.silu(h @ layer["w"] + layer["b"])
    last = layers[-1]
    return h @ last["w"] + last["b"]


def _init_transformer(key, in_dim: int, max_chunks: int, width: int, layers: int, out_dim: int):
    import jax
    import jax.numpy as jnp

    init = jax.nn.initializers.lecun_normal()
    pos_init = jax.nn.initializers.normal(stddev=0.02)
    keys = jax.random.split(key, 3 + layers)
    blocks = []
    for blk_key in keys[3:]:
        k_qkv, k_out, k_f1, k_f2 = jax.random.split(blk_key, 4)
        blocks.append(
            dict(
                ln1_s=jnp.ones((width,), dtype=jnp.float32),
                ln1_b=jnp.zeros((width,), dtype=jnp.float32),
                qkv_w=init(k_qkv, (width, 3 * width), dtype=jnp.float32),
                qkv_b=jnp.zeros((3 * width,), dtype=jnp.float32),
                out_w=init(k_out, (width, width), dtype=jnp.float32),
                out_b=jnp.zeros((width,), dtype=jnp.float32),
                ln2_s=jnp.ones((width,), dtype=jnp.float32),
                ln2_b=jnp.zeros((width,), dtype=jnp.float32),
                f1_w=init(k_f1, (width, 4 * width), dtype=jnp.float32),
                f1_b=jnp.zeros((4 * width,), dtype=jnp.float32),
                f2_w=init(k_f2, (4 * width, width), dtype=jnp.float32),
                f2_b=jnp.zeros((width,), dtype=jnp.float32),
            )
        )
    return dict(
        in_w=init(keys[0], (in_dim, width), dtype=jnp.float32),
        in_b=jnp.zeros((width,), dtype=jnp.float32),
        pos=pos_init(keys[1], (1, max_chunks, width), dtype=jnp.float32),
        blocks=blocks,
        final_s=jnp.ones((width,), dtype=jnp.float32),
        final_b=jnp.zeros((width,), dtype=jnp.float32),
        head_w=init(keys[2], (width, out_dim), dtype=jnp.float32),
        head_b=jnp.zeros((out_dim,), dtype=jnp.float32),
    )


def _apply_transformer(params, tokens, valid, heads: int):
    import jax
    import jax.numpy as jnp

    batch, seq_len, _ = tokens.shape
    width = params["in_b"].shape[0]
    head_dim = width // heads
    x = tokens @ params["in_w"] + params["in_b"] + params["pos"][:, :seq_len, :]
    x = jnp.where(valid[..., None], x, 0.0)

    causal = jnp.tril(jnp.ones((seq_len, seq_len), dtype=bool))[None, None, :, :]
    eye = jnp.eye(seq_len, dtype=bool)[None, None, :, :]
    attn_mask = (causal & valid[:, None, None, :]) | ((~valid)[:, None, :, None] & eye)

    for block in params["blocks"]:
        normed = _layer_norm(x, block["ln1_s"], block["ln1_b"])
        qkv = normed @ block["qkv_w"] + block["qkv_b"]
        q, k, v = jnp.split(qkv, 3, axis=-1)
        q = jnp.transpose(q.reshape(batch, seq_len, heads, head_dim), (0, 2, 1, 3))
        k = jnp.transpose(k.reshape(batch, seq_len, heads, head_dim), (0, 2, 1, 3))
        v = jnp.transpose(v.reshape(batch, seq_len, heads, head_dim), (0, 2, 1, 3))
        scores = jnp.einsum("bhqd,bhkd->bhqk", q, k) * (head_dim ** -0.5)
        scores = jnp.where(attn_mask, scores, -1e9)
        weights = jax.nn.softmax(scores, axis=-1)
        att = jnp.einsum("bhqk,bhkd->bhqd", weights, v)
        att = jnp.transpose(att, (0, 2, 1, 3)).reshape(batch, seq_len, width)
        x = x + (att @ block["out_w"] + block["out_b"])
        x = jnp.where(valid[..., None], x, 0.0)

        ffn_in = _layer_norm(x, block["ln2_s"], block["ln2_b"])
        ffn_h = jax.nn.silu(ffn_in @ block["f1_w"] + block["f1_b"])
        x = x + (ffn_h @ block["f2_w"] + block["f2_b"])
        x = jnp.where(valid[..., None], x, 0.0)

    x = _layer_norm(x, params["final_s"], params["final_b"])
    denom = jnp.maximum(jnp.sum(valid, axis=1, keepdims=True), 1.0)
    pooled = jnp.sum(jnp.where(valid[..., None], x, 0.0), axis=1) / denom
    return pooled @ params["head_w"] + params["head_b"]


def _init_gru(key, in_dim: int, hidden_dim: int, out_dim: int):
    import jax
    import jax.numpy as jnp

    k1, k2, k3 = jax.random.split(key, 3)
    init = jax.nn.initializers.lecun_normal()
    return dict(
        wx=init(k1, (in_dim, 3 * hidden_dim), dtype=jnp.float32),
        wh=init(k2, (hidden_dim, 3 * hidden_dim), dtype=jnp.float32),
        b=jnp.zeros((3 * hidden_dim,), dtype=jnp.float32),
        head=_init_mlp(k3, (hidden_dim, hidden_dim, out_dim)),
    )


def _apply_gru(params, tokens, valid):
    """RMA-style GRU history encoder over valid steps [B, K, D] -> [B, out_dim]."""
    import jax
    import jax.numpy as jnp

    bsz, _, _ = tokens.shape
    h_dim = params["wh"].shape[0]
    h0 = jnp.zeros((bsz, h_dim), dtype=jnp.float32)

    def step_fn(h, inp):
        x_t, v_t = inp
        gx = x_t @ params["wx"] + params["b"]
        gh = h @ params["wh"]
        xz, xr, xn = jnp.split(gx, 3, axis=-1)
        hz, hr, hn = jnp.split(gh, 3, axis=-1)
        z = jax.nn.sigmoid(xz + hz)
        r = jax.nn.sigmoid(xr + hr)
        n = jnp.tanh(xn + r * hn)
        h_new = (1.0 - z) * n + z * h
        h_out = jnp.where(v_t[:, None], h_new, h)
        return h_out, None

    h_final, _ = jax.lax.scan(step_fn, h0, (jnp.swapaxes(tokens, 0, 1), jnp.swapaxes(valid, 0, 1)))
    return _apply_mlp(params["head"], h_final)


def init_reader(key, cfg: ReaderConfig):
    """Initialize parameters for any of the reader architectures."""
    import jax

    k1, k2, k3, k4 = jax.random.split(key, 4)
    vis_ch = {"luminance_flow": 7, "rgb": 8, "hybrid": 10}[cfg.visual_mode]
    if cfg.model_type == "frame0_probe":
        return dict(
            cnn=_init_cnn(k1, 5, cfg.cnn_width),
            mlp=_init_mlp(k2, (cfg.cnn_width + 8, cfg.mlp_width, cfg.mlp_width, 4)),
        )
    if cfg.model_type == "joint_transformer":
        return dict(
            cnn=_init_cnn(k1, vis_ch, cfg.cnn_width),
            tx=_init_transformer(
                k2,
                cfg.cnn_width + 12,
                cfg.max_chunks,
                cfg.transformer_width,
                cfg.transformer_layers,
                4,
            ),
        )
    if cfg.model_type == "rma_gru":
        return dict(
            cnn=_init_cnn(k1, vis_ch, cfg.cnn_width),
            gru=_init_gru(k2, cfg.cnn_width + 12, cfg.mlp_width, 4),
        )
    scale_head = _init_mlp(k3, (5, 64, 64, 2))
    grip_head = _init_mlp(k4, (4, 48, 48, 2))
    if cfg.model_type == "factored_no_equiv":
        return dict(
            cnn=_init_cnn(k1, vis_ch, cfg.cnn_width),
            vis_head=_init_mlp(k2, (cfg.cnn_width + 2, cfg.mlp_width, 3)),
            scale_head=scale_head,
            grip_head=grip_head,
        )
    if cfg.model_type == "factored_equiv":
        return dict(
            cnn=_init_cnn(k1, vis_ch, cfg.cnn_width),
            vis_vec=_init_mlp(k2, (cfg.cnn_width, cfg.mlp_width, 3)),
            scale_head=scale_head,
            grip_head=grip_head,
        )
    raise ValueError(cfg.model_type)


def _normalize_cs(c_raw, s_raw, has_valid):
    import jax.numpy as jnp

    norm = jnp.sqrt(jnp.square(c_raw) + jnp.square(s_raw) + 1e-8)
    cos_hat = jnp.where(has_valid, c_raw / norm, 1.0)
    sin_hat = jnp.where(has_valid, s_raw / norm, 0.0)
    return cos_hat, sin_hat


def _procrustes_yaw(commands, vectors, weights, has_valid):
    """Weighted SO(2) alignment of public command XY to visual motion XY."""
    import jax.numpy as jnp

    ax, ay = commands[..., 0], commands[..., 1]
    vx, vy = vectors[..., 0], vectors[..., 1]
    dot = jnp.sum(weights * (ax * vx + ay * vy), axis=-1)
    cross = jnp.sum(weights * (ax * vy - ay * vx), axis=-1)
    return _normalize_cs(dot, cross, has_valid)


def _vis_inputs(img_before, img_after, visual_mode: str = "rgb"):
    import jax.numpy as jnp

    b, k, h, w, _ = img_before.shape
    coords = jnp.broadcast_to(_coord_grid(h, w)[None, None], (b, k, h, w, 2))
    diff = img_after - img_before
    if visual_mode in ("luminance_flow", "hybrid"):
        luma_b = jnp.mean(img_before, axis=-1, keepdims=True)
        luma_a = jnp.mean(img_after, axis=-1, keepdims=True)
        dt = luma_a - luma_b
        motion_mag = jnp.sqrt(jnp.mean(jnp.square(diff), axis=-1, keepdims=True) + 1e-6)
        avg = 0.5 * (luma_b + luma_a)
        gx = jnp.pad(0.5 * (avg[:, :, :, 2:, :] - avg[:, :, :, :-2, :]), ((0, 0), (0, 0), (0, 0), (1, 1), (0, 0)))
        gy = jnp.pad(0.5 * (avg[:, :, 2:, :, :] - avg[:, :, :-2, :, :]), ((0, 0), (0, 0), (1, 1), (0, 0), (0, 0)))
        if visual_mode == "hybrid":
            x = jnp.concatenate((dt, motion_mag, gx, gy, luma_b, diff, coords), axis=-1)
            return x.reshape(b * k, h, w, 10), b, k
        x = jnp.concatenate((dt, motion_mag, gx, gy, luma_b, coords), axis=-1)
        return x.reshape(b * k, h, w, 7), b, k
    x = jnp.concatenate((diff, img_before, coords), axis=-1)
    return x.reshape(b * k, h, w, 8), b, k


def _unit_rgb(value):
    import jax.numpy as jnp

    image = jnp.asarray(value)
    # Disk tapes use bytes. The motor's online callback already supplies [0, 1].
    return image.astype(jnp.float32) / 255.0 if image.dtype == jnp.uint8 else image.astype(jnp.float32)


def _history_visual_features(params, batch, visual_mode: str = "rgb"):
    import jax.numpy as jnp

    mask = jnp.asarray(batch["valid"], dtype=bool)[..., None, None, None]
    before = jnp.where(mask, _unit_rgb(batch["img_before"]), 0.0)
    after = jnp.where(mask, _unit_rgb(batch["img_after"]), 0.0)
    images, b, k = _vis_inputs(before, after, visual_mode=visual_mode)
    return _apply_cnn(params["cnn"], images).reshape(b, k, -1)


def visual_motion_vectors(params, batch, visual_mode: str = "rgb"):
    """Read the equivariant model's pre-contraction visual XY and confidence logit."""
    return _apply_mlp(params["vis_vec"], _history_visual_features(params, batch, visual_mode=visual_mode))


def _apply_scale_and_grip(params, issued_mean, delta_eef, delta_jaw, valid):
    import jax
    import jax.numpy as jnp

    has_valid = jnp.any(valid, axis=1)

    # Per-axis translation scale head shared across x, y, z.
    iss_xyz = issued_mean[..., :3]
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
    s_out = _apply_mlp(params["scale_head"], scale_feats)
    s_val, s_conf = s_out[..., 0], s_out[..., 1]
    s_w = jnp.where(valid[..., None], jax.nn.softplus(s_conf) + 1e-4, 0.0)
    s_den = jnp.maximum(jnp.sum(s_w, axis=(1, 2)), 1e-6)
    log_scale = jnp.where(has_valid, jnp.sum(s_w * s_val, axis=(1, 2)) / s_den, 0.0)

    # Gripper polarity head.
    iss_g = issued_mean[..., 6:7]
    grip_feats = jnp.concatenate((iss_g, delta_jaw, -iss_g * delta_jaw[..., 0:1]), axis=-1)
    g_out = _apply_mlp(params["grip_head"], grip_feats)
    g_val, g_conf = g_out[..., 0], g_out[..., 1]
    g_w = jnp.where(valid, jax.nn.softplus(g_conf) + 1e-4, 0.0)
    g_den = jnp.maximum(jnp.sum(g_w, axis=1), 1e-6)
    grip_logit = jnp.where(has_valid, jnp.sum(g_w * g_val, axis=1) / g_den, 10.0)

    return log_scale, grip_logit


def apply_reader(params, cfg: ReaderConfig, batch):
    """Run forward pass returning (cos_yaw, sin_yaw, log_scale, grip_logit), each shape [B]."""
    import jax
    import jax.numpy as jnp

    if cfg.model_type == "frame0_probe":
        f0_img = _unit_rgb(batch["frame0_img"])
        f0_state = jnp.asarray(batch["frame0_state"], dtype=jnp.float32)
        b, h, w, _ = f0_img.shape
        coords = jnp.broadcast_to(_coord_grid(h, w)[None], (b, h, w, 2))
        vis = _apply_cnn(params["cnn"], jnp.concatenate((f0_img, coords), axis=-1))
        raw = _apply_mlp(params["mlp"], jnp.concatenate((vis, f0_state), axis=-1))
        cos_hat, sin_hat = _normalize_cs(raw[:, 0], raw[:, 1], jnp.ones((b,), dtype=bool))
        return cos_hat, sin_hat, raw[:, 2], raw[:, 3]

    valid = jnp.asarray(batch["valid"], dtype=bool)
    has_valid = jnp.any(valid, axis=1)

    def signal(field):
        return jnp.where(valid[..., None], jnp.asarray(batch[field], dtype=jnp.float32), 0.0)

    issued = signal("issued_mean")
    # Commands are dimensionless. Feed centimetres to the neural motion heads.
    delta_eef = 100.0 * signal("delta_eef")
    delta_jaw = 100.0 * signal("delta_jaw")
    if cfg.model_type == "joint_transformer":
        vis = _history_visual_features(params, batch, visual_mode=cfg.visual_mode)
        tok = jnp.concatenate((vis, issued, delta_eef, delta_jaw), axis=-1)
        raw = _apply_transformer(params["tx"], tok, valid, cfg.transformer_heads)
        cos_hat, sin_hat = _normalize_cs(raw[:, 0], raw[:, 1], has_valid)
        log_scale = jnp.where(has_valid, raw[:, 2], 0.0)
        grip_logit = jnp.where(has_valid, raw[:, 3], 10.0)
        return cos_hat, sin_hat, log_scale, grip_logit

    if cfg.model_type == "rma_gru":
        vis = _history_visual_features(params, batch, visual_mode=cfg.visual_mode)
        tok = jnp.concatenate((vis, issued, delta_eef, delta_jaw), axis=-1)
        raw = _apply_gru(params["gru"], tok, valid)
        cos_hat, sin_hat = _normalize_cs(raw[:, 0], raw[:, 1], has_valid)
        log_scale = jnp.where(has_valid, raw[:, 2], 0.0)
        grip_logit = jnp.where(has_valid, raw[:, 3], 10.0)
        return cos_hat, sin_hat, log_scale, grip_logit

    log_scale, grip_logit = _apply_scale_and_grip(params, issued, delta_eef, delta_jaw, valid)

    if cfg.model_type == "factored_no_equiv":
        vis = _history_visual_features(params, batch, visual_mode=cfg.visual_mode)
        vis_in_head = jnp.concatenate((vis, issued[..., :2]), axis=-1)
        v_out = _apply_mlp(params["vis_head"], vis_in_head)
        v_w = jnp.where(valid, jax.nn.softplus(v_out[..., 2]) + 1e-4, 0.0)
        c_raw = jnp.sum(v_w * v_out[..., 0], axis=1)
        s_raw = jnp.sum(v_w * v_out[..., 1], axis=1)
        cos_hat, sin_hat = _normalize_cs(c_raw, s_raw, has_valid)
        return cos_hat, sin_hat, log_scale, grip_logit

    if cfg.model_type == "factored_equiv":
        # Purely visual motion vector [vx, vy] + confidence weight; NO action input before Procrustes.
        vec_out = visual_motion_vectors(params, batch, visual_mode=cfg.visual_mode)
        w = jnp.where(valid, jax.nn.softplus(vec_out[..., 2]) + 1e-4, 0.0)
        cos_hat, sin_hat = _procrustes_yaw(issued[..., :2], vec_out[..., :2], w, has_valid)
        return cos_hat, sin_hat, log_scale, grip_logit

    raise ValueError(cfg.model_type)


def predict_embodiment(params, cfg: ReaderConfig, batch):
    """Return decoded (yaw_deg_hat, xy_scale_hat, gripper_sign_hat) as NumPy arrays [B]."""
    cos_h, sin_h, log_scale_h, grip_logit_h = map(np.asarray, apply_reader(params, cfg, batch))
    if not all(np.isfinite(x).all() for x in (cos_h, sin_h, log_scale_h, grip_logit_h)):
        raise FloatingPointError("Nonfinite reader prediction")
    yaw_deg = np.degrees(np.arctan2(np.asarray(sin_h), np.asarray(cos_h))).astype(np.float32)
    xy_scale = np.exp(np.clip(log_scale_h, np.log(0.25), np.log(4.0))).astype(np.float32)
    grip_sign = np.where(np.asarray(grip_logit_h) >= 0.0, 1, -1).astype(np.int32)
    return yaw_deg, xy_scale, grip_sign


def reader_loss(params, cfg: ReaderConfig, batch):
    """Compute joint factor identification loss (cosine yaw + MSE log scale + BCE gripper)."""
    import jax
    import jax.numpy as jnp

    fields = FRAME0_FIELDS if cfg.model_type == "frame0_probe" else HISTORY_FIELDS
    cos_h, sin_h, log_scale_h, grip_logit_h = apply_reader(params, cfg, {k: batch[k] for k in fields})
    yaw_rad = jnp.deg2rad(jnp.asarray(batch["yaw_deg"], dtype=jnp.float32))
    cos_t = jnp.cos(yaw_rad)
    sin_t = jnp.sin(yaw_rad)
    log_scale_t = jnp.log(jnp.asarray(batch["xy_scale"], dtype=jnp.float32))
    grip_t = jnp.asarray(batch["gripper_sign"] > 0, dtype=jnp.float32)

    yaw_loss = jnp.mean(1.0 - (cos_h * cos_t + sin_h * sin_t))
    scale_loss = jnp.mean(jnp.square(log_scale_h - log_scale_t))
    grip_loss = jnp.mean(jax.nn.softplus(grip_logit_h) - grip_t * grip_logit_h)
    total = yaw_loss + scale_loss + grip_loss
    metrics = dict(total=total, yaw_loss=yaw_loss, scale_loss=scale_loss, grip_loss=grip_loss)
    return total, metrics


def make_procrustes_equiv_batch(
    issued_xy: np.ndarray,
    true_yaw_deg: float,
    xy_scale: float = 1.0,
    gripper_sign: int = 1,
):
    """Build deterministic unit-test helper inputs verifying analytical SO(2) Procrustes contraction."""
    rad = math.radians(float(true_yaw_deg))
    r = np.array([[math.cos(rad), -math.sin(rad)], [math.sin(rad), math.cos(rad)]], dtype=np.float32)
    v_world = (issued_xy @ r.T) * float(xy_scale)
    return v_world, float(true_yaw_deg), int(gripper_sign)
