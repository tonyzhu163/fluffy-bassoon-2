"""Continuous-latent flow-velocity adapter around a frozen π0.5 action expert."""

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class AdapterConfig:
    """Architecture and initialization hyperparameters for the velocity adapter."""

    feature_dim: int
    action_dim: int = 7
    latent_dim: int = 16
    width: int = 192
    num_conditions: int = 4
    use_film: bool = True
    condition_init_scale: float = 0.25

    def __post_init__(self):
        int_fields = (
            ("feature_dim", self.feature_dim),
            ("action_dim", self.action_dim),
            ("latent_dim", self.latent_dim),
            ("width", self.width),
            ("num_conditions", self.num_conditions),
        )
        for name, value in int_fields:
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer, got {value!r}")
        if type(self.use_film) is not bool:
            raise ValueError(f"use_film must be a bool, got {self.use_film!r}")
        if (
            isinstance(self.condition_init_scale, bool)
            or not isinstance(self.condition_init_scale, (int, float))
            or not math.isfinite(self.condition_init_scale)
            or self.condition_init_scale <= 0.0
        ):
            raise ValueError(
                f"condition_init_scale must be finite and positive, got {self.condition_init_scale!r}"
            )


def _film_modulate(hidden, latent, weight, bias):
    import jax.numpy as jnp

    gamma, beta = jnp.split(latent @ weight + bias, 2, axis=-1)
    return hidden * (1.0 + gamma[:, None, :]) + beta[:, None, :]


def init_latent_adapter(key, cfg):
    """Initialize condition prototypes, SiLU MLP, optional zero-FiLM, and zero output head."""
    import jax
    import jax.numpy as jnp

    k_cond, k1, k2 = jax.random.split(key, 3)
    init = jax.nn.initializers.lecun_normal()
    params = dict(
        condition_table=init(k_cond, (cfg.num_conditions, cfg.latent_dim), dtype=jnp.float32)
        * jnp.asarray(cfg.condition_init_scale, dtype=jnp.float32),
        hidden1=init(k1, (cfg.feature_dim + cfg.latent_dim, cfg.width), dtype=jnp.float32),
        bias1=jnp.zeros((cfg.width,), dtype=jnp.float32),
        hidden2=init(k2, (cfg.width, cfg.width), dtype=jnp.float32),
        bias2=jnp.zeros((cfg.width,), dtype=jnp.float32),
        out=jnp.zeros((cfg.width, cfg.action_dim), dtype=jnp.float32),
        bias_out=jnp.zeros((cfg.action_dim,), dtype=jnp.float32),
    )
    if cfg.use_film:
        params.update(
            film1_w=jnp.zeros((cfg.latent_dim, 2 * cfg.width), dtype=jnp.float32),
            film1_b=jnp.zeros((2 * cfg.width,), dtype=jnp.float32),
            film2_w=jnp.zeros((cfg.latent_dim, 2 * cfg.width), dtype=jnp.float32),
            film2_b=jnp.zeros((2 * cfg.width,), dtype=jnp.float32),
        )
    return params


def lookup_condition_latent(params, condition_ids, *, drop_mask=None):
    """Look up condition prototypes [B, D_z], optionally zeroing dropped entries."""
    import jax.numpy as jnp

    ids = jnp.asarray(condition_ids, dtype=jnp.int32)
    latents = params["condition_table"][ids]
    if drop_mask is not None:
        mask = jnp.asarray(drop_mask, dtype=bool)
        latents = jnp.where(mask[:, None], 0.0, latents)
    return latents


def apply_latent_adapter(params, features, latent, *, active=None):
    """Compute velocity residual [B, H, action_dim] from expert features and latent."""
    import jax
    import jax.numpy as jnp

    z = jnp.broadcast_to(latent[:, None, :], (*features.shape[:2], latent.shape[-1]))
    inputs = jnp.concatenate((features, z), axis=-1)
    h1 = jax.nn.silu(inputs @ params["hidden1"] + params["bias1"])
    if "film1_w" in params:
        h1 = _film_modulate(h1, latent, params["film1_w"], params["film1_b"])
    h2 = jax.nn.silu(h1 @ params["hidden2"] + params["bias2"])
    if "film2_w" in params:
        h2 = _film_modulate(h2, latent, params["film2_w"], params["film2_b"])
    residual = h2 @ params["out"] + params["bias_out"]
    if active is not None:
        active_mask = jnp.asarray(active, dtype=bool)[:, None, None]
        residual = jnp.where(active_mask, residual, 0.0)
    return residual


def latent_norm_penalty(params):
    """Return mean squared prototype norm for mild latent table regularization."""
    import jax.numpy as jnp

    return jnp.mean(jnp.square(params["condition_table"]))
