"""Unit tests for continuous-latent velocity adapter and decoupled causal history encoder."""

import pytest


def test_pack_control_tokens_padding_and_boundary():
    np = pytest.importorskip("numpy")
    from fluffy_bassoon.libero import HistoryEncoderConfig, pack_control_tokens

    cfg = HistoryEncoderConfig(max_support_controls=20, max_prefix_controls=10, controls_per_token=5)
    assert cfg.support_token_count == 4
    assert cfg.prefix_token_count == 2
    assert cfg.token_input_dim == 80

    # Empty trace -> zeros and all False validity.
    empty_tok, empty_val = pack_control_tokens(
        np.zeros((0, 7)),
        np.zeros((0, 3)),
        np.array([0.1, 0.2, 0.3]),
        np.zeros((0, 2)),
        np.zeros((0,)),
        np.zeros((0,)),
        max_controls=20,
        controls_per_token=5,
    )
    assert empty_tok.shape == (4, 80) and empty_tok.dtype == np.float32
    assert not np.any(empty_tok) and not np.any(empty_val)

    # Partial 7-step trace checks delta_pos alignment, boundary bit, and valid mask.
    init_pos = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    positions = init_pos[None] + np.cumsum(np.ones((7, 3), dtype=np.float32) * 0.1, axis=0)
    issued = np.arange(49, dtype=np.float32).reshape(7, 7)
    gripper = np.full((7, 2), 0.5, dtype=np.float32)
    rewards = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    waits = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)

    tokens, valid = pack_control_tokens(
        issued,
        positions,
        init_pos,
        gripper,
        rewards,
        waits,
        max_controls=20,
        controls_per_token=5,
        episode_boundary_at_start=True,
    )
    assert tokens.shape == (4, 80)
    np.testing.assert_array_equal(valid, [True, True, False, False])
    unpacked = tokens.reshape(20, 16)
    np.testing.assert_allclose(unpacked[:7, 7:10], 0.1, atol=1e-6)
    assert unpacked[0, 14] == 1.0 and not np.any(unpacked[1:, 14])
    np.testing.assert_array_equal(unpacked[:7, 15], 1.0)
    assert not np.any(unpacked[7:, 15])

    # Truncation past max_controls.
    long_tok, long_val = pack_control_tokens(
        np.ones((25, 7)),
        np.ones((25, 3)),
        init_pos,
        np.ones((25, 2)),
        np.zeros((25,)),
        np.zeros((25,)),
        max_controls=20,
        controls_per_token=5,
    )
    assert long_tok.shape == (4, 80)
    assert bool(np.all(long_val))


def test_latent_adapter_zero_init_film_and_active_mask():
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    np = pytest.importorskip("numpy")
    from fluffy_bassoon.libero.latent_adapter import (
        AdapterConfig,
        apply_latent_adapter,
        init_latent_adapter,
        lookup_condition_latent,
    )

    for use_film in (True, False):
        cfg = AdapterConfig(feature_dim=12, action_dim=7, latent_dim=6, width=16, num_conditions=4, use_film=use_film)
        params = init_latent_adapter(jax.random.key(11), cfg)
        features = jax.random.normal(jax.random.key(12), (2, 10, 12))
        latents = lookup_condition_latent(params, jnp.array([0, 3]))
        out0 = apply_latent_adapter(params, features, latents)
        assert out0.shape == (2, 10, 7)
        assert np.array_equal(np.asarray(out0), np.zeros((2, 10, 7), dtype=np.float32))

        # Active mask zeroes inactive lane even with non-zero weights and FiLM.
        mutated = {k: v for k, v in params.items()}
        mutated["out"] = jnp.ones_like(params["out"]) * 0.25
        mutated["bias_out"] = jnp.ones_like(params["bias_out"]) * 0.1
        if use_film:
            mutated["film1_w"] = jnp.ones_like(params["film1_w"]) * 0.3
            mutated["film2_w"] = jnp.ones_like(params["film2_w"]) * 0.3
        masked_out = apply_latent_adapter(mutated, features, latents, active=jnp.array([True, False]))
        assert float(jnp.max(jnp.abs(masked_out[0]))) > 0.0
        assert np.array_equal(np.asarray(masked_out[1]), np.zeros((10, 7), dtype=np.float32))

        # Non-zero gradient at initialization on output head.
        target = jnp.ones((2, 10, 7), dtype=jnp.float32)

        def loss_fn(p):
            z = lookup_condition_latent(p, jnp.array([1, 2]))
            return jnp.mean(jnp.square(apply_latent_adapter(p, features, z) - target))

        _, grad = jax.value_and_grad(loss_fn)(params)
        assert float(jnp.linalg.norm(grad["out"])) > 0.0
        assert float(jnp.linalg.norm(grad["bias_out"])) > 0.0


def test_history_encoder_causality_and_ablation_masks():
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    np = pytest.importorskip("numpy")
    from fluffy_bassoon.libero.history_encoder import (
        HistoryEncoderConfig,
        apply_history_encoder,
        decode_steering_latent,
        history_belief_loss,
        init_history_encoder,
    )

    cfg = HistoryEncoderConfig(
        controls_per_token=5,
        max_support_controls=10,
        max_prefix_controls=10,
        query_dim=16,
        width=32,
        layers=2,
        heads=4,
        num_conditions=3,
        latent_dim=4,
    )
    params = init_history_encoder(jax.random.key(41), cfg)
    cond_table = jax.random.normal(jax.random.key(42), (3, 4))

    s_tok = jax.random.normal(jax.random.key(1), (2, 2, cfg.token_input_dim))
    s_val = jnp.array([[True, True], [True, False]])
    p_tok = jax.random.normal(jax.random.key(2), (2, 2, cfg.token_input_dim))
    p_val = jnp.array([[True, False], [False, False]])
    q_feat = jax.random.normal(jax.random.key(3), (2, 16))

    # Invariance to support_tokens when retain_support=False.
    l_nosup_a, d_nosup_a = apply_history_encoder(
        params, cfg, s_tok, s_val, p_tok, p_val, q_feat, retain_support=False, retain_prefix=True
    )
    l_nosup_b, d_nosup_b = apply_history_encoder(
        params, cfg, s_tok * 100.0 + 50.0, s_val, p_tok, p_val, q_feat, retain_support=False, retain_prefix=True
    )
    np.testing.assert_allclose(np.asarray(l_nosup_a), np.asarray(l_nosup_b), atol=1e-6)
    np.testing.assert_allclose(np.asarray(d_nosup_a), np.asarray(d_nosup_b), atol=1e-6)

    # Invariance to prefix_tokens when retain_prefix=False.
    l_nopfx_a, d_nopfx_a = apply_history_encoder(
        params, cfg, s_tok, s_val, p_tok, p_val, q_feat, retain_support=True, retain_prefix=False
    )
    l_nopfx_b, d_nopfx_b = apply_history_encoder(
        params, cfg, s_tok, s_val, -p_tok * 50.0, p_val, q_feat, retain_support=True, retain_prefix=False
    )
    np.testing.assert_allclose(np.asarray(l_nopfx_a), np.asarray(l_nopfx_b), atol=1e-6)
    np.testing.assert_allclose(np.asarray(d_nopfx_a), np.asarray(d_nopfx_b), atol=1e-6)

    # Cleared mode (False, False) and loss gradients are strictly finite with zero NaNs.
    l_clr, d_clr = apply_history_encoder(
        params, cfg, s_tok, s_val, p_tok, p_val, q_feat, retain_support=False, retain_prefix=False
    )
    assert jnp.isfinite(l_clr).all() and jnp.isfinite(d_clr).all()

    batch = dict(
        support_tokens=s_tok,
        support_valid=s_val,
        prefix_tokens=p_tok,
        prefix_valid=p_val,
        query_features=q_feat,
        condition_id=jnp.array([0, 2]),
    )
    (loss, metrics), grad = jax.value_and_grad(
        lambda p: history_belief_loss(p, cfg, batch, cond_table), has_aux=True
    )(params)
    assert jnp.isfinite(loss)
    assert all(jnp.isfinite(v).all() for v in metrics.values())
    assert all(jnp.isfinite(g).all() for g in jax.tree.leaves(grad))

    # Prototype vs soft decoding.
    proto_z = decode_steering_latent(l_nopfx_a, d_nopfx_a, cond_table, mode="prototype")
    soft_z = decode_steering_latent(l_nopfx_a, d_nopfx_a, cond_table, mode="soft")
    np.testing.assert_allclose(np.asarray(proto_z), np.asarray(cond_table[jnp.argmax(l_nopfx_a, axis=-1)]))
    expected_soft = jax.nn.softmax(l_nopfx_a, axis=-1) @ cond_table + d_nopfx_a
    np.testing.assert_allclose(np.asarray(soft_z), np.asarray(expected_soft), atol=1e-6)


def test_counterfactual_margin_loss_behavior():
    pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    from fluffy_bassoon.libero.history_encoder import counterfactual_margin_loss

    targets_a = jnp.array([[1.0, 0.0], [1.0, 0.0]], dtype=jnp.float32)
    targets_b = jnp.array([[0.0, 1.0], [1.0, 0.0]], dtype=jnp.float32)
    # Only row 0 differs in target; choose logits whose log_softmax difference matches [+1, -1].
    logits_a = jnp.array([[1.0, 0.0], [0.5, 0.2]], dtype=jnp.float32)
    logits_b = jnp.array([[0.0, 1.0], [0.5, 0.2]], dtype=jnp.float32)

    matched_loss = float(counterfactual_margin_loss(logits_a, logits_b, targets_a, targets_b))
    ignored_loss = float(counterfactual_margin_loss(logits_a, logits_a, targets_a, targets_b))
    swapped_loss = float(counterfactual_margin_loss(logits_b, logits_a, targets_a, targets_b))
    assert matched_loss < 1e-6
    assert matched_loss < ignored_loss < swapped_loss

    # Masked contrast selection isolates only valid pairs.
    masked_same = float(
        counterfactual_margin_loss(
            logits_b, logits_a, targets_a, targets_b, pair_mask=jnp.array([0.0, 1.0])
        )
    )
    assert masked_same < 1e-6
