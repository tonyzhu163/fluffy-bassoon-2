"""Bounded synchronization and per-observation decay options of the persistent CTM (audit F2/F3)."""
import pytest
import torch

from fluffy_bassoon.meta_rl.ppo import PersistentCTMVectorCore


def core(**options):
    torch.manual_seed(3)
    return PersistentCTMVectorCore(5, width=8, ticks=4, history_length=3, synapse_depth=2, heads=2, **options).eval()


def roll(model, steps, batch=2):
    torch.manual_seed(4)
    state, features = None, []
    for _ in range(steps):
        output, state = model(torch.randn(batch, 5), state)
        features.append(output)
    return features, state


def test_defaults_keep_the_saved_legacy_formulas():
    expert = core().expert
    with torch.no_grad():
        expert.decay_params_action.uniform_(0, 2)
        expert.decay_params_out.uniform_(0, 2)
    r_action, r_out = expert._synchronisation_decays()
    torch.testing.assert_close(r_action, expert.decay_params_action.neg().exp(), rtol=0, atol=0)
    torch.testing.assert_close(r_out, expert.decay_params_out.neg().exp(), rtol=0, atol=0)
    alpha, beta = torch.randn(3, 16), torch.rand(3, 16) + 1
    torch.testing.assert_close(expert._normalise_synchronisation(alpha, beta), alpha / torch.sqrt(beta), rtol=0, atol=0)


def test_per_observation_decay_is_one_learned_decay_across_all_ticks():
    expert = core(decay_per_observation=True).expert
    with torch.no_grad():
        expert.decay_params_out.uniform_(0, 2)
    _, r_out = expert._synchronisation_decays()
    torch.testing.assert_close(r_out ** expert.num_ticks, expert.decay_params_out.neg().exp())


def test_bounded_readout_is_unit_rms_and_invariant_to_accumulated_mass():
    expert = core(bounded_synchronisation=True).expert
    alpha, beta = torch.randn(3, 16), torch.rand(3, 16) + 1
    bounded = expert._normalise_synchronisation(alpha, beta)
    torch.testing.assert_close(bounded.square().mean(-1), torch.ones(3))
    torch.testing.assert_close(expert._normalise_synchronisation(50 * alpha, 50 * beta), bounded)


def test_explicit_initial_state_matches_the_episode_start():
    model = core()
    observation = torch.randn(2, 5)
    implicit, implicit_state = model(observation, None)
    explicit, explicit_state = model(observation, model.initial_state(2))
    torch.testing.assert_close(explicit, implicit)
    for key in implicit_state:
        torch.testing.assert_close(explicit_state[key], implicit_state[key])


def test_legacy_scale_grows_for_fixed_products_and_bounded_does_not():
    scales = {}
    for bounded in (False, True):
        expert = core(bounded_synchronisation=bounded).expert
        with torch.no_grad():
            expert.decay_params_out.zero_()
        # Arbitrary recurrent activations can cancel; growth is not guaranteed.
        # Fix the pair products to isolate the accumulation/normalization rule.
        activated = torch.ones(2, expert.hidden_dim)
        _, decay = expert._synchronisation_decays()
        state = expert._initial_synchronisation_memory(activated, decay)
        early = expert._output_synchronisation(state).square().mean().sqrt()
        for _ in range(64):
            state = expert._advance_post_activation(activated, state, decay)
        late = expert._output_synchronisation(state).square().mean().sqrt()
        scales[bounded] = float((late / early).detach())
    assert scales[False] == pytest.approx(65 ** .5, rel=1e-5)
    assert scales[True] == pytest.approx(1, abs=1e-4)


@pytest.mark.parametrize("options", [{}, dict(bounded_synchronisation=True, decay_per_observation=True)])
def test_gradients_reach_decays_through_a_carried_state(options):
    model = core(**options).train()
    features, _ = roll(model, 3)
    torch.stack(features).square().sum().backward()
    for name in ("decay_params_action", "decay_params_out"):
        gradient = getattr(model.expert, name).grad
        assert gradient is not None and torch.isfinite(gradient).all() and gradient.abs().sum() > 0


def test_invalid_options_are_rejected():
    with pytest.raises(ValueError):
        PersistentCTMVectorCore(5, width=8, bounded_synchronisation=1)
    with pytest.raises(ValueError):
        PersistentCTMVectorCore(5, width=6, heads=4)
