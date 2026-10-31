"""RL²-PPO: garage's recurrent baseline adapted to the common ML10 v3 protocol.

This is a source-informed PyTorch implementation, not original RL²-TRPO and
not a renamed instance of the repository's parameter-matched GRU control.
See docs/metaworld-rl2.md for the pinned reference and scientific departures.
"""

import argparse
import math
import sys

import gymnasium as gym
from gymnasium.spaces.utils import flatdim
import torch
from torch import nn
from torch.distributions import Normal

from .cores import policy_size
from .ppo import RecurrentActorCritic, generalized_advantages


GARAGE_COMMIT = "2d594803636e341660cab0e81343abbe9a325353"
GARAGE_SOURCE = f"https://github.com/rlworkgroup/garage/tree/{GARAGE_COMMIT}"
METHOD = "rl2-ppo-garage-adapted"
PROTOCOL = "metaworld_rl2_ppo_garage_adapted_v1"


class RL2Core(nn.Module):
    """One GRU transition per observation; physical resets leave memory intact.

    Raw observations accompany hidden features solely for the fitted value
    baseline. The actor reads only the GRU output, with no state bypass.
    """

    def __init__(self, input_dim, width=64):
        super().__init__()
        if any(type(x) is not int or x < 1 for x in (input_dim, width)):
            raise ValueError("RL² requires positive input and hidden dimensions")
        self.input_dim, self.width, self.ticks = input_dim, width, 1
        self.output_dim = input_dim + width
        self.cell = nn.GRU(input_dim, width)
        # garage uses Keras' reset_after GRU, sigmoid gates, tanh candidate,
        # Glorot input/recurrent matrices and zero biases. PyTorch's r,z,n gate
        # ordering differs, but its equations have the same reset placement.
        nn.init.xavier_uniform_(self.cell.weight_ih_l0)
        nn.init.xavier_uniform_(self.cell.weight_hh_l0)
        nn.init.zeros_(self.cell.bias_ih_l0)
        nn.init.zeros_(self.cell.bias_hh_l0)

    def initial_state(self, batch):
        return self.cell.weight_hh_l0.new_zeros(batch, self.width)

    def forward(self, observation, state=None):
        if state is None:
            state = self.initial_state(observation.shape[0])
        _, state = self.cell(observation[None], state[None])
        state = state[0]
        return torch.cat((observation, state), -1), state

    def observe_after_update(self, observations):
        # Deliberately no running observation normalization in this baseline.
        pass


class HiddenReadout(nn.Linear):
    def __init__(self, input_dim, width, action_dim):
        super().__init__(width, action_dim)
        self.input_dim = input_dim
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, features):
        return super().forward(features[..., self.input_dim:])


class LinearFeatureBaseline(nn.Module):
    """Checkpointed ridge regression on observation, squares and trial time.

    Time is recovered from the common wrapper's normalized public clocks,
    rather than a sequence-local arange, so cached-window replay is consistent.
    Coefficients and the regularizer actually used are both checkpoint buffers.
    """

    def __init__(self, input_dim, attempts, horizon, reg_coeff=1e-5):
        super().__init__()
        self.input_dim, self.attempts, self.horizon = input_dim, attempts, horizon
        if (not math.isfinite(reg_coeff) or reg_coeff <= 0
                or any(type(x) is not int or x < 1 for x in (input_dim, attempts, horizon))):
            raise ValueError("Invalid fitted RL² value baseline")
        self.register_buffer("coefficients", torch.zeros(2 * input_dim + 4, dtype=torch.float64))
        self.register_buffer("reg_coeff", torch.tensor(reg_coeff, dtype=torch.float64))
        self.register_buffer("last_reg_coeff", torch.tensor(reg_coeff, dtype=torch.float64))
        self.register_buffer("fit_count", torch.tensor(0, dtype=torch.int64))

    def features(self, observations):
        obs = observations[..., :self.input_dim].double()
        clipped = obs.clamp(-10, 10)
        # Dict flatten order: attempt_end, clock[episode,step], action, reward, state.
        time = ((obs[..., 1:2] * max(self.attempts - 1, 1) + obs[..., 2:3])
                * self.horizon / 100.)
        return torch.cat((clipped, clipped.square(), time, time.square(), time.pow(3),
                          torch.ones_like(time)), -1)

    @torch.no_grad()
    def fit(self, observations, returns):
        if observations.shape[:-1] != returns.shape or not torch.isfinite(returns).all():
            raise ValueError("Fit the linear baseline on matching finite trial returns")
        features = self.features(observations).reshape(-1, self.coefficients.numel())
        targets = returns.reshape(-1).double()
        gram, rhs = features.T @ features, features.T @ targets
        identity = torch.eye(gram.shape[0], dtype=gram.dtype, device=gram.device)
        regularizer = self.reg_coeff
        for _ in range(5):
            try:
                coefficients = torch.linalg.solve(gram + regularizer * identity, rhs)
            except torch.linalg.LinAlgError:
                regularizer = regularizer * 10
                continue
            if torch.isfinite(coefficients).all():
                self.coefficients.copy_(coefficients)
                self.last_reg_coeff.copy_(regularizer)
                self.fit_count.add_(1)
                return
            regularizer = regularizer * 10
        raise FloatingPointError("RL² linear baseline ridge regression failed")

    @torch.no_grad()
    def forward(self, features):
        return (self.features(features) @ self.coefficients).to(features.dtype).unsqueeze(-1)


class RL2Policy(RecurrentActorCritic):
    def __init__(self, env, width=64, baseline_reg_coeff=1e-5):
        nn.Module.__init__(self)
        keys = tuple(env.observation_space.spaces) if isinstance(env.observation_space, gym.spaces.Dict) else ()
        if (keys != ("attempt_end", "clock", "previous_action", "previous_reward", "state")
                or flatdim(env.observation_space) != 47 or env.observation_space["state"].shape != (39,)
                or not isinstance(env.action_space, gym.spaces.Box) or env.action_space.shape != (4,)
                or not (env.action_space.low == -1).all() or not (env.action_space.high == 1).all()):
            raise ValueError("RL² adapter requires the common 39D-state Meta-World trial interface")
        self.discrete = False
        self.core = RL2Core(47, width)
        self.actor = HiddenReadout(47, width, 4)
        self.critic = LinearFeatureBaseline(47, env.attempts, env.horizon, baseline_reg_coeff)
        self.log_std = nn.Parameter(torch.zeros(4))

    def _action_and_value(self, features, raw_action=None, *, deterministic=False, bounded_entropy=False):
        if bounded_entropy:
            raise ValueError("RL² uses an unclipped Gaussian likelihood, not squashed-Gaussian entropy")
        mean = self.actor(features)
        distribution = Normal(mean, self.log_std.exp())
        if raw_action is None:
            raw_action = mean if deterministic else distribution.sample()
        # The latent Gaussian draw defines the policy likelihood. The simulator
        # receives its clipped command, also used in the next feedback token.
        return (raw_action.clamp(-1, 1), raw_action, distribution.log_prob(raw_action).sum(-1),
                distribution.entropy().sum(-1), self.critic(features).squeeze(-1))

    def evaluate_sequence(self, observations, raw_actions, resets=None, *, bounded_entropy=False,
                          initial_state=None):
        if resets is not None and (resets.shape != observations.shape[:2] or resets.dtype != torch.bool
                                   or resets.device != observations.device):
            raise ValueError("Reset mask must be boolean [time, batch] on the observation device")
        # CuDNN requires its training reserve buffer for backward even though
        # this one-layer, zero-dropout GRU has identical train/eval semantics.
        self.core.cell.train(torch.is_grad_enabled())
        if resets is not None and resets.any():
            return super().evaluate_sequence(observations, raw_actions, resets,
                bounded_entropy=bounded_entropy, initial_state=initial_state)
        # Unlike CTM, this reference recurrence supports an exact fused GRU
        # scan. Give RL² its native fast full-trial replay, using precisely the
        # same recurrent weights as one-step behavior collection.
        hidden, _ = self.core.cell(observations,
            None if initial_state is None else initial_state[None])
        features = torch.cat((observations, hidden), -1)
        _, _, log_prob, entropy, values = self._action_and_value(features, raw_actions,
                                                               bounded_entropy=bounded_entropy)
        return log_prob, entropy, values


def make_rl2_policy(config, env):
    expected = {"width", "baseline_reg_coeff", "parameter_matching"}
    if set(config) != expected or type(config["parameter_matching"]) is not bool:
        raise ValueError("Unknown or incomplete RL² policy configuration")
    return RL2Policy(env, config["width"], config["baseline_reg_coeff"])


def match_rl2_policy(settings, env, budget=250_000):
    """Source width64 by default; budget matching is an explicitly labeled option."""
    if (set(settings) != {"width", "baseline_reg_coeff", "parameter_matching"}
            or type(settings["width"]) is not int or settings["width"] < 1
            or type(settings["parameter_matching"]) is not bool
            or type(budget) is not int or budget < 1):
        raise ValueError("Invalid RL² learner settings")
    config = dict(settings)
    if config["parameter_matching"]:
        # GRU weights/biases + direct mean readout + learned constant log_std.
        def parameters(width):
            return 3 * width * (47 + width + 2) + 4 * width + 8
        widths = [w for w in range(16, 1025, 4) if parameters(w) <= budget]
        if not widths or parameters(widths[-1]) < .85 * budget or widths[-1] == 1024:
            raise ValueError("Cannot match RL² within 85–100% of the requested parameter cap")
        config["width"] = widths[-1]
    policy = make_rl2_policy(config, env)
    return config, dict(policy_size(policy), parameter_cap_enforced=config["parameter_matching"])


def _advantages(policy, rewards, values, ends, *, gamma, gae_lambda, entropy_coef):
    # The reference's maximum-entropy objective recomputes current entropy,
    # while stopping its gradient. Constant state-independent std makes this
    # independent of recurrent replay and exact for every chronological trial.
    entropy = Normal(torch.zeros_like(policy.log_std), policy.log_std.exp()).entropy().sum().detach()
    return generalized_advantages(rewards + entropy_coef * entropy, values, ends,
                                  gamma=gamma, gae_lambda=gae_lambda)[0]


def update_rl2_ppo(policy, optimizer, rollout, *, epochs=10, minibatch_trials=32,
                   clip=.2, gamma=.99, gae_lambda=.95, entropy_coef=.02,
                   value_coef=0., max_grad_norm=None, normalize_advantages=False,
                   replay_chunk_length=0):
    """Fit MC baseline, then actor-only full-trial clipped PPO with dense entropy.

    Full-trial BPTT is required here. This avoids silently importing a truncated
    gradient or episode-local return estimator into the named source baseline.
    """
    if (not isinstance(policy, RL2Policy) or type(epochs) is not int or epochs < 1
            or type(minibatch_trials) is not int or minibatch_trials < 1
            or not 0 < clip < 1 or not 0 <= gamma <= 1 or not 0 <= gae_lambda <= 1
            or not math.isfinite(entropy_coef) or entropy_coef < 0 or value_coef != 0
            or normalize_advantages is not False or replay_chunk_length != 0
            or max_grad_norm is not None):
        raise ValueError("RL² source adaptation requires full-trial actor-only PPO without clipping gradients or centering advantages")
    rewards, ends = rollout["rewards"], rollout["trial_ends"]
    mask = rollout.get("action_mask", torch.ones_like(ends))
    if not mask.all() or not ends[-1].all() or ends[:-1].any() or rollout["resets"].any():
        raise ValueError("RL² update requires complete unlesioned fixed-length trials")
    with torch.no_grad():
        entropy = Normal(torch.zeros_like(policy.log_std), policy.log_std.exp()).entropy().sum()
        _, returns = generalized_advantages(rewards + entropy_coef * entropy,
            torch.zeros_like(rewards), ends, gamma=gamma, gae_lambda=1.)
        policy.critic.fit(rollout["observations"], returns)
        baseline = policy.critic(rollout["observations"]).squeeze(-1)
        base_advantages = generalized_advantages(rewards, baseline, ends,
            gamma=gamma, gae_lambda=gae_lambda)[0]
        entropy_weights = generalized_advantages(torch.ones_like(rewards), torch.zeros_like(rewards),
            ends, gamma=gamma, gae_lambda=gae_lambda)[0]
        variance = returns.var(unbiased=False)
        explained_variance = 1 - (returns - baseline).var(unbiased=False) / variance.clamp_min(1e-8)
    policy.eval()
    batch = rewards.shape[1]
    totals, samples, optimizer_steps = {}, 0, 0
    for _ in range(epochs):
        order = torch.randperm(batch, device=rewards.device)
        for start in range(0, batch, minibatch_trials):
            lanes = order[start:start + minibatch_trials]
            log_prob, entropy, _ = policy.evaluate_sequence(rollout["observations"][:, lanes],
                rollout["raw_actions"][:, lanes], rollout["resets"][:, lanes])
            # GAE is linear in rewards. These precomputed unit-reward weights
            # exactly reproduce current-entropy GAE without a repeated temporal
            # scan. Std, hence entropy, is independent of observation/history.
            advantage = base_advantages[:, lanes] + entropy_coef * entropy.detach() * entropy_weights[:, lanes]
            log_ratio = log_prob - rollout["log_probs"][:, lanes]
            ratio = log_ratio.exp()
            loss = torch.maximum(-advantage * ratio, -advantage * ratio.clamp(1 - clip, 1 + clip)).mean()
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite RL² PPO objective")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            # Measure gradients without changing garage's unclipped update.
            gradient_norm = torch.linalg.vector_norm(torch.stack([
                p.grad.norm() for p in policy.parameters() if p.grad is not None]))
            if not torch.isfinite(gradient_norm):
                raise FloatingPointError("Nonfinite RL² PPO gradients")
            optimizer.step()
            weight = advantage.numel()
            metrics = dict(loss=loss.detach(), actor_loss=loss.detach(),
                exploration_entropy=entropy.detach().mean(), gradient_norm=gradient_norm,
                approximate_kl=((ratio - 1) - log_ratio).detach().mean(),
                clip_fraction=((ratio - 1).abs() > clip).float().mean())
            for key, value in metrics.items():
                totals[key] = totals.get(key, 0.) + float(value) * weight
            samples += weight
            optimizer_steps += 1
    return dict({key: value / samples for key, value in totals.items()},
        optimizer_steps=optimizer_steps, forward_samples=samples, active_action_samples=samples,
        baseline_explained_variance=float(explained_variance),
        baseline_reg_coeff=float(policy.critic.last_reg_coeff), baseline_fits=int(policy.critic.fit_count))


def method_contract(config):
    return dict(method=METHOD, display_name="RL²-PPO (garage adapted, GRU" + str(config["width"]) + ")",
        entrypoint="fluffy_bassoon.meta_rl.metaworld_rl2",
        reference=dict(repository="https://github.com/rlworkgroup/garage", commit=GARAGE_COMMIT,
                       license="MIT", example="src/garage/examples/tf/rl2_ppo_metaworld_ml10.py",
                       paper="https://arxiv.org/abs/1611.02779"),
        normalization="none; public clocks retain the common fixed normalization",
        reward="native dense reward multiplied by 0.1 for feedback; optimizer adds 0.02 times stopped Gaussian entropy; success scored separately",
        actor_critic="GRU with direct linear mean; fitted ridge linear-feature baseline; no shared neural critic",
        action_distribution="unsquashed diagonal Gaussian; save preclip draw/likelihood; execute clipped [-1,1] command",
        initialization="Glorot input/recurrent/readout weights, zero biases/state, learned constant log_std initially zero",
        original_algorithm="RL² paper uses TRPO; this is an explicitly named PPO adaptation",
        parameter_matching=config["parameter_matching"],
        baseline=dict(features="clipped observations, squares, trial t/100 and powers 2,3, constant",
                      regularizer=config["baseline_reg_coeff"], fit="discounted entropy-augmented trial returns before PPO",
                      solver="float64 ridge normal equations; retry regularizer x10 up to five attempts"),
        source_departures=[
            "Meta-World 3.1.1 ML10 goal-visible 39D state; five 500-step episodes; common per-episode instance resampling",
            "Common public clocks and previous executed clipped action; terminal feedback survives physical reset",
            "Native reward multiplied by 0.1; source example uses its environment reward unchanged",
            "PyTorch implementation; no TensorFlow/garage runtime and no claimed seedwise numerical equivalence",
            "Fixed requested PPO epochs; source optimizer also has loss-change early stopping at tolerance 1e-6",
            "Common deterministic all-episode success evaluation, raw lane outcomes and measured GPU-hour accounting",
            "Optional parameter matching changes reference GRU64 width and must be labeled separately"])


def main(argv=None):
    extras = argparse.ArgumentParser(add_help=False)
    extras.add_argument("--rl2-width", type=int, default=64)
    extras.add_argument("--rl2-match-budget", action="store_true")
    extras.add_argument("--benchmark", choices=("ml10",), default="ml10")
    extras.add_argument("--replay-chunk-length", type=int, default=0)
    args, forwarded = extras.parse_known_args(sys.argv[1:] if argv is None else argv)
    if args.rl2_width < 1 or args.replay_chunk_length != 0:
        raise ValueError("RL² requires a positive width and full-trial replay (--replay-chunk-length 0)")
    if any(x == "--core" or x.startswith("--core=") for x in forwarded):
        raise ValueError("The RL² entry point selects its own learner")
    learner = dict(name=METHOD,
        protocol=PROTOCOL,
        settings=dict(width=args.rl2_width, baseline_reg_coeff=1e-5, parameter_matching=args.rl2_match_budget),
        make=make_rl2_policy, match=match_rl2_policy, contract=method_contract, update=update_rl2_ppo,
        optimizer=dict(lr=1e-3, eps=1e-8),
        defaults=dict(ppo_epochs=10, minibatch_trials=32),
        optimization=dict(entropy_coef=.02, value_coef=0., normalize_advantages=False, max_grad_norm=None))
    from .metaworld_train import main as train_main
    return train_main(["--core", "gru", "--benchmark", args.benchmark, *forwarded], learner=learner)


if __name__ == "__main__":
    main()
