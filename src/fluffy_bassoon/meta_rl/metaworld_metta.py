"""METTA paper network on the existing Meta-World collector and checkpoint API.

Architecture/chronological prefix replay are implemented; preprocessing and
unspecified architecture choices remain explicit adaptations. Not a claim to
reproduce METTA's published scores or speed. See docs/metaworld-metta.md.
"""

import math

from gymnasium.spaces.utils import flatdim
import torch
from torch import nn
from torch.distributions import Normal
from torch.utils.checkpoint import checkpoint

from .cores import policy_size
from .metta import MettaMemory
from .metaworld_policy import StateContextCore
from .ppo import RecurrentActorCritic, generalized_advantages


DEFAULT_SETTINGS = dict(width=128, layers=2, chunk_length=128, expansion=4,
                        max_episodes=128, aggregation="attention", ema_alpha=.5,
                        base_lr=1., online_updates=True, checkpoint_length=32)


class MettaObservationMemory(nn.Module):
    def __init__(self, space, settings):
        super().__init__()
        self.input_dim, self.ticks = flatdim(space), 1
        self.fields = {}
        offset = 0
        for name, field in space.spaces.items():
            self.fields[name] = slice(offset, offset + flatdim(field))
            offset += flatdim(field)
        if not {"state", "previous_action", "previous_reward", "attempt_end"} <= self.fields.keys():
            raise ValueError("METTA needs state/action/reward/boundary observations")
        size = sum(flatdim(space[key]) for key in ("state", "previous_action", "previous_reward", "attempt_end"))
        self.memory = MettaMemory(size, **{k:v for k,v in settings.items() if k != "checkpoint_length"})
        self.output_dim = self.memory.output_dim

    def initial_state(self, batch):
        return self.memory.initial_state(batch)

    def forward(self, observation, state=None):
        if state is None:
            state = self.initial_state(observation.shape[0])
        boundary = observation[:, self.fields["attempt_end"]].squeeze(-1) > .5
        # Algorithms 1/2 reset previous action/reward/done at each new episode.
        # Boundary is kept separately for committing summaries; clocks are absent.
        beginning = boundary | ~state["active"]
        current = observation[:, self.fields["state"]]
        feedback = torch.cat([observation[:, self.fields[key]] for key in
                              ("previous_action", "previous_reward", "attempt_end")], -1)
        feedback = torch.where(beginning[:, None], torch.zeros_like(feedback), feedback)
        return self.memory(torch.cat((current, feedback), -1), state, episode_start=boundary)


class MettaActorCritic(RecurrentActorCritic):
    def __init__(self, env, settings=None):
        settings = dict(DEFAULT_SETTINGS if settings is None else settings)
        memory = MettaObservationMemory(env.observation_space, settings)
        core = StateContextCore(memory, state_dim=flatdim(env.observation_space["state"]))
        super().__init__(core, env.action_space)
        if self.discrete:
            raise ValueError("Meta-World METTA requires continuous actions")
        def head(outputs):
            return nn.Sequential(nn.Linear(core.output_dim, 64), nn.Tanh(), nn.Linear(64, 64),
                                 nn.Tanh(), nn.Linear(64, outputs))
        self.actor, self.critic = head(env.action_space.shape[0]), head(1)
        nn.init.orthogonal_(self.actor[-1].weight, gain=.01)
        nn.init.zeros_(self.actor[-1].bias)
        self.log_std.data.zero_()
        self.checkpoint_length = settings.get("checkpoint_length", 32)
        if type(self.checkpoint_length) is not int or self.checkpoint_length < 0:
            raise ValueError("Activation checkpoint length must be a nonnegative integer")

    def _action_and_value(self, features, raw_action=None, *, deterministic=False, bounded_entropy=False):
        if bounded_entropy:
            raise ValueError("METTA's adapter uses an unsquashed Gaussian likelihood")
        mean, value = self.actor(features), self.critic(features).squeeze(-1)
        distribution = Normal(mean, self.log_std.exp().clamp(.1, 1.5))
        if raw_action is None:
            raw_action = mean if deterministic else distribution.sample()
        return (raw_action.clamp(-1, 1), raw_action, distribution.log_prob(raw_action).sum(-1),
                distribution.entropy().sum(-1), value)

    def evaluate_sequence(self, observations, raw_actions, resets=None, *, bounded_entropy=False,
                          initial_state=None):
        if observations.ndim != 3 or not observations.shape[0]:
            raise ValueError("Require a nonempty [time,batch,features] sequence")
        if resets is None:
            resets = torch.zeros(observations.shape[:2], dtype=torch.bool, device=observations.device)
        if (resets.shape != observations.shape[:2] or resets.dtype != torch.bool
                or resets.device != observations.device):
            raise ValueError("Require boolean resets matching time and batch")
        state, features = initial_state, []
        block_size = self.checkpoint_length or observations.shape[0]
        for start in range(0, observations.shape[0], block_size):
            block = observations[start:start + block_size]
            flags = resets[start:start + block_size]
            def scan(inputs, masks, carried):
                hidden = []
                # This function has no closure over loop indices or mutable state:
                # checkpoint recomputation must replay the identical block.
                any_reset = masks.any(1).tolist()
                for i in range(inputs.shape[0]):
                    h, carried = self._features(inputs[i], carried, masks[i] if any_reset[i] else None)
                    hidden.append(h)
                return torch.stack(hidden), carried
            if self.checkpoint_length and torch.is_grad_enabled():
                hidden, state = checkpoint(scan, block, flags, state, use_reentrant=False, preserve_rng_state=False)
            else:
                hidden, state = scan(block, flags, state)
            features.append(hidden)
        _, _, log_prob, entropy, values = self._action_and_value(torch.cat(features), raw_actions,
                                                               bounded_entropy=bounded_entropy)
        return log_prob, entropy, values


def make_policy(settings, env):
    return MettaActorCritic(env, settings)


def match_policy(settings, env, budget):
    # Preserve the paper's width/depth rather than silently shrinking to CTM's cap.
    return dict(settings), policy_size(make_policy(settings, env))


def update_metta_ppo(policy, optimizer, rollout, *, epochs=10, minibatch_trials=10, clip=.2,
                     gamma=.99, gae_lambda=.95, entropy_coef=0., value_coef=.5,
                     max_grad_norm=.5, normalize_advantages=True, replay_chunk_length=0):
    """Appendix B.1: one optimizer step per episode segment and lane minibatch.

    Every step reconstructs the COMPLETE prefix with current slow parameters.
    Loss applies only to the current episode, while gradients may reach all
    preceding episodes and inner updates. TTT chunks do not truncate gradients.
    """
    if (replay_chunk_length != 0 or type(epochs) is not int or epochs < 1
            or type(minibatch_trials) is not int or minibatch_trials < 1
            or not 0 < clip < 1 or not math.isfinite(max_grad_norm) or max_grad_norm <= 0
            or not math.isfinite(value_coef) or value_coef < 0
            or not math.isfinite(entropy_coef) or entropy_coef < 0):
        raise ValueError("Invalid METTA PPO settings; cached detached replay is not supported")
    ends = rollout["attempt_ends"]
    if ends.dtype != torch.bool or not torch.equal(ends, ends[:, :1].expand_as(ends)):
        raise ValueError("Episode-segment PPO needs aligned fixed-length episode boundaries")
    segment_ends = (ends[:, 0].nonzero().flatten() + 1).tolist()
    if not segment_ends or segment_ends[-1] != ends.shape[0]:
        raise ValueError("Require complete trials ending at an episode boundary")
    advantage, target = generalized_advantages(rollout["rewards"], rollout["values"],
                                               rollout["trial_ends"], gamma=gamma, gae_lambda=gae_lambda)
    mask = rollout.get("action_mask", torch.ones_like(advantage, dtype=torch.bool))
    if mask.shape != advantage.shape or mask.dtype != torch.bool or not mask.any(0).all():
        raise ValueError("Require valid action masks for every trial")
    if normalize_advantages:
        active = advantage[mask]
        advantage = (advantage - active.mean()) / (active.std(unbiased=False) + 1e-8)
    totals, count, steps, prefix_samples = {}, 0, 0, 0
    policy.eval()
    for _ in range(epochs):
        order = torch.randperm(ends.shape[1], device=ends.device)
        for offset in range(0, ends.shape[1], minibatch_trials):
            lanes = order[offset:offset + minibatch_trials]
            start = 0
            for stop in segment_ends:
                log_prob, entropy, value = policy.evaluate_sequence(rollout["observations"][:stop, lanes],
                    rollout["raw_actions"][:stop, lanes], rollout["resets"][:stop, lanes])
                valid = mask[start:stop, lanes]
                log_ratio = (log_prob[start:stop] - rollout["log_probs"][start:stop, lanes])[valid]
                ratio = log_ratio.exp(); adv = advantage[start:stop, lanes][valid]
                actor = torch.maximum(-adv * ratio, -adv * ratio.clamp(1-clip, 1+clip)).mean()
                v, old, ret = value[start:stop][valid], rollout["values"][start:stop, lanes][valid], target[start:stop, lanes][valid]
                clipped_value = old + (v - old).clamp(-clip, clip)
                critic = .5 * torch.maximum((v-ret).square(), (clipped_value-ret).square()).mean()
                ent = entropy[start:stop][valid].mean()
                loss = actor + value_coef * critic - entropy_coef * ent
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite METTA PPO loss")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                grad = nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm, error_if_nonfinite=True)
                optimizer.step()
                weight = int(valid.sum())
                metrics = dict(loss=loss.detach(), actor_loss=actor.detach(), critic_loss=critic.detach(),
                    exploration_entropy=ent.detach(), approximate_kl=((ratio-1)-log_ratio).detach().mean(),
                    clip_fraction=((ratio-1).abs()>clip).float().mean(), gradient_norm=grad)
                for key, metric in metrics.items():
                    totals[key] = totals.get(key, 0.) + float(metric) * weight
                count += weight; steps += 1; prefix_samples += stop * len(lanes)
                start = stop
    return {**{key:v/count for key,v in totals.items()}, "optimizer_steps": steps,
            "forward_samples": count, "active_action_samples": count,
            "reconstructed_prefix_samples": prefix_samples}


def contract(settings):
    return dict(display_name="METTA paper network (independent implementation)",
        paper="https://arxiv.org/html/2610.31656v1", parameter_matching=False,
        observation="normalized state + previous action/reward/boundary; no public clocks; feedback cleared at episode start",
        normalization="common running state moments frozen during collection/replay; fixed reward scaling (not paper running-return scaling)",
        actor_critic="two-layer residual GELU TTT-MLP stack; episode-summary attention; separate current-state bypass and tanh64,64 actor/critic",
        action_distribution="unsquashed Gaussian std[.1,1.5]; save latent likelihood, execute clipped[-1,1] command",
        replay="fresh full prefixes per chronological episode segment; gradients through fast updates and summaries; no detached behavior state",
        implementation="paper-based architecture and prefix optimizer; unspecified details in docs/metaworld-metta.md; published convergence not reproduced")


def main(argv=None):
    from .metaworld_train import main as train
    return train(argv, learner=dict(core_name="metta", name="metta-paper-network", settings=DEFAULT_SETTINGS,
        match=match_policy, make=make_policy, update=update_metta_ppo, contract=contract,
        protocol="metaworld_metta_paper_prefix_ppo_v1", defaults=dict(replay_chunk_length=0)))


if __name__ == "__main__":
    main()
