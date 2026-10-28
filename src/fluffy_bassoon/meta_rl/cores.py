"""One construction path for training, budget matching and checkpoint evaluation."""

from gymnasium.spaces.utils import flatdim

from .cmm import CMMCore
from .ppo import (GRUCore, LoopedTransformerCore, PersistentCTMVectorCore,
                  RecurrentActorCritic)
from .transformer_xl import TransformerXLCore


CORE_NAMES = ("gru", "ctm", "persistent_ctm", "cmm", "looped_transformer", "transformer_xl")


def core_config_from_args(args):
    config = dict(width=args.width, ticks=args.ticks)
    if args.core == "transformer_xl":
        config.update(memory_length=args.transformer_memory_length,
                      layers=args.transformer_layers, heads=4)
    elif args.core == "looped_transformer":
        config.update(memory_tokens=args.looped_memory_tokens)
    elif args.core == "persistent_ctm":
        config.update(history_length=args.stm_length, synapse_depth=args.synapse_depth,
                      bounded_synchronisation=args.bounded_synchronisation,
                      decay_per_observation=args.decay_per_observation)
    elif args.core != "gru":
        config.update(stm_length=args.stm_length, ltm_tokens=args.ltm_tokens,
                      synapse_depth=args.synapse_depth)
        if args.core == "ctm":
            config.update(layers=0, ltm_tokens=0)
    return config


def build_policy(core, core_config, env):
    factory = {"gru": GRUCore, "ctm": CMMCore, "cmm": CMMCore,
               "looped_transformer": LoopedTransformerCore,
               "persistent_ctm": PersistentCTMVectorCore,
               "transformer_xl": TransformerXLCore}[core]
    return RecurrentActorCritic(factory(int(flatdim(env.observation_space)), **core_config),
                               env.action_space)


def policy_size(policy):
    state = policy.core.initial_state(1)
    tensors = state.values() if isinstance(state, dict) else (state,)
    return dict(parameters=sum(p.numel() for p in policy.parameters()),
                retained_state_bytes_per_lane=sum(x.numel() * x.element_size() for x in tensors))
