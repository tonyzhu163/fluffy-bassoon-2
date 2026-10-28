# Sakana CTM core

Pinned upstream: https://github.com/SakanaAI/continuous-thought-machines/tree/4a6c9c3a7fb5dc4bca6381cc7883a3b9252c6466

`src/lerobot_policy_ctm/shared/sakana_reference.py` contains the unchanged Identity, Squeeze,
SynapseUNET and SuperLinear definitions, plus the core synchronization and setup
methods from that revision. Imports are limited to their dependencies.

The project owns the per-action conditioning/attention adapters, masking,
direct readouts and training orchestration. Upstream's Apache-2.0 license
is included here. No upstream image backbones or unrelated task utilities are
vendored.

`shared/post_sliding_ctm.py` adapts the finite post-activation window from
`models/ctm_rl.py` and decay weighting from `models/utils.py` at the same revision.
It uses the RL learned M-entry initial post trace and retains LYCO pairing,
`[0,15]` decay bounds and attention. The window applies to both existing banks,
with independently configured pre/post lengths; see the post-sliding designs.
