# Frozen pi0.5 feature bridge

`rl/pi05_feature_bridge.py` is the primary visual bridge interface for comparing
a persistent CTM with Transformer-XL. The scalar cached A1 pilot remains a
separate reward-only plumbing control. This module provides a recurrent reader,
a zero-initialized residual, and inference integration; it does not yet provide
teacher data, a training objective, PPO likelihoods or a qualified live result.

## Native source hooks

The RLinf runtime pinned in [the ManiSkill protocol](maniskill-pi05-icrl.md)
provides these methods, inspected directly in its installed source:

- `openpi/pi0.py:411`, `build_prefix_cache`: returns final prefix hidden states,
  a boolean prefix mask and a tuple of layer KV pairs.
- `openpi/modules/gemma.py:305-342`: native caches contain RoPE keys and values
  in `[batch,prefix,kv_heads,head_dim]` order. Gemma-2B and the 300M action expert
  have 18 layers, one KV head and head dimension 256.
- `openpi/pi0.py:436`, `run_suffix`: consumes the unchanged prefix cache and
  returns `[batch,8,1024]` action-expert hidden states.
- `openpi/pi0.py:472`, `velocity_from_suffix`: projects those states to native
  `[batch,8,32]` flow velocities.
- `openpi/env_io.py:139`, `_predict_eval`: calls deterministic Euler sampling,
  then `decode_actions`. This also applies to eval mode of the RL motor.

The bridge reads the final layer's detached native keys and values, concatenated
to width 512. It uses every valid prefix token with the original mask. It does
not substitute final VLM hidden states or append trainable tokens to the native
cache. CTM attends through the house persistent CTM attention and retains its
complete recurrent state. Transformer-XL uses a learned masked cross-attention
reader followed by the house relative-attention recurrence. Both read the same
native layer, mask and public feedback, and use the same residual interface.
Equal width does not match parameters or retained state; qualification must
record both counts and tune budgets before architecture claims.

## State and intervention contract

Call `read_context` once per observation/action chunk, with detached frozen KV,
public feedback and the carried learner state. Reuse the resulting features
through all four denoising passes. Advancing recurrence once per Euler pass
would silently change the learner's memory horizon. The caller resets selected
lanes only at trial starts; physical attempt resets retain memory. A memory
lesion supplies the reset mask at each attempt boundary.

The default 11-dimensional feedback vector is reserved for seven previous
executed action coordinates, previous latched reward, attempt-end indicator,
attempt-index/K and chunk-index/chunks-per-attempt. The caller must construct
this vector from public history, zero it at trial start, and zero its reward
coordinate for reward blinding. Hidden task settings and diagnostic outcomes
must not be included. Feedback construction and the live trial collector still
need integration and validation.

`correct_velocity` concatenates the recurrent features, detached native action
expert states, detached noisy actions and time, then predicts a tanh-bounded
velocity residual. Its final projection starts at exactly zero, so the initial
bridge reproduces the complete native motor for identical observations and
noise. Only this final projection receives residual-path gradients on the first
update; a nonzero projection subsequently trains the reader and recurrence.
The default residual scale is 0.1 in normalized flow-velocity units and must be
qualified on motor rollouts. Native weights and caches remain unchanged.

`predict_action_batch_with_bridge` uses the native environment transforms,
preprocessing, prefix/suffix helpers and action decoder. Pass its decoded actions
through the existing `prepare_actions` and `env.chunk_step` path. No monkeypatch,
third-party source mutation or second VLM pass is required.

## Interpretation and next qualification

Visual KV and action-expert states can expose sensorimotor evidence about a
hidden rotation. A reward-blind visual learner can therefore improve through
system identification. Use both same-checkpoint lesions and separately trained
reward-blind controls to establish how much improvement depends on reward.
The scalar A1 control has a stricter reward-only information contract. A visual
bridge adaptation curve alone cannot establish reward-driven in-context RL.

On allocated Linux resources run `tests/test_pi05_feature_bridge.py`. Synthetic
tests check exact zero-residual Euler parity, masking, lane reset, immutable
incoming state, blocked gradients into native features and one recurrent update
per chunk. `scripts/diagnostics/maniskill_pi05_feature_bridge_gate.py` runs one
80-step native episode, checking both zero CTM and zero Transformer-XL bridges
against native pi0.5 on each of its 16 live observations. It clones the CUDA
generator before each query and demands exact equality of all 32-dimensional
model actions, decoded actions, prepared simulator commands and resulting noise
streams. All methods share the one proven-identical executed trajectory. This
is motor preservation qualification, not three independent success trials or
an adaptation test.

The gate uses the same `--rlinf-root`, `--model-path`, `--norm-stats`,
`--expected-model-sha256`, `--source-commit` and unique `--output-dir` arguments
as the existing object gate. It writes an allocation-local GPU probe, actual
renderer binding, source/runtime/motor hashes, public feedback, complete issued
commands, native KV/suffix shapes and motor/bridge fingerprints before and
after inference. Qualification intentionally computes a native pass and one
pass per bridge on each observation; ordinary bridge inference uses one prefix
pass. Continue with teacher residual training only after this gate passes.
Launch GPU work within the user-approved task and resource scope once verified
code and allocated resources are ready; ask before expanding scope or materially
increasing the resource budget, as specified in AGENTS.md.
