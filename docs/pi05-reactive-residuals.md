# Optional reactive pi0.5 residual hooks

This is implementation preparation, **not allocated-test or live qualification**.
The development contract is
`outputs/frozen-pi05-steering-screen-20261031-r1/PLAN.md` on the redesign
owner's checkout. No motor, teacher, action bounds or resource grant is selected
by these hooks. Existing production defaults remain unchanged.

`rl/pi05_residual.py:ReactiveKVResidual` adds rank-r, zero-output-initialized
corrections to selected native post-RoPE K/V cache layers. Token counts,
positions, `[B,P,KVH,HD]` layout, dtype, masks and padded values are preserved.
Inputs are current public features and each token's detached native K/V;
there is no history, prefix LoRA, token insertion or motor-weight update.
Native-cache scale is not a calibrated safety bound. The existing
`Pi05FeatureBridge` remains a distinct **flow-velocity** residual.

`velocity_with_kv_residual` recomputes one suffix pass with autograd enabled
through edited KV and frozen motor weights. It rejects no-grad/inference-mode
execution or trainable motor parameters. It does not detach suffix outputs or
substitute saved detached suffix/velocity records. It is a primitive for future
flow matching, not a learner or a complete training loss.

`ReactiveALOHAActionResidual` reads the same current public feature vector,
the detached base decoded action proposal, and normalized action-slot index.
It adds a tanh-bounded correction to `[B,H,14]` decoded ALOHA targets: six left
arm joints, left absolute gripper (index6), six right joints, right gripper
(index13). Arm/gripper bounds are separately required in decoded command units;
no physical values are guessed. Optional absolute command limits require the
base proposal already inside their box, preserving zero-head parity. The result
separately exposes base, requested/applied delta, clipping and final commands.
Both hooks expose reconstructable configuration, including action-head width.
They do not assert equal parameter counts or information processing: KV editing
also reads individual native tokens and the action head reads base proposals.

The scalar `RobotwinNoiseEnv` forwards optional `kv_adapter=` or
`action_residual=` to its existing backend; exactly one may be selected. Pass
an explicitly configured eval-mode hook on the motor's device. KV editing is
before the native cached sampler; action correction is after native decoding
and before the existing `chunk_step`. Both retain the native checkpoint,
sampler, horizon, normalization, decoder, renderer and actual-action mask path.
An optional-interface contract and typed per-decision metadata identify the
intervention; action proposals/deltas/clipping are retained alongside actual
issued commands and the existing native valid-duration mask. No raw corpus is
rewritten. Public features are bound to the current prepared observation;
stale bindings fail before motor execution. Parallel production lane-cache
migration is intentionally not added or qualified here. Existing lanes do not
carry these optional feature bindings and must not enable the hooks.

Run focused synthetic tests **only inside a separately admitted Linux CPU
allocation**, with a fresh identity/resource ledger and JUnit/raw receipts:

```sh
uv run pytest tests/test_pi05_residual.py tests/test_robotwin_noise.py -q
```

These tests use a small CPU suffix and simulator fake to exercise zero native
parity, layout/padding, nonmutation, frozen-gradient isolation, a nonzero
adapter-output gradient through the suffix, distinct ALOHA bounds, command
clipping, backend placement, metadata and executed-duration accounting. They
do not load pi0.5, initialize CUDA or prove actual RLinf/simulator parity.
Live qualification, capacity measurement and any teacher fitting remain
separate and unperformed. No closed CPU/GPU grant is reused.
