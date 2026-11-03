# Flow-SDE source → AD/DPT learned-KV steering

## Decision and scope

On November 3, 2026, user chat `01a1a01f-5594-7e52-9478-c8df1815df5b`
prioritized “the stronger ppo policy” and accepted this proposed route with
“yes we should use this”:

1. Improve the source action expert using Flow-SDE PPO, keeping the VLM frozen.
2. Retain actual chronological interaction histories and source policy versions.
3. Fit separate AD and DPT-style history models plus learned KV adapters using
   action-derived flow-matching supervision through the frozen original VLA.
4. Deploy with all weights frozen; actual history changes the KV edits.

This adopts the source/student architecture. It does not declare a working
implementation, improving teacher, new allocated source run or new GPU budget.
Benchmark/task selection, the simulator-first learning gate and Flora reward
qualification remain governed by `docs/FLORA-PPO.md`. Existing live/held runs,
their contracts, checkpoints and cost ledgers remain immutable. The older
velocity32/KV1152 steering-code routes are separate comparisons.

## Source and student are different policies

Source: `pi_theta(o)`, trained action expert plus value head, frozen VLM.
Flow-SDE provides Gaussian stochastic denoising transitions and the likelihood
construction used for PPO. Its sampled noise is not a downstream steering label.
One improving source campaign can supply both AD and DPT datasets; a combined
KV/velocity PPO controller or separate steering-PPO sources are unnecessary.

Student: history decoder `f_psi(H, o)` and a learned KV adapter operating on the
original frozen prefix cache. Freeze the original source initializer's VLM,
action expert, tokenizer and normalization throughout bridge fitting and
deployment. A final source checkpoint is a label teacher, not an unnoticed
replacement for the student's frozen motor. Publish exact checkpoint and
transform hashes before native fitting/evaluation.

Use learned token-local low-rank edits of both K and V with padding masks and
zero-initialized output bases. Declare exact layer coverage, rank and scale in
the implementation configuration and qualify it on the actual runtime. Keep
the plain causal decoder as the history backbone. The fixed random projection
in `robocasa_interface.py` is not this learned adapter. A velocity adapter may
later share the same source for a separate supervised interface comparison.

## Data and losses

History input contains completed `(observation, executed_action, reward,
episode_boundary)` transitions and the current public observation. Encode
actual action chunks for history; full-weight sources have no executed KV
codes. Retain memory across episodes of a trial and clear it between trials.
Preserve task-pure chronological streams. Split whole source streams before
window sampling, normalization fitting and teacher relabeling. Policy version,
collection clock and future rewards remain audit metadata, not student inputs.

| Objective | Query target | Historical context |
| --- | --- | --- |
| AD | Actual action chunk from that source policy version | Actual chronological interactions |
| DPT-style | Independently qualified final source teacher's action at the saved query observation | Exactly the same actual interactions as AD |

DPT never replaces historical actions or their observed rewards with teacher
queries. Bind query-noise law/seeds and teacher checkpoint separately from
behavior sampling. A final checkpoint is not automatically an expert or an
optimal-action labeler.

For target model-space action chunk `a*`, sample independent fitting noise
`epsilon` and interpolation time `tau`, construct
`x_tau = (1 - tau) * a* + tau * epsilon`, and minimize
`||v_frozen_with_bridge(x_tau, tau, observation, history) - (epsilon - a*)||^2`
over the declared valid horizon and active model coordinates. Train only the
history model and KV adapter. Autograd traverses the frozen suffix; expert
weights stay outside the optimizer and unchanged. No inverse KV-code solver
or extra bridge PPO campaign is required for this supervision.

Action encoding is a numerical qualification, not a reshape of the existing
600-dimensional physical-action labels. Prefer the native pre-decoding model
action chunks and retain the executed physical commands beside them. Bind
normalization, model padding, binary/gripper mapping, state-dependent decoding
and loss masks. Physical binary outputs are not invertible to the teacher's
continuous pre-decoding values. If only physical targets are available,
qualify and disclose a canonical action encoder; do not claim reconstructed
raw teacher outputs or reuse unknown/padded coordinates as supervised truth.

## Existing code and integration work

| Component | Existing capability | Work required for this route |
| --- | --- | --- |
| `rlinf_robocasa_flow_ppo*` | Native action-expert Flow-SDE source, checkpoints and histories | Qualify an improving source recipe and exact prospective history payload |
| `flow_history.py` | AD/DPT physical-action corpus and direct-action student backend | Add a separately versioned bridge target/transform view; keep existing direct-action runs compatible |
| `causal_decoder.py` | Plain causal history backbone | Encode actual previous actions/rewards/boundaries and emit causal adapter context |
| `rl/pi05_residual.py` | Zero-init learned token-local KV residual; differentiable frozen-suffix primitive | Connect causal context, native layer coverage and action-derived flow loss |
| `steering_history_native.py` | Own-history native attempt evaluation | Add a declared learned-bridge backend with its own checkpoint/state contract |

Do not reuse `FrozenRoboCasaInterface.execute_batch` unchanged for fitting:
it deliberately runs without gradients and detaches steering codes. The
bridge requires the differentiable native suffix path and a history-state
contract. Keep motor inference and bridge-training autograd paths explicit.
Checkpoint the history model/adapter, optimizer, sampler, normalization and
all relevant RNG state. Never include unfrozen copies of the expert in the
bridge optimizer.

## Qualification and execution order

1. **Source learning:** choose a concrete current-main Flow-SDE recipe and
   verify native likelihood replay, effective actor updates and physical
   success improvement on a fixed development panel. Keep a separate final
   test panel. Our completed eighteen-update OpenDrawer reference did not
   pass its improvement gate; it may qualify plumbing, not an improving
   common teacher. Preserve raw histories for a future successful source.
2. **Core numerical checks on allocated Linux:** zero-output native parity;
   padding/nonmutation; nonzero adapter gradient with unchanged expert
   weights; reference flow interpolation/loss masks; causal history without
   current/future label leakage; unchanged AD/DPT contexts; exact next-step
   checkpoint recovery. Keep tests for these model/numerical behaviors only.
3. **Native bridge qualification:** probe the actual allocated GPU, verify
   native cache/suffix consumption and action encoding, measure fitting batch
   capacity and held-out teacher imitation, and run real simulator episodes.
   Matching supervised loss does not by itself establish closed-loop success.
4. **AD/DPT fitting:** share the exact source/split/encoder and train each
   student to its declared validation stopping rule. Use an admitted improving
   source for learning claims; label non-improving-source fits as plumbing.
5. **Own-history robot evaluation:** execute paired native cases/streams with
   memory carried across attempts. Compare same-weight reset and reward-blind
   controls, original motor and final teacher. Report first-attempt strength
   separately from the benefit attributable to feedback/history.

Full-expert PPO is a strong online adaptation comparison. Claims that isolate
the learner at fixed trainable capacity additionally need online PPO on the
same learned bridge. This matched comparison is not a source-data prerequisite.
Report unique source interactions once across AD/DPT, labeler computation,
each fit's computation, deployment interactions and all failed attempts.

Implementation and resource binding precede dispatch. No GPU job is launched
by adopting this plan. Apply the repository's budget rules to a concrete new
run; this plan neither reallocates the held steering campaigns' budgets nor
lifts their human hold. Do not edit another owner's coordination row or
message another chat without explicit user authorization.

## References

- [VLRLA-bridge model and flow-matching objective](https://arxiv.org/html/2610.19517v1#S4.SS1): full action-expert source training and frozen-expert bridge fitting are distinct stages.
- [πRL Flow-SDE](https://arxiv.org/html/2510.25889v3#S4.SS2): stochastic denoising transitions supply the PPO likelihood construction.
- `docs/literature/robocasa-ppo-review.md`: released source recipes, fidelity gaps and controlled next checks.
- `outputs/robocasa-opendrawer-rlinf-sim-ppo-20261102-r1/RESULT.md`: completed source reference and failed improvement gate.

## Acceptance record

This adoption changes `docs/PAPER.md`, `docs/METHODS.md` and
`docs/FLORA-PPO.md`, with one owned row in `outputs/coordination.md`.
Implementation/native qualification and a successful source remain outstanding;
this decision record is not a training result.
