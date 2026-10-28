# Audit: in-context RL with CTMs in this codebase

Snapshot: 28 October 2026, `lycoreco_rl` at `089bd61`. This is an opinion
audit of the RL code (`src/fluffy_bassoon/meta_rl/`, `src/fluffy_bassoon/rl/`)
and the shared CTM (`src/lerobot_policy_ctm/shared/`). It asks one question:
what is the best way to get in-context RL (ICRL) out of a CTM here? The
literature background is in [the ICRL review](in-context-rl-review.md).

## Verdict

The learner infrastructure is in good shape: trial environments, full-trial
recurrent PPO, recovery, likelihoods, budget accounting and controls. **No
current experiment tests the CTM as the in-context learner.** The selected
LIBERO plan uses the legacy CTM only as a frozen motor, reset at every
physical start, and puts the in-context memory in a GRU or Transformer-XL
adapter. The compact "CTM control" is the CMM port with its memory blocks
removed, running one tick by default. That is not the robot CTM.

My recommendation is to make a **separate in-context CTM adapter**. It sits
on the frozen, gate-qualified motor and carries its state across attempts,
with three changes:

1. a bounded synchronization normalization;
2. per-observation rather than per-tick decay;
3. explicit episode-summary tokens.

Train it AD-style first, then fine-tune with the existing meta-PPO. Judge it on
a task that only reward can solve, against Transformer-XL at matched budgets.
The findings below explain why.

## How the pieces fit today

| Component | File | Role in an ICRL loop |
| --- | --- | --- |
| Persistent robot CTM | `shared/persistent_ctm.py`, `shared/token_ctm.py` | Carries activated state, a 32-step pre-activation trace and two synchronization banks per observation, T8 ticks. Episode reset clears them. Has no reward input |
| Legacy LIBERO motor | `rl/libero_motor.py` | Frozen H20/E10 policy, 388/400 historically. State and queue reset at each physical start |
| Gain adapter | `rl/gain_adaptation.py` | Scalar xyz calibration on the motor output. Cores: **GRU or Transformer-XL only** |
| Compact cores | `meta_rl/cmm.py`, `meta_rl/ppo.py`, `meta_rl/transformer_xl.py` | `--core ctm` is `CMMCore(layers=0, ltm_tokens=0)`, with `ticks` defaulting to 1 |
| Trial PPO | `meta_rl/ppo.py`, `meta_rl/train.py`, `meta_rl/robot_train.py` | Whole-trial replay. Meta mode carries state and returns across attempts. Gamma = GAE lambda = 1 |
| Episode PPO for the motor | `rl/actor_critic.py`, `rl/recurrent_ppo.py` | Complete-episode PPO of the saved MIKASA CTM. Sparse first-success reward, SFT anchor |
| Robot frontend | `meta_rl/native_prefix.py` | Frozen native K/V read by 4 learned queries at width 64, plus 12 feedback fields (previous command7, reward, attempt_end, absorbing, clock2) |

## Findings

### F1. The CTM is not the in-context learner anywhere (design gap)

- In the LIBERO plan, the motor CTM resets per physical start
  (`collect_gain_budget(... reset motor execution state while retaining
  adaptation memory)`), and the memory that spans attempts is
  `GainAdapter` (GRU or Transformer-XL). The handoff orders the work as
  "CTM comparison later".
- The compact `ctm` core is the CMM code path with zero memory blocks. It has
  linear-weight synchronization, RMSNorm and a shallow GLU neuron model, and
  differs from the robot CTM's dual banks, `alpha/sqrt(beta)` normalization,
  deep NLMs and attention over context. Its results do not transfer to the
  robot CTM, and the DESIGN file says so.
- **Consequence:** the five-core screen (Transformer-XL 0.730 versus CTM
  control 0.669 sampled reward) says little about the CTM that will be used on
  robots. A CTM claim needs a CTM adapter in the robot loop.

### F2. The legacy synchronization drifts in scale over long carried trials (correctness risk)

`sakana_reference.compute_synchronisation` updates
`alpha = r·alpha + product` and `beta = r·beta + 1`, then returns
`alpha / sqrt(beta)`. For r close to 1, beta grows with the tick count n, and
the output grows roughly as sqrt(n) times the mean pair product.

The robot CTM applies this once per **tick** and persists it across
observations. Over a K4/H220 trial at T8 that is up to about 7,000 ticks.
For pairs with near-zero learned decay, the feature scale can be about 80
times larger than after one tick. Pairs with large decay saturate quickly, so
the effect depends on the learned decays. The motor was trained with carry
inside one episode only, so carrying its state across attempts feeds the
readout magnitudes it never saw.

The CMM port already uses the bounded `alpha / beta` followed by RMSNorm.
**Fix:** any CTM that carries state across attempts should use a bounded
normalization (`alpha / beta`, then RMSNorm or LayerNorm). Do not carry the
legacy motor's synchronization banks across attempts. Check this empirically
by logging per-bank synchronization norms against trial step.

### F3. Ticks set both compute and memory horizon (confound)

The decay r = exp(−d), with d clamped to [0, 15], is applied per tick. The
effective memory horizon in observations is therefore about
1 / ((1 − r) · T). Doubling ticks halves how many past observations the
synchronization remembers.

The research plan's tick sweeps ("continuing computation adds useful
learning capacity") would confound extra computation with faster forgetting.
**Fix:** for the in-context CTM, apply the cross-observation decay once per
observation (decay within an observation's ticks can stay per tick). The
alternative is to report the effective horizon per learned decay alongside
every tick sweep.

### F4. The CTM-defining property was not exercised in the compact screens (evidence gap)

`meta_rl/train.py` defaults `--ticks 1`, and the evaluation records in Git
do not show a tick override for the five-core screen. With one tick, the CTM
control is a recurrent cell with neuron-level history models. The internal
recurrence that distinguishes a CTM is absent. Re-running the bandit screen
at T = 1, 4 and 8, with F3 fixed, is a cheap check before robot work.

### F5. Cross-attempt evidence has no explicit store (capacity)

The pre-activation trace holds 32 steps, which is four observations at T8
(10 in the compact core). Anything remembered across attempts must survive
in the activated state and the synchronization accumulators. Both are
continuously overwritten by within-attempt control.

ECET and VLRLA both add an explicit episode-level memory. The CMM long-term
tokens are one option; a cheaper one is a small FIFO of **episode-summary
tokens**. At each `attempt_end`, write the CTM's output synchronization (or
a learned projection) into a slot that later ticks attend to, next to the
observation K/V.

### F6. The current robot task does not require reward (scientific validity)

The handoff already notes this: a translation gain can be identified from
issued command against observed motion. The 12-field feedback includes both,
so a learner can recover without using reward at all. That is in-context
system identification, which is useful, but not ICRL in the sense of the
literature.

**Fix:** add a reward-only variant. Examples: K visually identical targets
with one hidden correct target, an underspecified instruction with a hidden
goal (VLRLA's relabeling), or a MIKASA ShellGame-style choice whose
reward depends on a trial-fixed hidden choice. Keep the gain trial as a
system-identification control.

### F7. The adapter's view of the scene is very compressed (capacity)

`NativePrefixReadout` reduces the native K/V to 4 queries at width 64. The
motor CTM attends to 641 tokens at width 512. That is enough to correct a
scalar gain. A reward-driven choice among objects needs spatial detail.
**Fix:** let the in-context CTM attend directly to the motor's projected
observation K/V, the same tensors the motor reads, as its context. Do not
pass a pooled vector.

### F8. Full-trial BPTT at T > 1 is not capacity-qualified (cost)

Trial PPO replays every trial from its initial state each epoch. The robot
Transformer-XL is qualified for K4/H220 (880 steps). A CTM adapter at T8 over
880 observations is roughly 7,000 recurrent ticks per lane with attention
over 641 tokens. That needs activation checkpointing or truncated BPTT with
carried values, which the stream sampler already supports for supervised
training. Qualify memory before launch.

### F9. Learner details that are already right (keep)

- Latent-space Gaussian likelihoods with the tanh Jacobian, scoring the saved
  latent rather than the clipped command.
- Zero-initialized new feedback columns and critic, so PPO starts exactly at
  the SFT policy.
- Unnormalized advantages for single-lane trials; zeroed actor terms on
  absorbing padding.
- Decay parameters projected after the optimizer step rather than mutated at
  inference.
- Separate episode-reset controls and reset lesions, disjoint seed windows,
  and exact interaction accounting.

Reuse all of these unchanged for the CTM adapter.

## Recommended design

### Architecture: an in-context CTM adapter on a frozen motor

1. **Motor:** the gate-qualified legacy CTM, frozen, H20/E10, with state reset
   at every physical start. This is unchanged from the plan.
2. **In-context CTM:** a new `PersistentCTMCore` instance, for example D256
   and T4, as its own saved identity. Its attention context is:
   - the motor's projected observation K/V for the current frame;
   - a feedback token (previous chunk summary, reward, attempt_end,
     absorbing, clocks);
   - a FIFO of E episode-summary tokens (F5).
   Its state is carried across attempts and reset only at trial start.
3. **Synchronization:** bounded `alpha / beta` plus RMSNorm (F2), with
   cross-observation decay applied once per observation (F3).
4. **Output:** a zero-initialized residual. The first version is a correction
   on the motor's action-chunk mean (the general form of the scalar gain
   correction). A later version adds to the motor's conditioning token,
   following VLRLA-bridge. Step zero then reproduces the gate-qualified motor
   exactly, as the existing zero-initialized feedback columns do.
5. **Critic:** reads detached synchronization features, as in
   `rl/actor_critic.py`.

The alternative is to make the motor CTM itself carry state across attempts,
with zero-initialized feedback tokens appended to its context. That is
attractive because it reuses a trained CTM. It is riskier: F2 means its
synchronization scale drifts, and fine-tuning it can break the motor gate.
Try it only after the adapter works.

### Training

1. **Stage 0:** pass the motor gate (pending job 47479) and a B0 headroom
   screen. This is already planned.
2. **Stage 1, distillation (AD/Vintix/VLRLA):** build trials whose later
   attempts improve. Options are noise-annealed motor rollouts, a wrong-then-
   correct hidden-target hypothesis switched by reward, and logged `rl/`
   PPO trials with failures kept. Train the adapter on the action residual
   with the stream sampler extended to trial lanes and TBPTT with carried
   values.
3. **Stage 2, meta-PPO (RL²/ECET):** `meta_rl/ppo.py` in meta mode with
   sparse latched success, an SFT anchor toward the Stage 1 adapter, and a
   small learning rate. Keep the existing padding and advantage rules.

### Evaluation

- Success at attempts 1..K with frozen weights, greedy and sampled.
- Controls: same-checkpoint reset lesion, separately trained episode-reset
  adapter, unadapted motor, privileged oracle (true gain or true target).
- Architecture controls: Transformer-XL and GRU adapters with identical
  inputs, matched on parameters, retained state bytes and measured compute.
  Run a tick sweep only after F3 is fixed.
- Tasks: compact bandit first (cheap check of F2 to F4); the LIBERO gain trial
  as system identification; a reward-only hidden-target trial as true ICRL.
  Meta-World ML10 with the compact CTM is optional, for comparison with ECET,
  TrMRL and AMAGO.
- Three or more training seeds, paired evaluation seeds, and predeclared gates,
  as in the existing plan.

## Concrete changes, in order

| Step | Change | Files |
| --- | --- | --- |
| 1 | Log synchronization norms against trial step for the motor carried across attempts (read-only check of F2) | `scripts/diagnostics/` (new) |
| 2 | Add bounded normalization and per-observation decay options to the CTM core, off by default so existing checkpoints are unchanged | `shared/persistent_ctm.py`, `shared/token_ctm.py` |
| 3 | Re-run the compact Bernoulli screen with the real `PersistentCTMCore` at T1/T4/T8, both normalizations | `meta_rl/ppo.py` (new core adapter), `meta_rl/train.py` |
| 4 | Add `core="ctm"` to `GainAdapter`, reading motor K/V plus feedback and summary tokens | `rl/gain_adaptation.py`, `meta_rl/native_prefix.py` |
| 5 | Add a reward-only hidden-target LIBERO or MIKASA trial wrapper | `rl/` or `meta_rl/libero_trial.py` |
| 6 | Extend the stream sampler to trial lanes for Stage 1 distillation | `training/sampling.py`, `meta_rl/warmstart.py` |
| 7 | Qualify K4/H220 T4 capacity, then run Stage 2 meta-PPO with matched Transformer-XL | `meta_rl/robot_train.py` |

Steps 1 to 3 are cheap and decide whether the CTM-specific changes matter.
Do them before spending GPU time on step 7.
