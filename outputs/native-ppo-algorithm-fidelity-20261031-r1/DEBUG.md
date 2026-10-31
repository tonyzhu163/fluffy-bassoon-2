# Native PPO algorithm-fidelity pass: static, not a learning result

Read from root worktree
`/Users/mowenjie/.codex/worktrees/vla-performance-pilot/fluffy-bassoon-2`
at `8fefa89d054984c3dde85017b4dc2cc8db617b70`; the separate YAML fix741df47
was already assigned. Official source was read directly on Baige at
`/mowenjie/fluffy-bassoon-2/sources/robotwin-baseline-smoke-20261028-r1/RLinf`,
verified HEAD `c605b9d55ed93a180119b099e6f836abff293b08`.
No tests, model construction, forward/backward, simulator, scheduler query,
allocation or source changes were performed. This does not revive the residual
probe or add a launch gate. Root owns the combined scientific summary.

## First separate the questions

- Earlier negative PPO results trained a recurrent **noise controller over a
  completely frozen VLA**. The completed RoboTwin reset run used32×32=1,024
  noise coordinates, bound1.5, and about311.72 actual valid decisions per global
  update;100 updates produced500 Adam steps. This is not native action-expert
  PPO. See `vla_noise_ppo.py:1,28` and the root steering summary's
  “RoboTwin reset100” section. Its nominal8192-slot VLRLA comparison is a
  capacity-versus-realized-data comparison, not matching components or batch.
- The current native pilot trains the **action expert**, keeping the VLM frozen.
  It had not trained on GPUs at this review. There is no native-pilot negative
  learning result to explain yet, and a correct native pilot would not itself
  establish that frozen-noise steering works or fails. If all VLA weights must
  stay frozen, DSRL/noise-controller work is the relevant requested experiment.
- “Uses identical RLinf” is insufficient: it does not equate the two trained
  components,4/8-rank stochastic objectives, initialization, or2 versus25 PPO
  iterations. None of the earlier weak results proves DSRL is unsteerable.

## Prioritized actionable findings

| Priority / status | Finding and exact source | Implication / action |
| --- | --- | --- |
| **P1 proven wrapper/API defect** | Root `rlinf_native_ppo_runtime.py:121–123` reads worker `global_step` to name and label the unshuffled batch. Official `embodied_fsdp_actor_worker.py:840–846` sets **version**, then the model's step; it never sets actor-worker `global_step`. The manager/Worker provide no such field. Tests manually set `actor.global_step` (`test_rlinf_native_ppo.py:180`), hiding the real API. | Both iterations can be labeled policy0/collection1 and overwrite the same history file; mask sidecars likewise use stepNone. This corrupts chronological teacher evidence, though official PPO math is unaffected. Root/finish_ppo accepted ownership of an authoritative setter/version plus collision guard and two-real-setter regression; no duplicate edit here. |
| **Proven4/8 distinction, not exact equivalence** | Global2048 =32×4×16 =32×8×8 is correct (`embodied_fsdp_actor_worker.py:91–101,619–655,833`). But GAE normalizes each actor's local shard (`algorithms/advantages.py:81–82`; `algorithms/utils.py:421–428`, no collective), and shuffle uses seed1234+rank (`actor:607–609`). | Same nominal global batch and gradient-accumulation factor; **not identical data or normalized objective** when changing world size. Describe4-rank as a nominal-batch execution variant, not a numerical8-GPU reproduction. Keep world/rank partition fixed across a paired before/after run. No upstream change requested. |
| **Proven upstream config/paper drift** | `algorithm.clip_ratio_c=3` is asserted in our recipe, but official actor forwards it only for GR00T/PI0_FAST (`actor:779–794`), excluding OPENPI. `actor_critic` forwards kwargs to an actor loss whose default is `clip_ratio_c=None` (`algorithms/losses.py:179–186,465–489`). | Effective OPENPI objective has ordinary ratio clipping0.8–1.2, **no dual clip**. Do not say that validating the YAML proves dual-clip3 is active. Preserve pinned core; report effective behavior and paper/code mismatch. |
| **Proven upstream critic estimate difference; effect unproven** | Rollout averages value-head outputs across all5 denoising steps (`openpi/tasks/rl.py:268–296`). With default `joint_logprob=False`, recompute uses only the selected step and its value (`:451–491,505–507`). | Unchanged-weight critic values need not reproduce the rollout mean exactly; clipped value loss compares different estimates. Could affect critic stability, but no GPU tensors/results yet establish harm. Selected-step policy log-probs remain the appropriate replay quantity. Do not silently change the core or assert exact critic replay. |
| **Proven inaccurate loss-weight annotation** | Root sidecar says “per32 masked-token means” (`runtime.py:176–182`). Official PPO replaces the selected aggregator when `loss_mask_sum` and episode horizon are present (`losses.py:229–237`), using `(values / loss_mask_ratio * mask).mean()` (`utils/utils.py:432–436`). | Record the effective ratio/length-scaled aggregation, not just the YAML's token-mean label. This is existing official math, not a new wrapper loss. No claim of exhaustive global valid-token equivalence. |

## Settings that statically match the intended native path

- `model/pi0_5.yaml:17–32` supplies `task=rl`, `train_expert_only=True`,
  `noise_method=flow_sde`; experiment YAML overrides ALOHA/RoboTwin,14D,50
  chunks,5 denoise steps, noise0.3 and detached critic input. The factory reads
  these fields directly (`openpi/__init__.py:153–195`), then freezes SigLIP and
  expert0 after loading (`:286–296`; `pi0.py:849–909`). Expert1/action projections
  remain trainable; detached critic input stops the value gradient into expert
  features, **not** the policy gradient (`sampling/rl_sampler.py:155–170`).
- The sparse relative-success reward path is1×termination difference, chunk-summed
  for GAE (`robotwin_env.py:217–248,375–386`; `algorithms/utils.py:81–89`). Training
  disables auto-reset and retains termination masks (`actor:225–239`); GAE uses
  terminal-aware bootstrap with gamma0.99/lambda0.95 (`advantages.py:66–82`).
  No fresh supported reward/GAE defect was found in this pass.
- Actor/value LR5e-6/1e-4 are separate trainable parameter groups
  (`fsdp_model_manager.py:565–609`). Constant LR with0 warmup is genuinely
  constant; `total_training_steps=0` does **not** zero it
  (`hybrid_engines/fsdp/utils.py:630–638`).
- Nominal collection is256×4×8=8192 chunk slots/iteration. Five epochs with
  global minibatch2048 give20 optimizer-step opportunities/iteration,40 for
  this2-iteration pilot, versus500 for a25-iteration published schedule.
  Actual valid decisions depend on termination masks. Official
  `optimizer_steps` increments before a nonfinite-gradient skip
  (`fsdp_model_manager.py:459–477`), so counter40 alone is not proof of40
  applied Adam updates or a performance gain.
- The derivative runs explicit SFT-before and fixed-iteration2-after panels,
  restores per-worker sampling RNG and checks reset IDs (`runtime.py:349–375,
  283–297`). Its receipt records actual initial-observation equality rather
  than enforcing it. Paired nominal seeds and RNG are useful; actual reset
  arrays must still be inspected before attributing changes solely to weights.

## What real execution still must establish, without new scope

The already planned native evidence must show actual nonzero reward/success and
valid masks; selected-step policy logprob/ratio consistency; finite gradients
attributed to expert1 and value head; actual applied Adam clocks/weight changes;
frozen VLM hashes; and the fixed before/after reset identities, observations and
sampling provenance. Finite loss or a CPU import/recipe test cannot establish
learning. Critic replay differences above need interpretation, not an automatic
zero-difference assertion. Two iterations test runtime and learning onset, not
published convergence or the frozen-VLA meta-learning claim.

No core RLinf patch, new method, resource request or scientific gate is proposed
by this report. Current user priority/parameter-freezing choice belongs to root.
