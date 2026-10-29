# Frozen-VLA in-context adaptation for robotic performance

Started 2026-10-29 Asia/Shanghai. User requests an actual Baige experiment,
parallel implementation, and a decision based on robot success, deployment
sample efficiency and final performance. ICRL/meta-RL supplies methods; CTM
architecture comparisons and artificial hidden perturbations are outside this
experiment.

## Question

Can a frozen VLA plus a meta-trained history Transformer reach higher robot
success with fewer deployment interactions than the native motor and ordinary
noise-space RL? All student weights remain frozen during deployment. Report
teacher collection and student training costs separately; low deployment cost
does not imply low total training cost.

## First implementation

- Standard LIBERO tasks, original rewards, observations and physics. Initial
  integration task: LIBERO-90 task 57, also used by the released DSRL example.
  This single-task qualification is not evidence of cross-task meta-learning.
- Reuse the staged OpenPI pi0.5 LIBERO checkpoint and native initial-noise
  interface, with exact source, checkpoint and normalization hashes recorded.
  Verify the checkpoint's actual action horizon, execution chunk and task
  observations; do not silently substitute a ManiSkill motor.
- Ordinary SAC in initial-noise space generates teachers and their actual
  learning histories. A Torch port is explicitly a DSRL-style implementation,
  not a reproduction of published pi0/JAX numbers. Record departures from the
  official implementation before interpreting results.
- A modest causal Transformer predicts stochastic noise decisions. AD uses
  actual chronological next decisions of an improving source learner; DPT
  labels query observations with a strong saved teacher while retaining the
  actions and outcomes actually executed in the context. Never relabel failed
  historical actions while preserving their old rewards.
- Keep observation, executed noise, executed action chunk, actual reward,
  termination, reset/scene seed, teacher update and source identifiers. Query
  labels and current/future outcomes are not student inputs. Split complete
  task/scene trials before constructing windows. Evaluate on student-generated
  histories, not only teacher-forced sequences.

## Evaluation and decision

Evaluate paired independent scene seeds and matched episode horizons. Carry
history across K attempts at the same scene; clear it before a new independent
trial. First use K=4 to get a complete comparison; extend to K=8 only as a
separately recorded evaluation. Include:

1. Native frozen VLA with its original full Gaussian-noise sampler.
2. Frozen VLA with the steering controller's untrained sampling convention
   (if noise tying or bounds differ from the native distribution).
3. Ordinary DSRL-style teacher trained on the same original task. Report both
   its learning curve and frozen final performance on held-out scenes.
4. Supervised history Transformer, weights frozen, history retained.
5. Separately trained observation-only/static controller on the same labels.
6. The history controller with memory cleared between attempts.

Report success at every attempt, mean success/return over the trial, success
against cumulative primitive environment steps, first-attempt performance,
and final fresh-episode success. Also report success at least once, without
using it alone as evidence of learning. Count failed attempts, resets, teacher
warmup and exploration interactions. Pair methods on the same scene bank and
preserve failures; do not select favorable test tasks or noise seeds afterward.
Independent scenes, not individual correlated attempts, are statistical units.

A useful early result requires a competent native motor, a teacher that
actually improves robot success, and a closed-loop student comparison. A
finite gradient, decreasing supervised loss or a passing runtime probe alone
does not meet that standard. If the teacher is strong but the student fails,
retain that result and diagnose distillation/history coverage. If the teacher
cannot improve, report the actual curve and identify whether the control
interface, optimizer, data budget or a success ceiling is limiting it.

After the first complete run, expand a declared training/held-out task split
and independent training seeds. Task holdout is relative to adapter training;
do not describe tasks as unseen to the VLA without pretraining provenance.
Topline comparisons require matched benchmark protocol and adequate evaluation
scenes. No broad efficiency or performance claim follows from a smoke test.

## Execution

Fresh GitHub main verified at 15:12 Shanghai:
`4a6a690ad2c98be7bf3c0ca06648870581f7bb6d`. Main clone has pre-existing
coordination edits; all implementation is in detached worktrees. The local
interactive shell lacks the documented `proxy_on` function; attempts and the
connected GitHub verification are recorded rather than changing shell config.

Baige qualification reservation: one exclusive RTX4090, 16 CPUs, 96 GiB RAM,
8 GiB shared memory, at most 45 minutes. Submit when the source is committed
and static review is complete. Use the allocation as the GPU boundary, run the
repository GPU probe in that container, then real LIBERO rollouts and actual
learner updates. Measure throughput and capacity before admitting the complete
teacher/student comparison. Preserve active unrelated jobs and environments.

Runtime, learner and integration subagents operate independently; the parent
integrates and reviews the exact launch commit. Deploy through Git (bundle if
needed), verify source and dependency pins on Baige, and retain all raw learning
and evaluation records plus a complete resumable checkpoint. Check initial
progress after launch, then on the repository's 20-minute cadence.
