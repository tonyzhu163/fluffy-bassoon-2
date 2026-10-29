# Continuous noise distillation for frozen-VLA steering

`meta_rl.vla_noise_distillation` implements the data-generation idea in
[Vintix, Algorithm 1 and Section 2.1](https://arxiv.org/html/2501.19400v1).
At each actual steering decision it executes

`z = (1 - epsilon) * deterministic_teacher(observation) + epsilon * Uniform(-1, 1)`.

This is an experimental adaptation of CND to a frozen VLA's 32-dimensional
initial-noise steering coordinates. The original procedure mixes physical
continuous actions. A competent fixed DSRL teacher supplies the bounded mean
coordinate; the VLA motor and teacher remain frozen during data collection.
The generator checks the checkpoint hash and actual source-state fingerprint
before and after collection. It does not update SAC or fabricate teacher-step
progress. Qualify teacher competence on real robot episodes for that exact
checkpoint/task/interface first; a few early source-training successes do not
establish an expert. The caller declares and records the required success rate
and evaluation count (defaults 70% across at least eight episodes).

The epsilon clock spans episode resets. Its duration and positive decay power
are sampled per trial from declared bounds/options; epsilon and schedule
parameters appear only in audit metadata, never student inputs. Complete
episodes continue until the minimum episode count and annealing duration have
both been met. The declared maximum complete-episode budget bounds cost. If it
ends before epsilon reaches zero, raw episodes and integrity receipts remain,
but no complete CND dataset is exported. A linear schedule is supported; powers
above/below one allow slower/faster initial decay. These choices are proposals
for this steering port, not a reproduction of Vintix's task-tuned schedules.

Each CND trial contains the actual fixed VLA observation features, executed
mixture, actual reward, done flag and episode starts. The next-action target is
the executed noisy mixture, as in original CND, not a relabeled clean action.
Objective `cnd` accepts a constant teacher checkpoint identity and requires
target/executed equality. Existing causal Transformer training consumes these
trials unchanged. Prefer the shared native VLA image/instruction features so
all tasks occupy a common frozen representation. Separate whole trials/scenes
before constructing windows; semantic task OOD requires the existing explicit
task manifest and exclusion of test-task data.

The caller owns allocated Linux execution, simulator/motor construction and
the fixed teacher. `collect_noise_distillation` accepts deterministic action
and fingerprint callbacks plus `run_episode` (or an allocated test callback).
It writes fresh hash-bound traces, schedule receipts, dataset manifests and
source-interaction accounting. Prior teacher training and new CND collection
are counted separately and summed once; deployment cost is reported later.

Required comparisons are the native VLA, ordinary DSRL at the same deployment
budget, expert-only BC, a matched observation-only student, and the history
student with memory reset between episodes. Include a fixed/no-history noise
schedule control and reward/history interventions: an increasing curve can
otherwise be a learned annealing clock. Inference starts with empty context,
uses the student's own outcomes, and supplies no schedule or noise amount.
Measure per-attempt robot success and environment interactions, not imitation
loss alone. Teacher competence does not imply either CND improvement or OOD
adaptation; those remain empirical questions.

`meta_rl.vla_cnd_run` runs the same-task proof of concept with the qualified
SAC feature interface. It measures the deterministic teacher on a declared
calibration bank before collecting any histories; a failed competence gate
retains receipts and exits without CND training. This calibration bank may be
within the teacher's training scenes and is not evidence of task transfer.
Deployment scenes remain excluded from teacher training and student datasets.

If the teacher is weak, `meta_rl.vla_teacher_continue` restores the complete
source state into a fresh run under an explicit migration contract and calls
the existing ordinary DSRL training loop. It preserves the original training
scene bank, batch size, UTD, optimizer, replay and RNG. Its additional-step
ceiling reserves a full final episode, so completion rounding stays within
the declared interaction budget. It then qualifies the new deterministic
teacher again with the same fixed threshold. No VLA weights are updated.
Teacher continuation and calibration are additional source costs; neither is
student adaptation. `vla_teacher_continue.sh` uses the same allocation probe,
runtime and simulator setup as `vla_cnd_poc.sh`. A qualified continuation's
fresh experiment directory can be supplied as the latter launcher's third
argument; the original active run and checkpoint remain untouched.
