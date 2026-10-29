# LIBERO task generalization with frozen VLA steering

The semantic task split belongs to the attachment's meta-training data. A task
held out from attachment training may already occur in the released VLA's
training data; this protocol makes no VLA pretraining novelty claim.

Use one released motor checkpoint, tokenizer and normalization contract across
all tasks. `--feature-interface vla-prefix` uses the motor's native preprocessing,
image encoder, instruction embedding and final VLM prefix layers. It separately
averages valid visual tokens and valid instruction tokens, and appends public
eight-dimensional robot state. The result has 4104 dimensions and a common
coordinate system across tasks. Features contain no teacher identity, simulator
object state, task index, future outcome or query label. They do contain the
ordinary public language instruction. A numerical check compares the pooled
function with unpooled native final-prefix hidden states. Current implementation
recomputes the native prefix during feature extraction and decoding, records that
cost, and does not retain a large KV cache in the history dataset.

For the first task-generalization pilot, declare a small training task bank,
separate validation tasks and a fixed semantic test bank before training. Resolve
the canonical BDDL task names on the pinned standard LIBERO source. The
`frozen_vla_multitask_v1` manifest lists each task's `task_key`, `suite`, `task_id`,
`semantic_task` (BDDL stem), and `split` (`train`, `validation`, or `test`). Training
and validation tasks additionally supply `datasets: {ad: PATH, dpt: PATH}`.
Test tasks cannot supply supervised datasets. Do not select test tasks or noise
seeds after seeing student performance. A small source-task headroom screen may
guide the declared training bank; retain its complete selection evidence.

Run independently pinned ordinary DSRL source learners on each training or
validation task. Their private image encoders need not align: all students read
the shared frozen VLA prefix features, collected at the actual query observation
before steering. Within a task, independently restore a shared source prior for
each training scene and collect K same-scene attempts, updating SAC only between
attempts. Preserve actual action/outcome chronology. AD targets are the executed
source coordinates. DPT targets use the final scene-adapted source at those query
observations; historical actions and outcomes stay unchanged.

`python -m fluffy_bassoon.meta_rl.vla_multitask --manifest MANIFEST --output OUT`
verifies actual source suite/task/BDDL identities, disjoint semantic task banks,
exactly matching motor/feature/noise contracts, and exact AD/DPT context equality.
It aggregates complete source trials into shared AD/DPT train and validation
collections before any windows are sampled. Train the existing
`vla_history_train` on each objective with identical Transformer, initialization,
normalization, data windows and optimizer budget; train a static comparator too.
Checkpoint selection uses only validation tasks. Freeze every candidate before
running held-out semantic tasks with its own history.

For a single-task integration comparison, `--compare-supervised` performs the
same-data AD/DPT training and paired full-history/static/reset evaluations in one
run. `--rl2-updates N --rl2-max-steps BUDGET` adds direct complete-trial RL²/PPO
on the declared training scenes. It uses the same generic frozen feature space
and training-only feature statistics; its own additional robot interactions are
counted. RL² semantic multi-task scheduling must sample the declared training
task bank and preserve one task/scene through each complete K-attempt trial;
the initial CLI qualifies the real single-task PPO loop, not that broader claim.

Compare native full-noise sampling, tied-noise sampling, frozen ordinary DSRL,
ordinary DSRL updating between held-out retries, AD, DPT and RL². For every
meta-trained candidate include state reset between attempts and an independently
trained static comparator where applicable. AD, DPT and RL² share the native
VLA feature interface. The original DSRL reference retains its published raw
RGB/proprioception encoder and per-task controller, which is an explicit input
representation difference. A feature-matched SAC control may be added if this
difference limits interpretation. Report success at each attempt and against
primitive interactions, frozen first-attempt competence, final performance,
training/teacher-generation cost, and deployment latency separately.
