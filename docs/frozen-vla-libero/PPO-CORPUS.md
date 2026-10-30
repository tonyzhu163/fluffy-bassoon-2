# Paired AD/DPT-style data from source40 PPO

`vla_source40_ppo_corpus` converts the existing single-task or shared source40
PPO collector artifacts into the standard `vla_history` datasets. The output
can be read directly by `vla_history_train`; it does not enter the SAC-specific
all40 assembler or silently replace a campaign's declared training data.

## Inputs and causal labeling

Pass a source run's `experiment/` directory, the SHA256 of its complete-trial
recovery, and an explicit exported source-training PPO teacher with its SHA256.
Training recovery is `learner/latest.pt`; frozen source-validation recovery is
`validation-recovery.pt`. The exporter enumerates only trials committed by that
recovery, verifies their raw episode sidecars and behavior provenance, and
rejects changed inputs. Use immutable snapshots when a collector is advancing.
Uncommitted/interrupted trailing episodes are not additional learning data.

The teacher must use the same official OpenPI40 motor, feature contract and
bounded32D noise interface. It must come from source training, never target
adaptation or validation. Its competence remains an independently measured
property; neither a final checkpoint nor finite labels establish expertise.

For each original source trial, replay actual observations, previous executed
noise, previous reward and boundary flags through the frozen teacher. Start
with empty teacher memory at that trial's start. The teacher's saved full versus
episode-reset mode determines feedback and memory resets, even if the source
behavior used another mode. Never load an old behavior-policy cache into the
teacher and never feed teacher proposals back as historical executed actions.
Current reward and future observations are unavailable to the current query.

The first labeling convention is deterministic `tanh(mean)` from the teacher's
squashed Gaussian. It is explicit in the manifest and needs its own performance
qualification. Labels are bounded noise coordinates, not pre-tanh samples,
critic values, advantages or log probabilities. No simulator step, VLA forward
or model update is needed: the frozen public features were already recorded.

## Learning histories and objectives

AD targets are actual executed coordinates. DPT-style targets are the frozen
teacher's proposals. Observations, executed coordinates, rewards and episode
boundaries are bitwise identical between paired datasets; targets never enter
the context builder. This is a paired chronological-query DPT-style objective,
not the original paper's independent-query data construction. A shared
recurrent teacher is also a declared policy-distillation variant rather than
an assumed task-optimal oracle.

Each original PPO trial has fixed behavior weights. Group chronological trials
of the same source task into a learning history, preserving their original
episode boundaries. Teacher memory is rebuilt separately per original trial;
the student's source history spans the grouped trials, as declared in metadata.
Shared-task streams never concatenate different goals into one history. Actual
task coverage, original trial/state/seed lists and behavior update IDs are saved.

AD requires at least two behavior-update identities for every exported task.
A first round-robin pass with one trial per task is insufficient. The exporter
rejects that AD request instead of dropping tasks or inventing update IDs.
Use explicit `--objectives dpt behavior_cloning` for fixed-policy data. Frozen
source-validation traces similarly support DPT/BC, not a fabricated changing
AD history. AD validation needs separately declared changing-policy histories.

## Export and train

Run inside an allocated Linux environment using its qualified Torch runtime:

```sh
python -m fluffy_bassoon.meta_rl.vla_source40_ppo_corpus \
  --source-run /snapshot/source/experiment \
  --recovery-sha256 <recovery-SHA256> \
  --teacher /snapshot/training/learner/candidate.pt \
  --teacher-sha256 <teacher-SHA256> \
  --objectives ad dpt --output /new/paired-ppo \
  --device cpu --threads 4 \
  --source-commit <exporter-commit> --expected-hostname <allocated-host>
```

For a frozen validation source, use its recovery SHA and `--objectives dpt`.
`index.json` records exact input hashes, teacher provenance, case/partition
identities and the dataset paths/hashes. Supply the exported train and validation
`dpt.npz` paths and hashes to `vla_history_train --train-data ... --train-sha256 ...
--validation-data ... --validation-sha256 ...`, with the existing mode and
training-budget arguments. It retains its checks for disjoint histories and
matching motor, features and noise contracts. Split whole source cases before
window extraction; writing two exports of one history does not create a split.

Collection costs are shared once across paired objectives. Recovery counters
cover committed learner trials only; the collector's full physical journal
continues to account for qualification, settling, interrupted work and retries.
Exporter metadata separately records zero new simulator actions and VLA queries.
