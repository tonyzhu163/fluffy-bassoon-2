# Small-prefix AD/DPT on a frozen 4D steering interface

`fluffy_bassoon.meta_rl.steering_history` is an admission/projection adapter for
the existing `vla_history.HistoryNoisePolicy` and `vla_history_train` learner.
It introduces no collector, learner, logger, simulator loop or scheduler.
All arrays, model execution and tests run on allocated Linux. A useful immutable
completed prefix can be fitted while collection continues into new episodes.
The admitted prefix, labels, teacher and fitting identity never change in place.

## Coordinates and canonical records

An interface binds Motus/RLinf H32, five flow steps, frozen motor and feature
digests, one selected noise/KV/decoded-action mapping, fixed bases/scales/units,
temporal interpretation, bounds and gripper rules. Student outputs are four
requested coefficients after the declared bound and before hidden actuator
calibration. The student observation is the native 2062 public features plus the
previous actually valid pre-calibration 14D request, yielding 2076 features.
The command suffix is zero at EVERY physical reset. Full memory can retain
earlier support tokens; static inputs cannot carry the previous support command.
Native noise, post-calibration commands, condition, teacher version and reset/case
IDs remain audit-only. DPT/SAC teachers receive the native 2062 slice only.

The control driver's `student_observation` owns the online projection and reset
state. `DriverCoefficientStudent` uses that exact callback in the existing
`run_complete_episode` loop. Offline projection checks the logged typed previous
command against `frozen_pi05_residual.public_agent_requested_actions` at the LAST
true native validity-mask index. It does not assume duration-minus-one or the
last proposed command. Without a hidden transform, the issued commands are the
public request; a hidden actuator command is never used as its replacement.

The existing raw corpus remains canonical. `canonical_trial` uses `CorpusReader`
to verify complete episode manifests, hash-bound chunks/arrays, connected frames,
actual reset identities, durations/masks and acting-weight archives. It reads
these typed `decision_metadata` keys:

| Key | Meaning |
| --- | --- |
| `steering_coefficients` | Actual requested float32 `[4]`, including warmup |
| `steering_mapping_sha256` | Exact basis/scale/mapping binding |
| `steering_interface_sha256` | Driver's native control interface hash, retained as `control_interface_sha256` |
| `student_previous_agent_command` | Driver-owned causal float32 `[14]` request suffix, zero on physical reset |
| `steering_lineage_id` | Audit-only lineage identity |
| `steering_condition_id` | Audit-only persistent condition |
| `steering_partition` | Audit-only `train` or `validation` role |
| `optimizer_updates` | Actual source update count before the decision |

Keep existing `source_warmup` and actual `sac:update:<n>:<actor_sha256>` context
with its acting archive. Updates occur between complete episodes. Warmup remains
warmup until an update has completed; a clock or a final changed checkpoint alone
does not prove that learning actors supplied the recorded behavior.

`frozen_vla_4d_source_prefix_v1` binds six whole lineages: two TRAIN and one VAL
seed for each of two conditions. Each lineage lists its canonical `corpus_root`,
initial/final complete NoiseSAC recovery references and actual update counts,
plus complete episode manifest references, actual seeds, canonical initial frame
content hashes and physical case IDs. Select the same completed episode cursor
across all six lineages before making windows; the collector publishes initial
12/24 boundaries, with 48 only an upper cap. The sampler takes at most nine
adjacent complete episodes, so a query sees at most eight earlier complete
episodes. Every such window must fit declared Transformer memory. Its first
token clears feedback from any unseen prior episode.
Split lineages and physical cases/seeds/observations before windows. Locked query
cases and teacher-qualification cases stay outside source fitting data.

## Independent objective gates

AD requires changing actors that actually acted, chronological update identities,
complete resumable recoveries and measured source learning above the predeclared
paired improvement threshold. Preserve failed/non-improving lineages and report
the failed gate; do not filter them into a successful-looking corpus. AD labels
equal the actual requested coefficients. Its independent VAL lineage is also a
learning trajectory; it is an immutable held-out collection, not a new teacher
chosen using student validation loss.

DPT does not require the behavior learner to improve. It does require a qualified
per-condition teacher in the exact selected 4D mapping. The primary implementation
accepts either the predeclared final TRAIN NoiseSAC actor or an explicit constant
4D search/inverse oracle whose same-basis physical execution is evidenced. An
external 14D actuator inverse is not a KV/noise coefficient label. If no qualified
same-mapping expert exists, DPT admission is unavailable for that interface.

One condition teacher labels both TRAIN and VAL contexts. Its qualified immutable
TRAIN snapshot may postdate the behavior prefix; that does not expose future
information as student input. A VAL learner supplies
held-out imperfect behavior, not a different labeling actor. Record the full
recovery hash separately from the verified actor tensor digest. SAC labeling uses
a separate fixed private Gaussian RNG in sorted-lineage/chronological-row order;
it never consumes or changes source interaction/optimizer RNG. The oracle receipt
records its exact coefficients and provenance. Relabel only `target_noise[t]`;
all earlier behavior, observations, rewards and boundaries remain actual.
Condition is used only to select the offline label teacher.

Qualification receipts reference existing predeclared physical A2/teacher VAL
evidence and the fixed paired outcome panel. They are not requests for additional
qualification jobs. AD improvement and DPT competence are checked separately.

## Matched fitting and final candidate admission

`export --objective ad|dpt` produces the portable existing history NPZ/JSON format
and an immutable assembly. `fit-commands` emits two calls to the existing learner,
full and separately trained static, with identical source data, normalization,
seed, Gaussian NLL, batches, optimizer schedule and update endpoint. Identically
sampled complete-episode windows provide identical supervised positions. The
initial profile uses a small
one-layer Transformer; it does not instantiate the fixed native 1024D policy.

`--checkpoint-selection final` exports `candidate.pt` only at the predeclared
update endpoint. Fixed held-out windows still produce validation NLL on the
predeclared schedule. A wall-clock partial fit keeps the complete optimizer,
sampler and RNG checkpoint but cannot enter robot evaluation. `load_student`
checks the endpoint, architecture, seed, schedule, source commit and both dataset
hashes/metadata. The generic learner's default best-validation behavior is
unchanged; resume contracts now also bind schedule and checkpoint selection.

The learner fits an unbounded diagonal Gaussian coefficient proposal; the motor
adapter clips to the fixed coefficient bound. NLL is therefore a declared
Gaussian coefficient surrogate, not the exact clipped-action density. Deployment
samples the learned Gaussian; it does not replace the policy with an MSE mean.
Normalization uses TRAIN observations only. All query weights are frozen.

## Four-block mechanism readout

Before opening queries, bind two conditions by two fresh scenes, paired reset and
action seeds, the selected task's real horizon, and the `student_evaluation`
action slice in the sole owner's shared phase ledger. `evaluation_plan` fixes:

| Component | Physical episodes |
| --- | ---: |
| AD/DPT full, one four-support prefix for each of four blocks | 32 |
| Independent full queries at K0/1/2/4 from cloned prefix snapshots | 32 |
| K4 same-weight reset/reward-blind/reward-shuffle/wrong-task queries | 32 |
| Separately trained static AD/DPT, matched K4 queries | 8 |
| Native baseline queries | 4 |
| Qualified stationary teacher, four retries plus query per block | 20 |
| Fresh wrong-task donor supports, four per full model | 8 |
| Total | 136 |

This is at most 81,600 primitive actions at horizon 600 with zero reset policy
actions; otherwise add 136 times the verified reset-action bound, inside the 105,000-action
student slice; settling, failed/unresolved calls and cleanup still consume that
slice. It is not 32 cases multiplied by every method and K. Teacher/source costs
are counted once in their own shared buckets, not charged anew by projection.

Collect each four-support prefix completely and save empty/K1/K2/K4 snapshots
before running any query. `capture_state`/`restore_state` clone memory, prior
feedback and private RNG; reseeding query draws preserves support memory. Query
feedback cannot affect later supports or another query. Reset clears all memory
and previous-feedback fields. Reward blind erases reward feedback only. Consistent
reward shuffle permutes rewards within the completed support trace, retaining
observations, requested actions and boundaries. Wrong-task replay uses newly
collected donor supports from the other declared source candidate task; it never
imports task57 or the old source100 bank. Donor cost is counted once even when
memory is cloned for independent queries. Record degenerate all-equal reward
shuffles honestly. Static and stationary retry remain separate controls.

`query_nll` reports query-only NLL by K on actual query behavior, with targets
excluded from feedback. Report the exact-matched-public-feature K0 label groups
in the DPT assembly: different condition teachers may imply an ambiguous mixture;
stochastic teacher draws add variability too. Neither a Gaussian mean nor low
NLL demonstrates latent identification. Report actual queried outcomes, support
and query costs, clipping and inference entropy beside NLL. Four case blocks are
a mechanism pilot. The separate locked 32-case confirmation and independent
seeds remain required before a reliable adaptation or generalization claim.
