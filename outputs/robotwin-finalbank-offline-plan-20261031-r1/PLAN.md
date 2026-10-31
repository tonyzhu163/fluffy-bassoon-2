# Final RoboTwin source-bank offline handoff

Status: **Final100 executable preparation; final bank, teacher panel and fits
not admitted or launched.** The fixed closed-SAC65 compatibility path is now
allocated-CPU-qualified at `80ffd1bba1c27ce40cc7f7d36682250e6fe97a07`.
See the [measured handoff](MEASURED-HANDOFF-20261031.md) for actual R6 receipts,
closed CPU accounting, exact SAC100 source/teacher identities and unchanged
remaining resource proposals. The execution owner retains final-bank and
resource admission. This documentation update starts no new work.
The [AD/BC-first amendment](AD-FIRST-HANDOFF-20261031.md) removes the unused
teacher dependency for the existing AD full/static subset. DPT/default paired
execution retains teacher review. New objective-aware code still needs its
necessary allocated CPU regression pass; final62-root input identities are bound.

Later compatibility preparation: production acting-policy paths are absolute.
The [completed bounded SAC65 CPU gate](../robotwin-sac65-cpu-compat-20261031-r6/PLAN.md)
adds a projection-specific checked archive alias and exact source/ancestry
bindings. R6 passed150 fixtures, full reference projection and paired export/
audit. That qualification covers its recorded65-round inputs; it does not
admit the final100 train/validation bank or qualify teacher competence.

The next proposed reference is four matched Transformer fits on the **complete
SAC source100 bank**: AD full/static and DPT full/static. This continues the
earlier SAC5-bank comparison without silently adding PPO/reset data arms.
Source100 admissions for PPO and trained-reset are also specified below;
their data are separate optional later arms, not extra fits in this request.

## Existing code and concrete compatibility findings

Reuse `robotwin_history_corpus` and `robotwin_history_train`, qualified at
`1aeea4f9fcc307bb147def92374bb81a3e2cb50e` (the original converter was9c27).
Integrated reference `2fd2096906e52aabeaeb984f333fe1aaa6c4e196` contains this
native path and the qualified native/shared code. Before any future launch,
review the then-latest main and changes since this reference, select a clean
Git pin and repeat any required compatibility qualification. Never mutate an
active source worktree or assume the old pin is automatically current.

The existing converter expects one flat `episodes/frames/policies` corpus.
It walks global source trial IDs1..committed cursor and reconciles every
selected duration against frozen total/task counters. Passing only the last
continuation root cannot provide the full bank. Symlinking old directories
also fails its resolved-path containment check.

At initial review the owner and optimization owner identified no full assembler.
Reuse the qualified metadata reader at c62be996:
`outputs/robotwin-trained-handoff-control-20261030-r1/control.py:corpus_chain`.
Its exact file SHA is
`67db52fa8f6fa4e8bb56bc6e5cea9f531b5fe41716845350f3372226c5de77a0`.
It returns oldest-to-newest roots and validates hash-bound contracts/migrations,
including a bound backup contract whose bytes equal the canonical file. Call
only this reader, never owner-control or ledger functions. Its header/member
checks elsewhere are not proof of RGB/blob byte preservation.

The owner supplied [SAC65 ancestry](SAC65-REFERENCE.json), SHA
`5e65bc308c8d73ab21ff80452c39fec98b873171e49a1432312a8645bd07d861`:
26 roots ending in closed slice0025,651,638 source actions/390 K4/65 rounds.
The tip manifest SHA is
`8c429b7d7a66b1374ad95a1136b783717509eef0651f457d775c39b59d5f75b6`;
coordinator SHA is
`b09c13f54e4db8994c9a5d94cd6f18b6af3514d4d5ef4e63f1f91e482578b0e0`.
This is an immutable compatibility reference, **not source100**. Its first
ancestor is a selected trial with39 completed rounds/236 complete K4 plus
protected prefix actions; the next root reaches41 rounds/246 K4. Roots are
not one-round shards. Do not truncate to39×6, discard the selected prefix,
renumber trials or count copied prefix actions as new collection.

SAC remains on e825 with DB scientific lineage; it did not migrate to570.
PPO/reset moved from e825 through570 and the prepared0c16 continuation path.
Producer execution pins and original contracts remain per root; the outer
manifest's consuming-executor rebinding does not rewrite producer history.
The26 supplied collection contracts agree on scientific fields after excluding
only executor/resume-manifest identity fields. R6 checked runtime-file equality,
collisions and complete byte projection on this fixed65 reference. Final100
still needs its own complete closed inputs and admission.

## Byte-preserving assembly and admission

[project_corpus.py](project_corpus.py) is the sole new thin assembly helper;
it does not implement another converter, teacher or trainer. It is qualified
on the fixed R6 reference; final100 admission remains pending. It imports the
exact c62 reader by SHA and constructs a fresh
flat projection using hardlinks, with explicit capacity-checked copying only
when cross-filesystem copying is requested. Original files are never edited,
chmodded, deleted or used as an output directory.

Every raw file in `episodes/frames/policies` gets a full SHA, including RGB
payload bytes. Equal path collisions require identical file SHA; unequal
collisions stop before publication. Do not choose the newest file, compare
only NPZ headers, rewrite paths or reserialize records. If real collisions
cannot be represented byte-for-byte, a reviewed multi-root reader extension
and new CPU qualification are required; never force this projection through.
Original collection/runtime files are preserved per-root by hash; projection
top-level files are byte copies of the oldest compatible root, with origin
recorded. Runtime comparisons ignore only `actual_render_pci`, matching the
existing converter. No scientific contract is relaxed or fabricated.

The view retains raw phases, including original evaluation artifacts. Only the
existing exporter's explicit source/validation selection may create training
arrays; target/native/preflight data never become rows, labels or normalization.
Projection alone is not proof of trial coverage or eligibility. The unchanged
converter must validate complete global K4 coverage, actual duration masks,
behavior archives/tensor identities, native causal metadata and total/task
action counters. All partial-prefix/suffix evidence remains available.

Before a final export, owner-hash an admission with:

1. Final source100 snapshot and ordered immutable-root ancestry for the chosen
   arm:600 complete K4/2,400 ended episodes,100 rounds, zero pending PPO
   rollouts, no unresolved physical intents, original schedule/parent/caps.
2. Exact per-method `stages/iteration-100/sac-recovery.pt` hash. It must match
   the closed coordinator's source state/actor identity. The per-method file
   supplies `source_iteration=100`, needed by validation export; a terminal
   coordinator filename alone does not provide that field.
3. Complete stage100 **source-validation** records and matching fixed source
   snapshot. No target-adapted SAC recovery or target replay state is allowed.
4. All producer contracts, runtime/features/norm/motor/horizon identities,
   behavior archives, source counters and case uniqueness reconciled. Keep
   task horizons600/500/400 and actual1024D physical actions intact.
5. A passed allocated CPU assembly/export compatibility gate, owner-reviewed
   teacher qualification when DPT is requested, fresh output identities, capacity/write probes and
   exact source/compute receipts. Live source writers/ledgers are untouched.

Use [projection admission template](projection-admission.template.json).
`--purpose finalbank` rejects a65-round reference. `--purpose reference`
requires separate closed-reference admission and never qualifies a final bank.
Never fill flags from a source filename or change an old receipt in place.

## Source cases, learning histories and selection

Primary training contains every eligible source100 SAC trial in original
chronological task order, including warmup/failed outcomes and real updates.
AD targets equal actual executed steering noise. For each task, both real
update counts and acting tensor identities must change; renamed clocks or
fixed-policy trajectories are not learning histories. The converter already
enforces changing identities and monotonic updates. Do not reorder by success,
filter rewards, replace behavior feedback with teacher proposals, or concatenate
unrelated learners into one history.

Source validation is the separate six-task K4 stage100 panel:24 episodes.
Check disjoint `(task,actual seed)` episode cases as well as full K4 case IDs
and initial observations; consistent reset retries must not introduce overlap.
The trainer's existing whole-case-ID intersection check is necessary but an
episode-level overlap audit is also required. Reused validation seeds across
stages are one source-local diagnostic bank, not fresh independent evidence.
Repeated sampled reset cases within training are retained and reported;
they are not duplicate execution records or permission to drop histories.

Export training as `ad dpt`. Export fixed-policy validation as
`behavior_cloning dpt`, with `--partition validation --validation-iteration100`.
AD validates against explicit actual-action BC; DPT uses the same teacher in
train and validation. Validation cannot be exported as pseudo-learning AD.
Observation normalization uses training observations only. Record train/val
case lists, selected raw costs, policy/version ranges and projection origins.

Preselect the **final source100 SAC snapshot** as the DPT teacher before new
outcomes. Labels remain deterministic physical `1.5*tanh(mean)` on the actual
2062D features. Other recurrent teacher candidates would need their own
causal replay/reset qualification and are not fallback candidates in this
plan. Preserve executed noise, rewards, boundaries, clocks and raw references;
only paired `target_noise` differs between AD and DPT. Actor/version/update
IDs are audit fields, not model inputs. Source-trained does not mean expert.
This retains the existing declared chronological-query DPT-style variant;
it does not introduce a different query-sampling algorithm or relabel history.

## Exact teacher measurement and residual request

[teacher-qualification-plan.json](teacher-qualification-plan.json) specifies
native unclipped Gaussian, frozen SAC100 deterministic mean and frozen SAC100
stochastic sample on the **same six source goals × K4 cases**, using source
validation requested seeds `3,000,000,000 + task_index×10,000 + episode×997`.
Match accepted seeds, initial raw/feature hashes, motor/normalization and
reset rules. No SGD, replay insertion, target task or target-selected teacher.
Keep private interaction RNG conventions and distinguish sampled actions from
mean actions. Report per-task and paired outcomes, actual costs and uncertainty.

Inspection of the exact source driver shows source SAC validation calls
`sac_trial` with its stochastic sampling convention; stage50 frozen/online
SAC also sampled. Those results do not measure mean-action competence. The
existing immutable9-condition/K8 student panel is pinned to SAC5/two targets;
its strict plan cannot be relabeled as a six-source-goal/K4 teacher panel.
The new source-local panel therefore needs an owner-reviewed bounded runner
using existing native/SAC primitives and its own CPU admission. No new policy
algorithm or duplicated exporter is requested. That runner is not implemented
or admitted by this static handoff.

Each condition has a10,800-action maximum: four times the sum of source
horizons2,700. **Default request:72 episodes, at most32,400 NEW actions and
3,600 scheduler GPU-seconds on one GPU**, including startup/probe/reset/failure/
teardown. A conservative case-boundary stop must leave cleanup time and retain
unknown charges; no retry or larger allocation is automatic.

Conditional reuse of the already-paid stage100 sampled source validation
requires the exact source100 checkpoint/actor, `deterministic=False`, no
updates, matching actual initial cases/bytes and audited RNG/sampling. If all
hold, reuse its24 episodes and request only native+mean: **48 new episodes,
at most21,600 new actions and2,700 GPU-seconds**. This reuse is not currently
admitted. Do not subtract a guessed sample cost or double-count the original
validation charge.

The owner's [cost reference](SAC65-COST-REFERENCE.json) pins original SAC
ledger total cap1,194,400/source cap1,100,000, planned stage100 source validation
maximum10,800 and development targets12,800. These are existing obligations,
not extra qualification capacity. No new qualification ledger or residual
authority is selected. The owner must reconcile a fresh stable canonical
snapshot, outstanding source/final-evaluation obligations and actual charged
compute before choosing the32,400/3,600 or conditional21,600/2,700 request.
Formally, admit only if separately authorized residual actions/time cover that
chosen new cap after all outstanding commitments. Record the exact ledgers,
snapshot SHA, reservation and cost provenance. Sealed student-panel/N6 leftovers
are unavailable. No source cap, original goal or target budget changes here.

The teacher stays SAC100; qualification outcomes cannot select a favorable
target endpoint or silently switch mean to sample labels. A weak mean must be
reported as a weak candidate. Any fallback teacher or broader selection needs
a separate source-only review. Equal-to-native ceiling results do not prove
teacher superiority.

## Bounded four-fit proposal

After owner bank/teacher admission, propose one sequential four-fit allocation:
one GPU/8CPU/32GiB RAM/shm8GiB, maximum3,600 scheduler GPU-seconds. Parent
deadline3,300s includes loading, four fits and audits; an outer bounded
TERM/KILL wrapper and actual GPU/CUDA probe are required. Each process has a
750s external bound and600s fitting-clock limit. These are proposed caps,
not measured guarantees for the final bank and not a reservation.

| Objective | Mode | Seed | Total updates | Portable milestones |
| --- | --- | ---: | ---: | --- |
| AD | full | 0 | 500 | 25,100,500 |
| AD | static | 0 | 500 | 25,100,500 |
| DPT | full | 0 | 500 | 25,100,500 |
| DPT | static | 0 | 500 | 25,100,500 |

Retain width128/layers2/heads4/memory300, maximum gradient window128, batch8,
AdamW3e-4/weight decay0/gradient cap1 and4 fixed validation batches every25
updates. Same data/seed/window sampling/normalization recipe across controls;
static clears recurrent state and previous noise/reward/boundary while keeping
current features and clocks. The four trajectories total2,000 updates;
milestones are exports, not extra optimizer trajectories or robot data.

Predeclare **candidate-000500.pt** as the primary frozen endpoint. Keep25/100
for source-only learning curves, not target-based selection. `candidate.pt`
is the existing best-source-NLL diagnostic and must not silently replace500.
NLL is not a comparable task-success metric across AD/DPT. The old SAC5-bank
four100-update models and their144-episode result keep their original hashes
and interpretation; do not call them source100 results or merge their fit cost.

The earlier four100-update job took360.12s of launcher time; a simple5× scale
is about1,801s before extra final-bank I/O. This is an estimate, not scheduler
billing or a speedup claim. Proposed3,600s bounds cost rather than promising
completion. A controlled partial fit reports only committed updates. Preserve
`latest.pt` with AdamW/sampler/Python/NumPy/Torch/CUDA RNG, index/teacher hashes,
all metrics and portable milestones. Exact-contract SHA-bound resume is a
separately reserved future invocation; no restart/retry/budget expansion here.

## Executable preparation and qualification gates

[run_offline.py](run_offline.py) prints exact existing-module command arrays
by default. It launches nothing without `--execute`, an exact allocated Linux
hostname, clean reviewed Git pin, filled hash-bound inputs and owner admission
receipts. [bindings.template.json](bindings.template.json) intentionally has
false admission flags and unresolved paths/hashes. Fill an owner-hashed COPY;
never treat this template as a run manifest. Export and fit are separate phases.
Before fitting, the runner parses the actual indexes and binds their source,
teacher, phase,600/6 K4 counts, source100 iteration, dataset objectives and paths
to the admitted projection/source100 snapshot. It checks dataset SHA and
episode/K4 train-validation disjointness; valid old SAC5/65 indexes cannot be
paired with unrelated source100 references. All four processes use the same
admitted indexes. No reset case is removed to satisfy these checks.

The owner-hashed bindings are the admission authority. Their attestation must
identify the exact phase, allocated host, code/source/projection/index hashes
and resource reservation. Referenced receipt SHA checks establish immutability;
this helper does not independently interpret every scheduler, GPU probe or
reservation receipt schema. This is an explicit owner trust boundary, not a
new approval flow or proof supplied by setting a boolean without evidence.

```bash
# Preparation only: print commands from the static template; no imports/models.
python3 run_offline.py --bindings bindings.template.json \
  --bindings-sha256 <template-SHA256> --phase export
python3 run_offline.py --bindings bindings.template.json \
  --bindings-sha256 <template-SHA256> --phase fit
```

The remaining proposed costs are separate, all currently unadmitted. R1–R6
CPU qualification is completed and accounted separately in the measured handoff:

| Phase | Resource | Maximum scheduler seconds | New robot actions |
| --- | --- | ---: | ---: |
| Final100 projection/export admission | 0GPU/8CPU/64GiB | 7,200 | 0 |
| Source teacher panel, all new | 1GPU | 3,600 | 32,400 |
| Four sequential500-update fits | 1GPU/8CPU/32GiB | 3,600 | 0 |

The default proposed GPU ceiling totals7,200s; verified sample reuse would
reduce it to6,300s. Remaining CPU allocation wall ceiling is7,200s, not GPU time.
These ceilings are not a combined reservation or permission to launch.

For final CPU work, an owner-supervised outer6,900s deadline starts before
projection/admission; pass only its remaining allowance into
`admitted_parent_seconds` for export (never more than6,900). The planning
runner counts its own input/Git checks against that remaining allowance.
Measure projection time separately and do not start labeling with insufficient
time. If the CPU gate or final assembly cannot finish inside its phase cap,
preserve evidence and request a separately reconciled continuation; never add
another cap silently. This is zero new robot data, not zero computation.

The completed closed65 gate supplies reference compatibility. Final100 still
requires its own inputs/admission; the original qualification checklist below
records the intended scope, with R6 results bounded by the measured handoff:

- Focused existing native corpus/policy/trainer tests plus meaningful helper
  fixtures: ancestry cycle/hash mismatch,65-as-final rejection, selected-prefix
  coverage, unequal collision, top-level/nested symlink escape, altered source bytes, partial
  files, explicit cross-device copy/capacity and incomplete-output rejection.
  Include valid production `source_iterations` dictionary acceptance and
  rejection of a valid old-bank index pair attached to unrelated source100
  refs, wrong objectives/counts/paths/projection and train-validation overlap.
- Source-prepared helper's full-file SHA projection and real unchanged
  converter admission on the closed multi-root reference. Reconcile the
  selected-trial prefix and all390 global K4/task/physical-action counters;
  do not assume a root corresponds to one round. No label/fit is run on macOS
  or the development SSH endpoint.
- Actual native PPO/reset vector-fragment compatibility from an owner-issued
  closed chain: causal saved3090 inputs, lane-local history/reset indices,
  fixed per-trial acting weights and original task clocks. This admits those
  later data arms; it does not add their fits to the primary four.
- Paired export fields equal except target labels; source/validation case
  disjointness at episode and K4 levels; validation BC/DPT semantics; bounded
  source-only CPU label/tiny synthetic recovery checks when separately
  admitted. Such fixtures are not final scientific fits or teacher competence.

Helper AST/static checks alone do not qualify assembly, collision freedom,
full raw coverage, teacher quality, fit completion or robot performance.
Final100/teacher/fit reservations and their own receipts remain pending. No new frozen target
evaluation is authorized here: final model hashes require a new reviewed
evaluation plan/admission, never substitution into the old fixed-SAC5 plan or
reuse of its sealed action/time remainder.
