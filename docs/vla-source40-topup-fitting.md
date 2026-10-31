# Fit from immutable complete top-up prefixes

`vla_source40_topup_assembly` is an explicit admission route for the canonical
top-up ledger. Legacy `canonical_inputs`, terminal source loading and campaign
checks retain their defaults. No collector, source recipe, student architecture
or fitting hyperparameter changes in this route.

Each dataset generation retains all40 training streams and the original eight
independent validation streams. It selects one latest committed paired AD/DPT
publication per canonical leaf, never overlapping prefix versions. A source
job may be running or partway through a later case. Only the selected complete
K4 prefix enters the snapshot; unfinished later receipts are excluded. During
leaf migration, a missing publication can fall back to its declared qualified
ancestor. A corrupt publication raises an error rather than selecting around it.

Snapshot indices explicitly say `completed=false` for the source run and
`stopping_reason=immutable_complete_prefix`. They do not pretend to exhaust
the physical quota or512-case bank. The normal source loader rejects these
indices. Only the hash-bound all40/fixed8 top-up admission enables the separate
complete-prefix path; whole-case chronology/state partitions, source-only
provenance, frozen-motor/features/noise, AD clock and paired-history checks
still run.

The immutable generation copies the selected checkpoint, unchanged NPZ arrays
and sidecars, actual producer contract, captured original index and compact
physical receipts. The producer may predate the current leaf contract when
its copied ancestor publication is still the latest committed prefix. It is
not rebound to pretend that an old actor trained on later data.

AD retains actual noises and chronological teacher clocks. DPT retains the
actor used for that committed publication, not a future actor from the still
running collection. Costs conservatively include the entire current-leaf
physical charge captured as of this generation, even if later interactions
are not in its selected data. Historical21,253 qualification/control actions
and any new final-competence actions are counted once globally. Unresolved
physical intents remain charged. There are zero new simulator actions.

Once captured, fitting never rereads live indices or reselects later data.
Collectors can keep advancing independently. To fit a later corpus version,
create a new dataset/training identity. Exact resume keeps the same immutable
assembly, dataset checksums, optimizer and RNG; it does not silently replace
training data underneath a student.

On allocated Linux, with the qualified OpenPI/cu128 execution environment:

```bash
python -m fluffy_bassoon.meta_rl.vla_source40_topup_assembly capture \
  --plan "$TOPUP_PLAN" --plan-sha256 "$PLAN_FILE_SHA256" \
  --ledger "$CANONICAL_TOPUP_LEDGER" --output "$NEW_GENERATION/snapshot"

python -m fluffy_bassoon.meta_rl.vla_source40_topup_assembly assemble \
  --admission "$NEW_GENERATION/snapshot/admission.json" \
  --admission-sha256 "$ADMISSION_FILE_SHA256" --output "$NEW_GENERATION/assembly"

python -m fluffy_bassoon.meta_rl.vla_source40_topup_assembly train \
  --manifest "$NEW_GENERATION/assembly/topup-inputs.json" \
  --manifest-sha256 "$INPUT_MANIFEST_FILE_SHA256" --output "$NEW_GENERATION/experiment" \
  --source-commit "$REVIEWED_EXECUTION_COMMIT" --expected-hostname "$ALLOCATED_HOSTNAME"
```

The existing matched milestone trainer fits AD/DPT full/static on each same
immutable corpus:one optimizer trajectory per model, exports at25/100/500,
seed0, batch8, LR3e-4, width128, two layers, four heads, context300 and windows
up to128. Per-controller600 seconds and total2700 seconds remain the existing
bounded recipe. A fitting result is an imitation candidate, not a robot gain.

The integrated launcher performs the actual allocation GPU probe, focused
Linux tests, capture, assembly and fitting, preserving full recoveries:

```bash
bash "$PINNED/scripts/diagnostics/vla_source40_topup_training.sh" \
  "$COMMIT" "$FRESH_RUN" "$TOPUP_PLAN" "$PLAN_FILE_SHA256" \
  "$CANONICAL_TOPUP_LEDGER" 2940 3000
```

The2940-second slice plus60-second startup allowance shares one3000-second
cumulative worker clock including setup/retries. `--resume` is the optional
last argument and retains already committed dataset generations. This script
submits no job and grants no compute. Its allocation must be funded by the
owner; no model, test or launch was run on macOS while preparing this path.
