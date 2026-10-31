# Execution-migration ancestry compatibility correction

Diagnosed2026-10-31 Asia/Shanghai. Static preparation only; no tests, models,
production payload reads, data scans, launches or retries occurred on Mac/devSSH.
This branch starts from immutable7401; its trainer/math/projector are unchanged.
Root owns actual terminal accounting, the single diagnosed residual CPU run,
projection reuse admission and all scheduling.

## Exact cause and correction

The retained final62 ancestry0af5082349ec1a032eff6b752e3b91fd473b8c922a394c496b4f75bceff68c7e
omits execution_migration on its original unresumed root, and has61 direct
{path,sha256} refs. The earlier SAC65 reference uses explicit originalnull and
{ref:{path,sha256,bytes},value:{...}} wrappers. Original files and proof contents
are not missing or changed. The old reader unconditionally indexed the first
field and later the wrapper's ref, so .get alone would not fix this incident.

Only the existing reader and post-index provenance decoder change. Both declared
encodings are interpreted in memory; no ancestry, source contract, migration,
projection, array or original receipt is rewritten. Absence/null is permitted
only for the original root whose producer has no resume reference. Every resumed
root must retain its migration reference. The reader hashes/reads the real
migration JSON, requires canonical root-parent location and exact previous
manifest path/SHA, producer resume SHA, executor, scientific lineage and method.
Legacy embedded values still must equal the actual hash-bound file. Unsupported
mixed/unbound reference shapes reject. c62/header/source actor/counter/root and
acting-policy aliases/coverage checks remain in place. Token/label/mask/math
code is unchanged.

## Prepared allocated-only regressions

New tests/test_robotwin_execution_migration_refs.py:38 cases, not executed here.
One bounded actual0af JSON-shape test; four positive encoding/genesis combinations
export byte-identical scientific AD arrays against the unchanged baseline and
leave the synthetic paid projection/provenance untouched with teacher calls
forbidden. Eighteen direct/wrapped negative proof cases cover absent/null/SHA,
canonical location, previous path/SHA, executor, lineage and method. Also check
legacy embedded-value mismatch, false genesis predecessor, malformed mixed refs,
both index encodings, missing resumed proofs and unrelated provenance paths.

Recommended affected suite in the owner's SAME residual CPU allocation:
- tests/test_robotwin_execution_migration_refs.py:38
- tests/test_robotwin_offline_compat_gate.py:58
- tests/test_robotwin_ad_first.py:24
- tests/test_robotwin_history_corpus.py:17
- tests/test_robotwin_finalbank_index_bindings.py:9
Total146 prepared cases. Actual allocated JUnit is authoritative. No standalone
qualification allocation or baseline rerun is requested outside that residual.
Use the existing0GPU/no-native/no-ledger/no-CUDA/no-optimizer import/owned-runtime
path and actual selected file list/expected count; no .pth execution.

## Reuse and honest pin boundaries

The already paid R1 projection is immutable at
/mowenjie/fluffy-bassoon-2/outputs/robotwin-final100-ad-export-20261031-r1/private/projection.
Its full89MB projection.json remains remote; root binds its actual full-file SHA,
passed complete projection phase and clean closure/cost receipts. No new SHA
is invented here. Reuse requires no ASSEMBLY-INCOMPLETE and exact existing
0af/tip5b/finalf551/stage59588 and original data identity. The reader still
rechecks bound provenance and every actually consumed source input. It does not
require rerunning the unchanged projector or rehashing all excluded raw payloads.
If the complete projection cannot be safely bound, stop for explicit diagnosis
and cost within the remaining approved CPU, not automatic reprojection.

R1 actual scheduler charge889 worker CPU-seconds is separate from parent865.659
and pod895. Prior software170 gives1059 spent of the current7200 CPU plan and
6141 residual; root proposes one6000 scheduler/5700 parent R2,141 unrequested.
Old sealed461/269 are not credit. These numbers are execution-owner reports;
actual receipts and the residual admission remain root-owned. R2 output/inputs
and export directories must be fresh; no old receipt/data worktree is altered.

Use this compatibility commit as the R2 EXPORT consumer and post-index validator;
existing exporter main must receive its actual new source commit to satisfy the
clean Git guard. Record this explicit migration. Root updates only its owned
operational profile/receipts; no alternate converter or teacher gate is needed.

Keep TRAINING code/root/source-commit7401 and READOUTf759 immutable. For later
fits invoke this compatibility commit's existing run_offline.py externally,
with bindings.code_root/code_commit pointing to the clean7401 fitting checkout,
PYTHONPATH=7401/src, and actual bank/index admissions. run_offline.main/command
construction are unchanged and still emit actual7401 trainer commands; its
post-index decoder now understands original0af. Do not patch/copy the runner
into the frozen7401 checkout. The new export consumer/compatibility runner pin
and the actual7401 training pin are distinct, recorded roles. The f759 selector
continues to require plan.fit_commit7401 matching manifest.contract.source_commit,
and the same fixed500 recipe.
No GPU fit or readout is authorized by a partial bank, this static correction,
or a successful byte projection alone.
