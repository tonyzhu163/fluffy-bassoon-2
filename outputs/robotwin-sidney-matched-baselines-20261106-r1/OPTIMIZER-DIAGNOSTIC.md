# Metadata-only native optimizer roundtrip

The old raw hash differs only for the optimizer, while all recorded weights,
scheduler/scaler/RNG and counters match. Its state/group details were not
persisted before the gate. This diagnostic collects those details before any
training retry. It neither relaxes the gate nor claims restored optimizer
semantics. No gradient step or simulator control is performed.

Fresh original-Sidney headless actors record named parameter ownership/local
shape/numel/dtype, state-entry presence, tensor shape/dtype/device/data hashes/
finite/nonzero/scalar values, ordered groups and typed hyperparameters before
and after native step0 serialization. A separate fresh actor process records
the counterpart before/after loading that same checkpoint and the raw
signature comparison. Results are explicitly diagnostic_complete_unqualified
with optimizer_semantic_qualified=False. The collected delta must explain the
failure before a gradient/full-runtime retry is admitted. Frozen gradient-view,
real-update full hashes and independent full12worker/case recovery remain
required for the scientific arms.

Use4GPU/64CPU/256GiB/64GiBSHM/14min, no retry; reserve15min including startup/
cleanup=3600GPU-sec=1GPUh. New storage16GiB. Keep old closed canonical fixtures.
After grad01's actual836scheduler/888conservative GPU-sec, completed actual is
63,795sec=17.72083GPUh including legacy; completed conservative additional is
61,633. Reserve3600optimizer diagnostic +3600latergradient replay +10,800full
qualifier =79,633additionalsec<23h. Including2561legacy and82,080futurepair gives
164,274sec=45.63167h<the human50h cap. Both scientific arms remain held and the
prospective46additional-hour migration remains inactive. No other budget is used.

Forward storage is16GiB diagnostic +16GiB later replay +40GiB full qualifier
+286GiB other owners/floor=358GiB. Main and closed attempts already occupy PFS
and have no future-growth reservation. No projected cleanup is counted; future
pair admission still requires fresh536GiB after verified eligible cleanup.
