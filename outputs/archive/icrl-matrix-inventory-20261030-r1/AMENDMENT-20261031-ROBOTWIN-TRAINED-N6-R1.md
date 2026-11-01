# First trained N6 source continuation — 2026-10-31

The trained batched execution path completed one additional source round for
each original PPO arm, with full recovery and clean teardown. This is
execution qualification under the existing100-round goal, not another
algorithmic evaluation or a completed source-training endpoint.

| Original learner | Source rounds | Complete K4 trials | Cumulative source actions | PPO updates / optimizer steps |
| --- | --- | --- | --- | --- |
| Persistent PPO | 52→53 | 312→318 | 499,022→508,540 | 52→53 /260→265 |
| Separately trained reset PPO | 54→55 | 324→330 | 516,264→525,625 | 54→55 /270→275 |

The two arms added **18,879 physical source actions,12 K4 trials,48 episodes
and two real PPO updates**. Per-arm action increments are9,518 and9,361.
There were611 recorded decisions and ten optimizer steps. No target or
validation actions, fresh learners or automatic retries were introduced.

## Measured execution cost and correctness

Baige `job-fcxa9mm1mh84` ended Succeeded at01:14:30 Shanghai on2026-10-31,
charging **450 scheduler GPU-seconds** on one96GB-class GPU. Runtime was
`57016ba08eed79043fd67bf8b085bd6d42e6d766`, launcher
`c353970a316837b13277ec65587e58e5cb25ce56`, with original scientific lineage
`db0184089fa88a5f96c153f9b80029bb91b77def`.

Full parent runtime was439.776s, including admission, probe, startup,
collection, corpus I/O, real updates, cleanup and final audit. The charged
rate is41.95 source actions per GPU-second. Peak sampled GPU use was
70,387,761,152 bytes =65.55GiB /70.39 decimal GB, with trained Adam states
restored and actual updates performed. Historical scalar runs used different
cases and timing boundaries; no paired speedup ratio is established.
Unused2,250s from this operational reservation are sealed.

Both real native-noise parity checks passed with max error0.0. All twelve
trial likelihood/value replay checks were exactly0.0 within the unchanged
tolerances. Six-trial/five-epoch/LR1e-6 update guards passed. Durable group
acknowledgments and final coordinator recoveries retain optimizer, sampler,
RNG, task schedule, completed stage50 units and source/target separation.
Pending PPO rollouts and canonical pending/uncertain action charges were zero.

The owner allocated and independent audits passed. Both final coordinator
file hashes match manifests, with all263 ZIP members per checkpoint passing
CRC. Native-mask inspection of all611 new decisions reproduces the action
increments. Both drivers exited0; saved cleanup receipts show no owned
survivors and zero remaining GPU context memory. The frozen motor was
unchanged. This short run does not establish indefinite simulator stability.

## Evidence and remaining scope

Owner evidence is committed/pushed at
`48461f4cae4a95f498efa4be50d8c941d1258961` in
`outputs/robotwin-trained-pair-20261031-r1/`. A byte-identical
[independent audit](ROBOTWIN-TRAINED-N6-R1-AUDIT-20261031.json) and
[provenance](ROBOTWIN-TRAINED-N6-R1-PROVENANCE-20261031.json) retain source
hashes, plan/runtime identities, before/after counters, final recovery and
manifest hashes, native-mask audit scope, cleanup and terminal scheduler cost.
This amendment reconciled captured JSON and hashes; it did not repeat
checkpoint/NPZ inspection, run tests/models or access active ledgers.

The owner's dated durable source-bank snapshot is1,575,051 source actions /
972 complete K4 trials /3,888 episodes across the three original learners.
This is an owner snapshot, not a fresh ledger query or new cost charged here;
target clone updates are excluded. A second slice is reported running under
the same qualified execution contract. No completed second-slice result or
new optimizer update is inferred before full-round recovery/audit. Any
continuous-supervisor implementation has its own qualification/admission.

Source-stage50 and the early SAC5-bank AD/DPT student panel retain their
separate scientific results, including no observed memory advantage. Their
captured stage50 input hashes are unchanged and are now also committed in
the owner evidence above. This c353/570 training run does not establish live
deployment of the separately CPU-qualified c166 frozen-panel guard.
Original100-round scope/caps remain binding. This amendment launches no
pilot, borrows no holds and changes no source run, budget or checkpoint.
