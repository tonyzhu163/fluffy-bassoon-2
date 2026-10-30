# RoboTwin direct PPO source100 and final development panel

Dated 2026-10-31, Asia/Shanghai. This amends the concrete inventory's direct
PPO rows using the coordinator's audited source100 reports. Both PPO learners
and all declared iteration10/25/50/100 evaluations are complete. This is a
documentation/static artifact update; it adds no implementation, collector,
model computation, test, physical action, allocation or budget authority.

## Completed direct learners and source learning quality

| Learner | Source rounds | Complete K4 | Source episodes | Source actions | PPO updates | Adam steps | Final source validation |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Persistent GTrXL PPO | 100 | 600 | 2,400 | 964,875 | 100 | 500 | 2/24 |
| Independently trained episode-reset GTrXL PPO | 100 | 600 | 2,400 | 963,759 | 100 | 500 | 3/24 |

Both use seed20261030, six complete K4 trials per global update, native task
horizons600/500/400, physical H32x32 noise with steering bound1.5 and the frozen
Motus motor. Scientific lineage is `db0184089fa88a5f96c153f9b80029bb91b77def`,
runtime `57016ba08eed79043fd67bf8b085bd6d42e6d766` and supervisor
`0c16c2b794f8de5b78603206917e72195083c265`. Full source recoveries, ancestry
and original costs remain preserved; neither source is extended here.

Completed optimization does not establish a competent teacher. Persistent
source training succeeded416/2,400 episodes (17.33%); trained reset421/2,400
(17.54%). Consecutive25-round blocks of600 episodes scored116/104/93/103 and
112/106/102/101, respectively. Source validation at10/25/50/100 was1/2/4/2 per24
for persistent and4/4/2/3 for trained reset. Source learning is weak and flat.
The source records contain31,211/31,172 unique valid chunk decisions; repeated
optimizer presentations are a separate quantity. These endpoints are not
qualified as experts and do not establish a memory or architecture benefit.

## Final development results and control meanings

| Controller | adjust_bottle | move_stapler_pad |
| --- | ---: | ---: |
| Native frozen motor, earlier matched16-case panel | 6/8 | 0/8 |
| Persistent PPO, full history | 7/8 | 1/8 |
| Same PPO weights, recurrent-cache reset | 6/8 | 1/8 |
| Same PPO weights, reward blind | 8/8 | 1/8 |
| Independently trained episode-reset PPO | 8/8 | 0/8 |

Each cell is one K8 trial on one development task; episode indices1/2/4/8 have
0/1/3/7 prior episodes. Eight episodes are not eight independent trials.
Full history shows no reliable memory advantage: the reward-blind controller
matches or exceeds normal on these cases, and stapler success is sparse.
One source seed and two development tasks do not establish broad rankings,
held-out adaptation or a statistically reliable improvement over native.

The saved `memory_reset` condition clears recurrent cache at episode start
while retaining explicit previous-noise/reward inputs. It is a cache-reset
lesion, not a fully memoryless policy. Independently trained episode-reset PPO
clears both cache and explicit feedback between episodes, retaining recurrence
within an episode. Reward blindness zeros reward input while other history
remains. These interventions answer different questions.

The earlier native panel is now a verified **cross-run matched initial-case
control** for these16 cases. The owner and coordinator audits compare48
persistent-condition records and16 trained-reset records to native actual
seeds and initial raw-observation/feature hashes; all match. Native executor
`2fd2096906e52aabeaeb984f333fe1aaa6c4e196` and final PPO runtime570 are
distinct, with different execution dates. Underlying capture/motor/norm/backend
configuration matches, but stochastic noise draws are not shared. Native uses
unclipped standard Gaussian noise. This is not a contemporaneous randomized
replication. The small stage-native probes use another seed bank and remain
unpaired references.

Native16 reused5,051 already-counted actions from the earlier panel. This
amendment adds **zero native actions and zero GPU-seconds**; it neither reopens
the earlier reservation nor charges that panel again.

## Separate offline cells and pending work

| Inventory cell | Evidence status |
| --- | --- |
| Direct PPO full and independently trained reset | Source100/final development evaluations complete as above |
| Early SAC5-bank AD/DPT full/static students | Existing four100-update fits and144-episode first panel remain at their original pins; no relabeling as source100 and no merging of fit costs |
| Final SAC100 paired AD/DPT full/static students | Four500-update trajectories, seed0, fixed25/100/500 exports are already prepared; final bank/projection qualification, exact admissions and resources remain pending with the execution owner |
| Independently trained supervised episode-reset AD/DPT | Not supplied by either direct PPO reset or inference lesions |
| GRU/CTM noise-interface cells | Planned; direct GTrXL results do not implement or measure them |

Preserve the [existing final-SAC100 handoff](../robotwin-finalbank-offline-plan-20261031-r1/PLAN.md):
all four fits share admitted source data and the primary frozen endpoint is
`candidate-000500.pt`. Fixed25/100 exports are source-only learning curves;
targets cannot select weights, teacher, source bank or a replacement endpoint.
The final sourceSAC100 checkpoint is preselected as DPT teacher subject to its
separate declared source-only competence measurement. Direct PPO completion
does not add PPO/reset offline data arms or qualify the final SAC bank. R6
code remains frozen at `80ffd1bba1c27ce40cc7f7d36682250e6fe97a07`; actual CPU
qualification and any later resources remain execution-owner admissions.

## Evidence and cost scope

Baige `job-fji2xx8oq53h` reached scheduler/pod Success, exit0. Actual scheduler
Running02:26:01 to Success06:43:32 on2026-10-31 is15,451 GPU-seconds. Pod15,455
seconds is a separate clock. The retained owner audits record176 clean owned
teardown receipts and unchanged source counters during evaluation; terminal
accounting records no automatic failed retries. Source100 and final
recoveries for both arms were independently SHA-checked; each263-member ZIP
passed CRC in the coordinator's saved artifact audit. Both arms have complete
stage10/25/50/100 pointers and `next_input=null`.

[Provenance](ROBOTWIN-PPO-SOURCE100-PROVENANCE-20261031.json) binds exact copied
reports under [source100-evidence-20261031](source100-evidence-20261031/).
It verifies saved JSON hashes, arithmetic, outcome vectors, control pairing
flags and retained checkpoint identities. It does not repeat checkpoint
streaming, canonical forest reconstruction, model deserialization, simulator
execution or scheduler queries. Coordinator result/diagnosis commits are
`a1208f37391f6ffe086ff69b4c5690da8138a4b2` and
`2157984f9ee8a0c58ea997bbedb30e689bae2435`; raw owner evidence is retained there.
Known historical terminal cost totals and the unfinished SAC charge are not
new charges or reusable credit in this amendment.
