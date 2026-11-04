# Five-task native source results

**Completed Nov4 2026.** Native training, reporting and corpus retention checks
are complete; all campaign GPU jobs are terminal.
The scheduled monitor is stopped; [shutdown receipt](MONITOR-END-R1.json).

All five Baige Pro6K jobs completed 130/130 updates and their final paired
64-case panels, exited 0 and reached scheduler SUCCESS. Each task used eight
GPUs, 16 TRAIN / 8 EVAL lanes per GPU and two actor ranks on immutable runtime
`0988b5f9`.

**None passed either preregistered improvement gate. All five corpora remain
reference sources; no improving teacher is admitted and no student fitting has
started.** The TRAIN gate requires at least 10 percentage points from updates
1–10 to 121–130. Paired evaluation requires wins greater than losses and exact
two-sided McNemar p<.05. These metrics use first success; success at end and all
per-update rates remain in each raw readout.

| Task | TRAIN early→late | Δ percentage points | Paired before→after | Wins/losses | p |
| --- | ---: | ---: | ---: | ---: | ---: |
| OpenDrawer | 823→882/1280 | +4.61 | 40→47/64 | 13/6 | 0.1671 |
| CloseToasterOvenDoor | 430→544/1280 | +8.91 | 21→25/64 | 13/9 | 0.5235 |
| PickPlaceDrawerToCounter | 501→521/1280 | +1.56 | 32→24/64 | 6/14 | 0.1153 |
| CoffeeSetupMug | 404→466/1280 | +4.84 | 20→23/64 | 15/12 | 0.7011 |
| TurnOffStove | 205→204/1280 | -0.08 | 9→7/64 | 4/6 | 0.7539 |

Each source contains 16,640 TRAIN episodes, 199,680 decision samples and
**9,984,000 executed robot controls**, plus 76,800 before/after evaluation
controls. All five total 49,920,000 valid TRAIN controls. The fully charged,
failed original OpenDrawer Q1 contributes 76,800 excluded TRAIN controls;
all-attempt policy controls total 50,380,800. There are zero unresolved action
intents and zero constructor reset controls. Each task completed 3,120 logical
optimizer steps; actor-rank copies are not added together.

All 650 cohorts passed the full Linux numerical payload, target, chain, RNG,
three-camera, state, action, reward and boundary joins. Their 650 join and raw
file manifest hashes were rechecked after completion: 83,200 episodes and
998,400 decision samples. Initial likelihood gaps are zero on both actor ranks
throughout; losses are finite and the frozen VLM is unchanged. Final models and
all 18 worker sidecars were hash verified. Physical/public state and case
identities match across paired evaluation; small rendering differences remain
diagnostic.

All GPU work is complete. The [cost closeout](GPU-COST-CLOSEOUT-R1.json)
accounts for **373.9338889 GPU-hours**, including every source, qualification
and reporting attempt. Native work accounts for 372.7102778 hours; the remainder
covers reporting. The ledger distinguishes process clocks from preserved
scheduler upper bounds; it is not a currency bill.

All 80 retained-checkpoint fixed B16 readouts are complete (16 per task), with
zero initial-policy replay error for every task. These signed conditional
transition log ratios use one saved original cohort per task; they are not
full-action marginal KL or teacher-performance gates. Download the verified
[650-row training table](result-tables-r1/training.csv),
[80-row likelihood table](result-tables-r1/initial-policy-log-ratios.csv), and
[five-source summary](result-tables-r1/source-summary.csv).

The [final retention check](FINAL-CLOSEOUT-R1.json) passed on allocated Linux.
All five full raw corpora and QC archives are retained in BOS, with full
object/member SHA256 readback receipts and restore commands. All 75 older full
checkpoints remain in BOS; reporting actually restored and strictly loaded
all 75 archived model payloads. The five newest full step130 checkpoints stay
on PFS, with their model and all 18 worker/optimizer/sampler/RNG sidecar hashes
rechecked at closeout. All 650 join and raw-file manifests were rehashed after
tiering. At the latest check PFS has 740.84 GiB free, above the 200 GiB floor.

OpenDrawer's interrupted cleanup was recovered with exact archive/inventory
checks, preserving all data. The first two reporting attempts made no simulator
actions, SGD or likelihood forwards. No student fitting or DPT label queries
were started. Further fitting or OOD PPO comparisons require their own plan.

Evidence: [all 650 joins](COMPLETE-CORPUS-JOINS-R1.json), each task's
FINAL-READOUT-R1.json and FINAL-ACTION-ACCOUNTING-R1.json, and
[execution](EXECUTION.json). Earlier vision-mode and previous-run limitations
remain documented in VISION-MODE-BUG.md. These source runs provide no OOD
adaptation result or student result.
