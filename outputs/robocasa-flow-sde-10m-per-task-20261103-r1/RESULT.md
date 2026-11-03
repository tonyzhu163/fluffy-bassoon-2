# Five-task native source results

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

Native all-attempt compute totals 372.7102778 GPU-hours, including failed Q1
and earlier diagnostic upper bounds. Later offline reporting attempts are
additional and listed in EXECUTION and monitor evidence. Campaign archival and
reporting closeout remains in progress: 75 superseded checkpoints are verified
in BOS; final 130 checkpoints stay on PFS. OpenDrawer, Toaster and
Drawer→Counter full corpus archives, QC metadata, readbacks and reclaims are
complete; Coffee and Stove follow serially. OpenDrawer's interrupted cleanup
was recovered with exact archive and inventory checks, preserving all data. The offline mean log-ratio
to initial is complete for all 16 checkpoints each of OpenDrawer, Toaster and
Drawer→Counter; Coffee and Stove reporting is still running. The first two reporting attempts made no
simulator actions, SGD or likelihood forwards.

Evidence: [all 650 joins](COMPLETE-CORPUS-JOINS-R1.json), each task's
FINAL-READOUT-R1.json and FINAL-ACTION-ACCOUNTING-R1.json, and
[execution](EXECUTION.json). Earlier vision-mode and previous-run limitations
remain documented in VISION-MODE-BUG.md. These source runs provide no OOD
adaptation result or student result.
