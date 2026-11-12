# Existing transformer results recovered on November 12

All four objectives (AD, DPT, DIT-style, IC-IQL) finished fitting with both KV
and velocity interfaces: eight six-layer transformers, with saved selected
candidates. Their joint training tasks were OpenDrawer, CloseToasterOvenDoor
and CoffeeSetupMug. The newer confirmed PPO teachers are Drawer, Toaster and
Faucet; these are different three-task sets.

| Frozen-expert transformer | Training | Original 50-case ID/OOD evaluation |
| --- | --- | --- |
| AD, KV and velocity | Complete | Complete, 6,000 episodes total |
| DPT, KV and velocity | Complete | Incomplete; recovered KV Drawer pilot below |
| DIT-style, KV and velocity | Complete | Incomplete; recovered KV Drawer pilot below |
| IC-IQL, KV and velocity | Complete | Held; no complete simulator comparison recovered |

The [eight-fit index](../robocasa-three-task-kv-velocity-meta-20261105-r1/finalized-fits-20261106-r1/INDEX.json)
records the selected candidates. The [AD readout](../robocasa-three-task-kv-velocity-meta-20261105-r1/STAGE1-RESULT.md)
covers three ID tasks plus held-out Fridge/Microwave and finds no consistent
history benefit. The full eight-model evaluation was held near the historical
150 GPU-hour ceiling. Existing direct-action AD/DPT and later GRU-based DPT
experiments are separate comparisons.

## Recovered November 9 KV pilot

The last local execution note stopped during startup, but object storage
contains the finished DIT full, DPT reset and DIT reset banks. The preceding
job contains DPT full. All four original raw tar archives passed fresh full
SHA256 readback on Baige Linux. The reducer verified binding hashes, 48
completed episodes per arm, eight identical paired cases, original seeds,
600-control horizons, frozen parameters, zero optimizer updates and exact
reconstruction of each success curve. [Audit and raw archive locations](recovered-pilot/AUDIT.json).

| Method | Full-history query successes, supports 0–5 | Reset query successes, supports 0–5 | Sixth-episode paired wins / losses |
| --- | --- | --- | --- |
| DPT KV | 7, 6, 6, 7, 7, 4 / 8 | 7, 6, 6, 7, 7, 5 / 8 | 0 / 1 |
| DIT KV | 8, 6, 6, 6, 7, 5 / 8 | 7, 5, 6, 5, 7, 5 / 8 | 0 / 0 |

This is an eight-case OpenDrawer deployment pilot, not the 50-case three-task
ID panel or an OOD comparison. Neither method gains successes from history
on the final query of this pilot. DIT already has a first-query discordance,
before cross-episode history exists. These small descriptive results do not
establish a general null or support choosing a winner among all four methods.
No new simulator inference was needed to recover them.

## Remaining readout

Reuse the saved KV AD reference and finish KV DPT, DIT and IC-IQL on the
original five tasks, 50 cases × six episodes × full/reset: 9,000 additional
episodes before removing any scientifically reusable complete banks. The
historical AD timing estimates 3.718 GPU-hours ID plus 2.696 OOD per model,
or 19.242 hours for all three before setup, checks and export. A bounded
reservation of 22.5 hours (three one-GPU jobs of at most 7.5 hours) is the
concrete full-panel scope. It cannot fit the old remaining allowance.
Reconcile the November 9 scheduler charges and obtain an explicit increase
of the old 150-hour ceiling before admitting all three full panels. The
new-work ceiling remains 50 GPU-hours across the readout and teacher pilots;
24 teacher + 22.5 evaluation leaves 3.5 for failures and qualification.
Do not promise a six-task refit from that remainder.
