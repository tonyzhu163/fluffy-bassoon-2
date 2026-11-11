# Frozen BPP motor qualification on LIBERO-Gen

The released frozen Behavior Prompting Policy executed both bowl-to-plate
choices successfully on all 20 paired native initializations: **20/20 center
bowl and 20/20 near-plate bowl**. This qualifies the two native motor choices
for further task construction. No history reader was trained or evaluated,
and no hidden-rule or custom relational layout was executed.

| Panel | Device reported by CUDA | Center | Near plate | Scope |
| --- | --- | ---: | ---: | --- |
| Runtime smoke `qual04` | GeForce RTX 4090 | 1/1 | 1/1 | Initialization0 |
| Engineering `panel02` | GeForce RTX 4090 | 5/5 | 5/5 | Initializations0–4; smoke scene reused |
| Expansion `panel03` | NVIDIA RPBZZZ6, Pro6K pool | 20/20 | 20/20 | Initializations0–19 |
| Expansion repeated subset | Same panel03 allocation | 5/5 | 5/5 | Initializations0–4 |
| Expansion new subset | Same panel03 allocation | 15/15 | 15/15 | Initializations5–19 |

These rows overlap and must not be pooled as independent successes. The
new subset is a motor development check within the same released task
family, not a held-out rule or a released benchmark score. Twenty cases per
candidate do not establish near-perfect population reliability.

## What was qualified

The exact tasks are
`pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate` and
`pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate_2`, both
in `libero_spatial_selected_combinations_inverse_view`. They share physical
BDDL content but reward different bowl instances placed on the same plate.
We copied the center task's native initialization bank under both task names
and checked matching initial RGB, wrist RGB and proprioceptive hashes for
all 20 cases. Each of the 20 initial observations was distinct.

All 967 released checkpoint tensors matched exactly after strict loading;
parameters remained frozen through both candidates. The parent agent independently
validated all 40 per-scene native outcomes, 20 paired resets and the strict
checkpoint/clock receipts after retrieval. The unchanged official
runner and its pair-prompt interface executed the motor, with one full
successful demonstration prompt per candidate, reused across its scenes.
The checkpoint ignores language and goal-image observations. Native success
traces agree with every per-scene score. The runner stops on the selected
skill's native success. This is appropriate for motor qualification; the
[custom-layout contract](CUSTOM-LAYOUT-NEXT.md) separately removes hidden-goal
termination leakage before a future reward intervention.

The model/runtime code was qualified at `8e526743` and unchanged in panel02
(`1b4b9b1f`) and panel03 (`230978e9960be5cd02952db5bea497b3d6caf0ed`).
BPP source is `58125e682acceefef3d60c071212dc483a4db1c9`; LIBERO source is
`8f1084e3132a39270c3a13ebe37270a43ece2a01`. The checkpoint SHA256 is
`74e0f841f3b86f1383adda23dab59fddee8806a4bcb5685ed7f826afb320c097`.
Three guarded construction/API repairs preserve the released architecture
and tensors; [construction receipt](panel03/motor/construction-receipt.json)
and [runner receipt](panel03/motor/runner-interface-receipt.json) record them.

The allocated GPU probe verified CUDA arithmetic, allocation visibility and
own-process GPU mapping. Pro6K's actual driver/CUDA name is `NVIDIA RPBZZZ6`
with 97,887 MiB reported memory; we do not replace that observed alias with
an assumed product label. Native rendering and inference completed 40 real
episodes on it. The first smoke videos were independently reviewed by the
parent agent: the two prompts grasped different intended bowls from the
same scene and moved each onto the plate, consistent with native success.

## Evidence and cost

[Panel03 readout](panel03-readout.json) contains every native score, step
count and repeated/new subset. [Raw panel03](panel03/) retains reset/action
records, native reward traces, checkpoint configuration, construction/probe
receipts and clocks. All 40 panel03 videos remain on shared storage and in
the downloaded/local archive, outside Git under the retention policy;
[EVIDENCE-MANIFEST.json](EVIDENCE-MANIFEST.json) records every hash and location. [Panel02 readout](panel02-readout.json)
and [raw panel02](panel02/) retain the prior engineering panel. No checkpoint
or heavy prompt cache was downloaded into Git; remote weights, source
checkouts and shared runtime remain intact.

| Identity | Scheduler seconds | Conservative seconds | Outcome |
| --- | ---: | ---: | --- |
| qual01 | 410 | 577 | Construction failure, no episode |
| qual02 | 270 | 453 | Strict-load legacy default failure, no episode |
| qual03 | 290 | 454 | Historical runner API failure, no episode |
| qual04 | 451 | 466 | Two native smoke successes |
| panel01 | 0 | 0 | Cancelled before allocation after 610s queue wait |
| panel02 | 831 | 841 | Ten native successes |
| panel03 | 391 | 421 | Forty native successes |
| Total | 2643 | 3212 | **0.734 scheduler / 0.892 conservative GPU-hours** |

All jobs used one GPU. Conservative accounting uses submission-to-final
terminal status for allocated jobs, including startup, failure/tolerance
waits and teardown. CPU preparation consumes zero GPU-hours. Panel03 ended
at 18:13:52 Shanghai on 2026-11-11; its process clock exited 0 after 380s.
No jobs remain active and no further allocation is planned under this
completed motor stage. The hard cap was 5 GPU-hours.

The next written proposal qualifies 16 custom layouts and 32 physical
executions, native predicate parity and an outcome-independent recording
interface. It has not launched. Learned feedback use requires that admission,
a separately trained then frozen reader, and paired true/masked/shuffled
outcomes on identical reward-independent histories. The binary version can
only establish one-bit task identification with motor noise; broader rule
inference needs at least three overlapping hypotheses and held-out compositions.
