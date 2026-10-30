# RoboTwin first frozen panel — 2026-10-31

The native Transformer AD/DPT exporter, fitting and frozen deployment paths
are implemented and qualified. The first paired development evaluation is
**COMPLETE: nine conditions × two tasks × K8 =18 units /144 episodes**.
At this early100-update endpoint, full history shows no advantage over the
static or same-weight episode-reset controls.

| Condition | adjust_bottle | move_stapler_pad |
| --- | --- | --- |
| Native Gaussian | 6/8 | 0/8 |
| Native zero | 6/8 | 0/8 |
| Frozen source SAC5 mean | 5/8 | 0/8 |
| AD full | 5/8 | 0/8 |
| AD static | 5/8 | 0/8 |
| DPT full | 5/8 | 0/8 |
| DPT static | 6/8 | 0/8 |
| AD full, episode reset | 6/8 | 0/8 |
| DPT full, episode reset | 5/8 | 2/8 |

This is one paired K8 trial per development task and condition, one fitting
seed, and100 fit updates per student. All four students used the same early
SAC5 bank:1,539 unique decisions,120 episodes,30 complete K4 trials and47,602
source actions across six goals. Each fit sampled39,621 positions. The DPT
teacher is the fixed feature-conditioned SAC5 deterministic mean,
`1.5*tanh(mean)`; its competence is not independently qualified. This panel
does not evaluate the eventual100-round source PPO/SAC endpoints or all six
target tasks, and it does not establish robust held-out ICRL.

Full/static variants have independently fitted weights. Full/reset variants
share exact weights; episode reset is an inference lesion that clears memory
and previous-action/reward/boundary inputs while preserving episode clocks.
Independently trained supervised episode-reset, GRU and CTM remain absent.

Every condition used the same requested and accepted cases and matching
initial raw-observation/feature hashes. Bottle E5 consistently used fallback
seed22265019 instead of requested22265018. The owner audit found all four
same-weight full/reset E1 traces identical at every decision: stored feature,
next feature, noise, motor target, reward, termination and duration. DPT-reset
stapler successes were E4/E6, after identical E1 failures; they provide no
evidence of an advantage from retaining memory across episodes.

## Execution and closed cost

Baige `job-ekg1ivu6j8pg` completed with scheduler state **Success** at
2026-10-31 00:21:02 Shanghai. The owner terminal receipt records49m11s on one
GPU: **2,951 scheduler GPU-seconds**. Wrapper elapsed2938.385s is a separate
lower-bound timing, not the billed value. Runtime source was
`2fd2096906e52aabeaeb984f333fe1aaa6c4e196`; wrapper and all18 children exited0.

The target-only panel consumed **47,137 physical primitive control actions**
across1,540 decisions, with zero SGD, pending or uncertain charges. The
57,600-action/5,400-GPU-second reservation is closed; unused10,463 actions and
2,449 GPU-seconds are sealed. Source ledgers, caps, jobs and reservations
remain owned separately. The owner reports source collection continuing
under its original100-round authority; this amendment makes no fresh source
scheduler-status claim.

## Evidence and verification limits

The owner's independent audit reports **22,135 checks passed**, zero errors,
all3,100 ledger events reconciled, model/teacher identities matched, renderer
receipts verified and native-noise parity0.0 for all18 units. Parameter
digests were matched to admission rather than independently reconstructed
from model tensors. The pinned runtime checks steering and motor identity
after each K8 before publishing results; no separate literal post-motor hash
field is stored. Raw RGB payloads were not rehashed by that audit.

This amendment independently reconciles the captured JSON's18 unique units,
outcomes, paired reset-case lists, action/chunk sums, model hashes, four E1
control summaries and reservation/scheduler arithmetic. It retains a
byte-identical [compact owner audit](ROBOTWIN-FIRST-PANEL-AUDIT-20261031.json)
and records source-file hashes, checkpoint identities, unit/result hashes,
the terminal scheduler receipt and scope in
[provenance](ROBOTWIN-FIRST-PANEL-PROVENANCE-20261031.json). It does not repeat
the owner's raw artifact-wide audit. Reports were captured from the owner's
local output directory and are not asserted to be committed at its HEAD.

The original reports are in
`outputs/robotwin-native-evaluation-20261030-r1/RESULTS.md` and
`independent-audit/SUMMARY.md` in the RoboTwin execution owner's checkout.
This supersedes the inventory's stale native adapter/student-pending rows;
earlier dated implementation amendments remain historical evidence.

Future parent-death guard
`c166171b847ada9a20a9629a07d5a0100b517097` remains source-prepared and staged
for a separate allocated Linux CPU gate. Completing this2fd panel does not
qualify or deploy that guard. This amendment launches no experiments and
changes no active code, source collection or resource budget.
