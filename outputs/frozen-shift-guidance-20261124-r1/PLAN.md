# Source guidance: pick the frozen policy's sample nearest a source success

Lane of the [proposal-shift program](../frozen-vla-proposal-shift-20261124-r1/PLAN.md). Written November 24 before any result in this directory. Base: `5fd6c426a8ea1ab2b75fb54d45c3d95357f66d7a`. Only archive formats were inspected before writing (which fields each tape holds, the gripper-command pattern and the archived camera-gate values of the four IIWA raw tapes, tool rotation range on Panda 14).

## Question

At each five-control replan, draw K=4 proposals from frozen π₀.₅ (the untreated noise plus three more). Does executing the one whose five-control displacement is closest to what a successful source arm did at the same public progress rescue Kinova3 failures on drawer-and-bowl, when the source successes come from other layouts?

## Prediction

1. CPU, archived tapes: a donor segment aligned by public progress predicts the next five controls of another source success (different layout) with lower displacement error than the segment at the same control index.
2. Native (not runnable here): guidance rescues some of layouts 14, 15, 17, 36, 37, 38 beyond the repeat's discordance and beyond the random-index branch.

## What counts against it

- CPU: progress alignment is no better than time alignment, or no better than a random donor window, on held-out source successes. Then "aligned source segment" carries no usable target and the native arm is not worth its compute.
- CPU: archived Kinova3 failures are no farther from the aligned donor than archived Kinova3 successes. Then distance to the donor does not separate the behaviours.
- Native: guidance equals the random index among the same K, or only the same-layout donor rescues.

## Prior work checked

- [Donor replay](../frozen-transfer-donor-20261124-r1/RESULT.md), [validation](../frozen-transfer-donor-validation-20261124-r1/RESULT.md): whole IIWA tapes replayed open loop on Kinova3. Same-layout donors rescued 0/2; IIWA 35 rescued 37, IIWA 34 rescued 14. Fixed tapes, no policy samples.
- [Source-demonstration handoff](../frozen-transfer-demonstration-20261124-r1/PLAN.md): issues the donor's own commands up to the first grip close, then hands off. This lane never issues a donor command. Every executed window is one of the frozen policy's own samples at the current observation; the donor only ranks them.
- Embodiment-Classifier Guidance ([arXiv 2610.22576](https://arxiv.org/abs/2610.22576), as cited in the [selector README](../frozen-vla-selector-20261124-r1/README.md)) steers frozen π₀ with a classifier trained on successful target rollouts. This lane uses no target success: donors are IIWA and Panda only, and Kinova3 successes appear only as CPU audit queries.
- Sampling lane of this program: a source-trained critic over K proposals. Difference: nonparametric nearest-donor target, no learned critic.
- [Selector signals](../frozen-vla-selector-20261124-r1/signals.py): the camera drawer reader reused here.

## Protocol

**Donor library** (`donors.py`, archived tapes verified by SHA-256):

| Donor | Archive | Has |
| --- | --- | --- |
| IIWA 34, 35, 36, 37 raw | [cue native tapes](../frozen-transfer-donor-20261124-r1/inputs/raw-tapes.tar.gz) | issued commands, archived camera gate per replan |
| Panda 14, 15 | [pose replay](../frozen-transfer-pose-20261124-r1/evidence/replay/), [drawer replay](../frozen-vla-reachability-20261123-r1/evidence/) | commands, tool rotation, jaw aperture, drawer joint |

**Public progress.** Phase key = (drawer open, jaw commanded closed). Within a phase, the coordinate is the summed issued translation command since the phase began. Online, drawer open is the selector's camera reader on `agentview_image` against the reset frame, and jaw state is the sign of the last issued gripper command. Donor windows carry the same key. No clock and no simulator state enter online.

Deviations from the lane brief, fixed here: (a) IIWA tapes archived no aperture, so the jaw flag is the issued command sign for every arm; aperture on the Panda tapes audits that flag. (b) IIWA drawer flag is the archived camera gate (the earlier reader). Panda tapes archived no camera frames, so the Panda donor flag is the archived drawer joint past the native 0.14 m threshold the reader is calibrated to. That is donor-side annotation from simulator state and is labelled as such. (c) Layouts share base placement and an IK-aligned reset tool pose, so displacement is compared in the tool frame initialised at reset: `D(t)ᵀ Δp` with `D(t)` the public rotation since reset. IIWA tapes have no orientation, so the default uses the reset frame (`D = I`) on both sides. The rotating frame is a variant available with Panda donors and is compared on CPU.

**Alignment.** For a query at a replan, each donor contributes the window with the same phase key whose within-phase coordinate is nearest; if the donor never visits that key, it contributes nothing. Target = that window's summed five-control translation. Score of a proposal = distance from its first-five summed translation to the nearest donor target. Choose the minimum. If no donor has the key, keep index 0 (the untreated sample) and record the fallback.

**Adapters** (`adapter.py`, `ADAPTERS` for the shared panel; `run_guidance.py` builds the layout-aware set per start because the harness withholds the layout id):

| Name | Donors | Choice |
| --- | --- | --- |
| `guidance` | all donors from layouts other than the query (driver) | nearest progress-aligned donor |
| `guidance_iiwa3435` | IIWA 34, 35 (other-layout for every failure start and 40–47; shared panel) | same |
| `guidance_time` | as `guidance` | donor window at the same control index |
| `guidance_random` | none | uniform index among the same K |
| `guidance_same` | the same-layout donor (14, 15: Panda; 36, 37: IIWA) | upper reference only |

All draw the same three extra noises from the harness candidate stream, so branches of a start see the same K at the opening replan.

**CPU analyses** (`analyze.py`, written to `evidence/`):

1. Leave-layout-out prediction on source successes: error of the progress-aligned, time-aligned and random donor window against the query's actual next five controls. Paired difference per query episode, bootstrap over query layouts. Six episodes on six layouts, so intervals are wide.
2. Reset frame against rotating tool frame on the Panda pair. Jaw command sign against aperture on Panda.
3. Archived Kinova3 raw tapes 34–37 (two successes, two failures) as queries against other-layout source donors: distance to the aligned donor by phase, and the phase each failure stalls in. Archived tapes hold one executed proposal per replan, so this cannot show whether K=4 samples contain a nearer one.

**Tests** (`test_guidance.py`): fixture rollouts through the shared harness, alignment on synthetic time-warped tapes, donor digests, layout exclusion.

**Native** (`run.sh`, not run here): development starts × {raw, repeat, guidance, guidance_time, guidance_random, guidance_same where a donor exists} on one RTX 4090.

Evidence kinds: tests are CPU fixture; analyses are offline readings of archived native tapes. Neither is task success.
